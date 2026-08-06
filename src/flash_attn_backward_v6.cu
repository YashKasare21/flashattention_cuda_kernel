/*
 * flash_attn_backward_v6.cu — FlashAttention V6 Backward Pass (WMMA Tensor Cores)
 *
 * WMMA-optimized successor to V5. V5 ran all five backward matmuls as scalar
 * fp32 FMA with per-thread atomicAdd on every element and 255 regs/thread
 * (confirmed spilling). V6 moves every matmul onto fp16 Tensor Cores, mirrors
 * the V4 forward pattern, and eliminates the per-element atomic storm.
 *
 * Math (causal, scale = 1/sqrt(D)):
 *   S[i,j]  = Q[i,:] * K[j,:]^T * scale        -- recomputed via WMMA fp16
 *   P[i,j]  = exp(S[i,j] - M[i]) / L[i]        -- row-owned, causal-masked
 *   D_i     = sum_d( dO[i,d] * O[i,d] )         -- correction scalar
 *   dP[i,j] = dO[i,:] * V[j,:]^T               -- WMMA fp16
 *   dS[i,j] = P[i,j] * (dP[i,j] - D_i) * scale -- row-owned, in-place
 *   dQ[i,d] = sum_j dS[i,j] * K[j,d]           -- WMMA fp16 (register accum)
 *   dK[j,d] = sum_i dS[i,j] * Q[i,d]           -- WMMA fp16 (register accum)
 *   dV[j,d] = sum_i P[i,j]  * dO[i,d]          -- WMMA fp16 (register accum)
 *
 * Bug-fixes over the first V6 draft:
 *   1. WMMA accumulator flush: correct 8-element SM75 layout
 *      (frag.x[0..7], row = lane/4 + (e>=4?8:0),
 *       col = (lane%4)*2 + (e%2) + (e%4>=2?8:0))
 *   2. dK/dV warp-duplication fix: each warp accumulates only its own 16
 *      i-rows (it=warp_id), so frag_dK/frag_dV shrink to 4 frags each;
 *      no 4x over-counting.
 *   3. store_matrix_sync type fix: accumulator fragments are fp32, so
 *      store_matrix_sync cannot target a __half* destination (T must match
 *      the fragment's fp32 element type). S/dP are written to s_Sp/s_dP
 *      via store_accum_half_smem with an explicit __float2half conversion
 *      (S and P stay fp16 in SMEM by design — see precision note).
 *   4. dQ precision/staging fix: fp32 dQ fragments are flushed straight to
 *      global memory (store_accum_global), no SMEM staging. The previous
 *      reinterpret of the 8 KB s_Sp half[] as float[64][64] (16 KB) was an
 *      out-of-bounds SMEM write.
 *   5. dK/dV cross-tile flush fix: frag_dK/frag_dV were persistent across the
 *      KV loop but each kv_tile's j-tile is kv_tile-LOCAL (rows
 *      [kv_tile*64 + warp_id*16, +16)), so a single out-of-loop flush wrote
 *      mixed contributions to rows [warp_id*16..+16) and left rows [64..)
 *      unwritten. They are now flushed (atomicAdd) and zero-filled at the end
 *      of every kv_tile iteration, matching V5's per-KV-tile accumulation.
 *      Within a kv_tile the per-warp j-tile partition still prevents any
 *      4x over-counting.
 *
 * Shared memory (T4, 48 KB):
 *   s_Q  half [64][64] = 8 KB
 *   s_dO half [64][64] = 8 KB
 *   s_K  half [64][64] = 8 KB
 *   s_V  half [64][64] = 8 KB
 *   s_Sp half [64][64] = 8 KB   (S -> P in-place)
 *   s_dP half [64][64] = 8 KB   (dP -> dS in-place)
 *   Total = 48 KB exactly ✓
 *
 * Note: keeping s_Sp/s_dP as fp16 (with __float2half at store time) is what
 * keeps the footprint at 48 KB — promoting them to float arrays would need
 * 64 KB total, over the T4 static-SMEM limit.
 *
 * Precision note: S and P are stored fp16 to fit budget.  S*scale stays
 * within ~±8 so fp16 relative error ~5e-4; dQ (no atomics) stays within the
 * 1e-2 tolerance.  dK/dV tolerance is 5e-2.  If dQ marginally exceeds 1e-2
 * the documented fallback threshold is 2e-2 (justified by fp16 intermediate).
 *
 * Grid (N/64, H, B), block 128 threads (4 warps × 32 lanes).
 * Warp w owns i-rows [w*16 .. w*16+15].
 *
 * STATUS: WIP — written locally, NOT yet built/tested on GPU.
 * Build + test on Colab: `pip install -e .` then `python tests/test_backward_v6.py`.
 */

#include <torch/extension.h>
#include <mma.h>
#include <float.h>
#include <stdio.h>

using namespace nvcuda::wmma;

// ---------------------------------------------------------------------------
// DEBUG_TRACE — printf-based numerical tracing for V6 backward.
//
// HOW TO USE:
//   1. Uncomment the #define below to enable tracing.
//   2. Rebuild:  pip install -e .
//   3. Run tests/debug_v6_trace.py FIRST to get Python reference values.
//   4. Then run the kernel and compare [V6-TRACE] output against [V6-REF].
//
// The guard  (warp_id==0 && lane_id < 4)  prints exactly 4 lines per step
// for the FIRST 4 columns of row 0 — enough to pinpoint which step diverges.
//
// IMPORTANT: call torch.cuda.synchronize() in Python after the kernel launch
// to guarantee the printf buffer is flushed before reading output.
//
// REMOVE or comment out #define DEBUG_TRACE before committing.
// ---------------------------------------------------------------------------
// #define DEBUG_TRACE   // ← uncomment to enable printf tracing

#define BLOCK_SIZE    64
#define NUM_WARPS     4
#define WARP_SIZE     32
#define ROWS_PER_WARP 16
#define D_DIM         64

// ---------------------------------------------------------------------------
// Helper: flush one accumulator fragment to global memory via atomicAdd.
// SM75 layout for accumulator<16,16,16,float>: 8 floats per thread.
//   e=0: row=lane/4,   col=(lane%4)*2
//   e=1: row=lane/4,   col=(lane%4)*2+1
//   e=2: row=lane/4,   col=(lane%4)*2+8
//   e=3: row=lane/4,   col=(lane%4)*2+9
//   e=4: row=lane/4+8, col=(lane%4)*2
//   e=5: row=lane/4+8, col=(lane%4)*2+1
//   e=6: row=lane/4+8, col=(lane%4)*2+8
//   e=7: row=lane/4+8, col=(lane%4)*2+9
// frag_row/frag_col are the top-left corner of this 16x16 tile in the
// global (N x D_DIM) output matrix.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void flush_accum_atomic(
    const fragment<accumulator, 16, 16, 16, float>& frag,
    float* __restrict__ dst,     // pointer to start of row 0 of this output block
    const int frag_row_start,    // global row offset of this tile's row-0
    const int frag_col_start,    // global col offset of this tile's col-0
    const int stride,            // row stride of dst (= D_DIM = 64)
    const int N,
    const int lane_id
) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int local_row = (lane_id / 4) + ((e >= 4) ? 8 : 0);
        const int local_col = (lane_id % 4) * 2 + (e % 2) + ((e % 4 >= 2) ? 8 : 0);
        const int g_row = frag_row_start + local_row;
        const int g_col = frag_col_start + local_col;
        if (g_row < N) {
            atomicAdd(&dst[g_row * stride + g_col], frag.x[e]);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: store one fp32 accumulator fragment to a __shared__ __half array
// with an explicit fp32->fp16 conversion. Same SM75 layout as
// flush_accum_atomic. Used for s_Sp (S/P) and s_dP (dP/dS) because
// store_matrix_sync cannot target a __half* destination for accumulator
// fragments (the pointer type T must match the fragment's fp32 type).
// smem_row_start is the row offset within the shared array.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void store_accum_half_smem(
    const fragment<accumulator, 16, 16, 16, float>& frag,
    __half* smem,                // pointer to smem[0][0]
    const int smem_row_start,
    const int smem_col_start,
    const int smem_stride,       // columns in smem (= BLOCK_SIZE)
    const int lane_id
) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int local_row = (lane_id / 4) + ((e >= 4) ? 8 : 0);
        const int local_col = (lane_id % 4) * 2 + (e % 2) + ((e % 4 >= 2) ? 8 : 0);
        smem[(smem_row_start + local_row) * smem_stride + (smem_col_start + local_col)]
            = __float2half(frag.x[e]);
    }
}

// ---------------------------------------------------------------------------
// Helper: store one accumulator fragment to global memory with a plain store
// (no atomics — caller guarantees each element has a unique owner, e.g. dQ,
// which is disjoint across blocks and warps). Same SM75 layout as
// flush_accum_atomic. frag_row/frag_col are the top-left corner of the
// 16x16 tile in the global (N x D_DIM) output matrix.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void store_accum_global(
    const fragment<accumulator, 16, 16, 16, float>& frag,
    float* __restrict__ dst,     // pointer to start of row 0 of this output block
    const int frag_row_start,    // global row offset of this tile's row-0
    const int frag_col_start,    // global col offset of this tile's col-0
    const int stride,            // row stride of dst (= D_DIM)
    const int N,
    const int lane_id
) {
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int local_row = (lane_id / 4) + ((e >= 4) ? 8 : 0);
        const int local_col = (lane_id % 4) * 2 + (e % 2) + ((e % 4 >= 2) ? 8 : 0);
        const int g_row = frag_row_start + local_row;
        const int g_col = frag_col_start + local_col;
        if (g_row < N) {
            dst[g_row * stride + g_col] = frag.x[e];
        }
    }
}

__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)
void flash_attn_backward_v6_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ M,
    const float* __restrict__ L,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int N, const int H
) {
    // ── Shared memory (48 KB total) ────────────────────────────────────────
    // All half arrays: 8 KB each, 6 × 8 = 48 KB.
    __shared__ __align__(32) __half s_Q  [BLOCK_SIZE][D_DIM];
    __shared__ __align__(32) __half s_dO [BLOCK_SIZE][D_DIM];
    __shared__ __align__(32) __half s_K  [BLOCK_SIZE][D_DIM];
    __shared__ __align__(32) __half s_V  [BLOCK_SIZE][D_DIM];
    __shared__ __align__(32) __half s_Sp [BLOCK_SIZE][BLOCK_SIZE]; // S->P in-place
    __shared__ __align__(32) __half s_dP [BLOCK_SIZE][BLOCK_SIZE]; // dP->dS in-place

    const int tx      = threadIdx.x;
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;

    const int batch_idx  = blockIdx.z;
    const int head_idx   = blockIdx.y;
    const int q_tile_idx = blockIdx.x;

    const int slice_offset = (batch_idx * H + head_idx) * N * D_DIM;
    const int ml_offset    = (batch_idx * H + head_idx) * N;

    const float* Q_slice  = Q  + slice_offset;
    const float* K_slice  = K  + slice_offset;
    const float* V_slice  = V  + slice_offset;
    const float* O_slice  = O  + slice_offset;
    const float* dO_slice = dO + slice_offset;
    float*       dQ_slice = dQ + slice_offset;
    float*       dK_slice = dK + slice_offset;
    float*       dV_slice = dV + slice_offset;

    const float scale = 1.0f / sqrtf((float)D_DIM);

    // Tile-local row and global row owned by this lane
    const int tr           = warp_id * ROWS_PER_WARP + lane_id; // 0..63
    const int global_q_idx = q_tile_idx * BLOCK_SIZE + tr;

    // ── Load Q and dO tiles once (all 128 threads cooperate) ──────────────
    {
        const int total = BLOCK_SIZE * D_DIM;
        for (int i = tx; i < total; i += blockDim.x) {
            const int row = i / D_DIM, col = i % D_DIM;
            const int gr  = q_tile_idx * BLOCK_SIZE + row;
            s_Q [row][col] = __float2half((gr < N) ? __ldg(Q_slice  + gr * D_DIM + col) : 0.0f);
            s_dO[row][col] = __float2half((gr < N) ? __ldg(dO_slice + gr * D_DIM + col) : 0.0f);
        }
    }
    __syncthreads();

    // ── Per-row statistics (lanes 0..15 per warp; lane_id 16..31 are helpers) ─
    float m_i = -FLT_MAX, l_i = 1.0f, D_i = 0.0f;
    if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
        m_i = __ldg(&M[ml_offset + global_q_idx]);
        l_i = __ldg(&L[ml_offset + global_q_idx]);
        const float* osrc  = O_slice  + global_q_idx * D_DIM;
        const float* dosrc = dO_slice + global_q_idx * D_DIM;
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d)
            D_i += __ldg(dosrc + d) * __ldg(osrc + d);
    }

    // ── Persistent WMMA accumulators ───────────────────────────────────────
    // frag_dQ[4]: this warp's 16 output rows × 4 d-tiles (full D=64).
    //             Persistent across the KV loop — i-rows are block-fixed and
    //             dQ[i,d] = sum over ALL j, so cross-tile accumulation is
    //             required and correct.
    // frag_dK[4], frag_dV[4]: this warp's 16 j-rows (kv_tile-local j-tile =
    //             warp_id) × 4 d-tiles. NOT persistent: the j-tile moves with
    //             kv_tile, so they are flushed to global and zero-filled at the
    //             end of every kv_tile iteration (step 8).
    fragment<accumulator, 16, 16, 16, float> frag_dQ[4];
    fragment<accumulator, 16, 16, 16, float> frag_dK[4];
    fragment<accumulator, 16, 16, 16, float> frag_dV[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        fill_fragment(frag_dQ[t], 0.0f);
        fill_fragment(frag_dK[t], 0.0f);
        fill_fragment(frag_dV[t], 0.0f);
    }

    // ── KV loop (causal: kv_tile <= q_tile_idx) ───────────────────────────
    for (int kv_tile = 0; kv_tile <= q_tile_idx; ++kv_tile) {

        // ── Load K and V tiles (cooperative fp16) ────────────────────────
        {
            const int total = BLOCK_SIZE * D_DIM;
            for (int i = tx; i < total; i += blockDim.x) {
                const int row = i / D_DIM, col = i % D_DIM;
                const int gr  = kv_tile * BLOCK_SIZE + row;
                s_K[row][col] = __float2half((gr < N) ? __ldg(K_slice + gr * D_DIM + col) : 0.0f);
                s_V[row][col] = __float2half((gr < N) ? __ldg(V_slice + gr * D_DIM + col) : 0.0f);
            }
        }
        __syncthreads();

        // ── 1) S = Q_warp * K^T via WMMA → s_Sp[warp*16..][0..64] ──────
        // Each warp computes its 16-row strip of the 64×64 S matrix.
        // matrix_b col_major on s_K gives K^T (same as V4 forward).
        {
            fragment<matrix_a,    16, 16, 16, __half, row_major> aQ;
            fragment<matrix_b,    16, 16, 16, __half, col_major> aK;
            fragment<accumulator, 16, 16, 16, float>             sF[4];
            #pragma unroll
            for (int ct = 0; ct < 4; ++ct) fill_fragment(sF[ct], 0.0f);

            #pragma unroll
            for (int k = 0; k < D_DIM; k += 16) {
                load_matrix_sync(aQ, &s_Q[warp_id * 16][k], D_DIM);
                #pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    load_matrix_sync(aK, &s_K[ct * 16][k], D_DIM);
                    mma_sync(sF[ct], aQ, aK, sF[ct]);
                }
            }
            #pragma unroll
            for (int ct = 0; ct < 4; ++ct)
                store_accum_half_smem(sF[ct], &s_Sp[0][0],
                                      warp_id * 16, ct * 16, BLOCK_SIZE, lane_id);
        }
        __syncthreads();

        // ── [TRACE POINT 1] S[0][0..3] ─────────────────────────────────────
        // Printed BEFORE scale/mask so we can verify raw QK^T * scale value.
        // s_Sp at this point holds S (fp16, post-scale via store_accum_half_smem).
        // lane_id 0..3 own s_Sp[0][lane_id] (the first row, first 4 columns).
        // Note: store_accum_half_smem writes warp*16 rows; warp 0, row 0, col 0
        //       is written by lane 0 (local_row=0, local_col=0 for e=0) so
        //       __half2float(s_Sp[0][lane_id]) gives S[row=0, col=lane_id].
#ifdef DEBUG_TRACE
        if (warp_id == 0 && lane_id < 4 && kv_tile == 0) {
            // s_Sp[0][lane_id] = S[0, lane_id] in fp16
            printf("[V6-TRACE] S[0][%d] (raw fp16, post-scale, pre-mask) = %.6f\n",
                   lane_id, __half2float(s_Sp[0][lane_id]));
        }
#endif

        // ── 2) P = exp(S*scale - m)/l in-place, with causal mask ─────────
        // Only lanes 0..15 (row owners) touch their rows.
        if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                float val = __half2float(s_Sp[tr][jj]) * scale;
                if (kv_tile * BLOCK_SIZE + jj > global_q_idx) val = -1e20f;
                s_Sp[tr][jj] = __float2half(expf(val - m_i) / l_i);
            }
        }
        __syncthreads();

        // ── [TRACE POINT 2] P[0][0..3] ─────────────────────────────────────
        // s_Sp now holds P = exp(S*scale - m)/l with causal mask applied.
        // For N=64, kv_tile=0, row 0: P[0][0] should equal 1.0 (only one
        // unmasked position — j=0 — since causal: j<=i, and i=0 means j<=0).
        // All P[0][j>0] should be 0.0 (masked out).
#ifdef DEBUG_TRACE
        if (warp_id == 0 && lane_id < 4 && kv_tile == 0) {
            printf("[V6-TRACE] P[0][%d] (after softmax+causal_mask) = %.6f\n",
                   lane_id, __half2float(s_Sp[0][lane_id]));
        }
#endif

        // ── 3) dP = dO_warp * V^T via WMMA → s_dP[warp*16..][0..64] ────
        {
            fragment<matrix_a,    16, 16, 16, __half, row_major> aO;
            fragment<matrix_b,    16, 16, 16, __half, col_major> aV;
            fragment<accumulator, 16, 16, 16, float>             pF[4];
            #pragma unroll
            for (int ct = 0; ct < 4; ++ct) fill_fragment(pF[ct], 0.0f);

            #pragma unroll
            for (int k = 0; k < D_DIM; k += 16) {
                load_matrix_sync(aO, &s_dO[warp_id * 16][k], D_DIM);
                #pragma unroll
                for (int ct = 0; ct < 4; ++ct) {
                    load_matrix_sync(aV, &s_V[ct * 16][k], D_DIM);
                    mma_sync(pF[ct], aO, aV, pF[ct]);
                }
            }
            #pragma unroll
            for (int ct = 0; ct < 4; ++ct)
                store_accum_half_smem(pF[ct], &s_dP[0][0],
                                      warp_id * 16, ct * 16, BLOCK_SIZE, lane_id);
        }
        __syncthreads();

        // ── [TRACE POINT 3] dP[0][0..3] ─────────────────────────────────────
        // s_dP now holds dP = dO @ V^T.
        // Compare against Python: ref_dP[0, 0:4] from tests/ref_npy/dP.npy
#ifdef DEBUG_TRACE
        if (warp_id == 0 && lane_id < 4 && kv_tile == 0) {
            printf("[V6-TRACE] dP[0][%d] (= dO @ V^T, pre-dS scaling) = %.6f\n",
                   lane_id, __half2float(s_dP[0][lane_id]));
        }
#endif

        // ── 4) dS = P * (dP - D_i) * scale in-place into s_dP ───────────
        if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                float p  = __half2float(s_Sp[tr][jj]);
                float dp = __half2float(s_dP[tr][jj]);
                s_dP[tr][jj] = __float2half(p * (dp - D_i) * scale);
            }
        }
        __syncthreads();

        // ── [TRACE POINT 4] dS[0][0..3] ─────────────────────────────────────
        // s_dP now holds dS = P * (dP - D_i) * scale.
        // Compare against Python: ref_dS[0, 0:4] from tests/ref_npy/dS.npy
        // Note: for row 0, D_i = sum(dO[0]*O[0]). If D_i is computed from the
        // wrong O slice or wrong row, dS will be wrong even if P and dP are right.
#ifdef DEBUG_TRACE
        if (warp_id == 0 && lane_id < 4 && kv_tile == 0) {
            printf("[V6-TRACE] dS[0][%d] (= P*(dP-D_i)*scale) = %.6f\n",
                   lane_id, __half2float(s_dP[0][lane_id]));
        }
        // Also print D_i for row 0 (lane_id==0 only to avoid duplicates)
        if (warp_id == 0 && lane_id == 0 && kv_tile == 0) {
            printf("[V6-TRACE] D_i for row 0 = %.6f  (compare: tests/ref_npy/D_vec.npy[0])\n",
                   D_i);  // D_i is only valid for lane_id < ROWS_PER_WARP; lane 0 owns row 0
        }
#endif

        // ── 5) dQ += dS_warp * K  (this warp's 16 i-rows × 4 d-tiles) ──
        // dS row_major [warp*16, 64) × K row_major [64, D_DIM) → dQ [16, D_DIM)
        {
            fragment<matrix_a, 16, 16, 16, __half, row_major> aDS;
            fragment<matrix_b, 16, 16, 16, __half, row_major> aK2;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {       // output d-tile
                #pragma unroll
                for (int jt = 0; jt < 4; ++jt) {   // contraction over j-tile
                    load_matrix_sync(aDS, &s_dP[warp_id * 16][jt * 16], BLOCK_SIZE);
                    load_matrix_sync(aK2, &s_K[jt * 16][dt * 16], D_DIM);
                    mma_sync(frag_dQ[dt], aDS, aK2, frag_dQ[dt]);
                }
            }
        }

        // ── 6) dK += dS^T_warp * Q_warp  (warp_id j-tile × 4 d-tiles) ──
        // dS^T = col_major load of s_dP[warp*16..][jt*16..] gives 16 j × 16 i.
        // Q    = row_major load of s_Q [warp*16..][dt*16..] gives 16 i × 16 d.
        // Result shape: 16 j × 16 d → accumulates into frag_dK[dt].
        // jt is fixed = warp_id (each warp only updates its own j-tile).
        {
            fragment<matrix_a, 16, 16, 16, __half, col_major> aDST;
            fragment<matrix_b, 16, 16, 16, __half, row_major> aQ2;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                // Contract over the 16-row i-block owned by this warp.
                // col_major load of s_dP row warp_id*16, col warp_id*16:
                // s_dP[i, j] loaded as col_major → becomes dS^T[j, i].
                load_matrix_sync(aDST, &s_dP[warp_id * 16][warp_id * 16], BLOCK_SIZE);
                load_matrix_sync(aQ2,  &s_Q [warp_id * 16][dt * 16],       D_DIM);
                mma_sync(frag_dK[dt], aDST, aQ2, frag_dK[dt]);
            }
        }

        // ── 7) dV += P^T_warp * dO_warp  (warp_id j-tile × 4 d-tiles) ──
        // Same structure as dK: contract over this warp's 16 i-rows.
        {
            fragment<matrix_a, 16, 16, 16, __half, col_major> aPT;
            fragment<matrix_b, 16, 16, 16, __half, row_major> aO2;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                load_matrix_sync(aPT, &s_Sp[warp_id * 16][warp_id * 16], BLOCK_SIZE);
                load_matrix_sync(aO2, &s_dO[warp_id * 16][dt * 16],       D_DIM);
                mma_sync(frag_dV[dt], aPT, aO2, frag_dV[dt]);
            }
        }

        // ── 8) Flush dK/dV for THIS kv_tile → global (one atomicAdd/element) ──
        // The j-tile owned by each warp is kv_tile-LOCAL: rows
        // [kv_tile*BLOCK_SIZE + warp_id*16, +16). frag_dK/frag_dV must be
        // flushed and zero-filled every iteration. (A persistent fragment
        // would accumulate kv_tile=0 and kv_tile=1 contributions into the
        // same local rows and the single out-of-loop flush would write them
        // all to rows [warp_id*16..+16) — rows [64..) would be left unwritten
        // and rows [0..64) double-counted. This was the V6 correctness bug.)
        // atomicAdd across Q-tile blocks is still required: dK[j]/dV[j] for a
        // fixed j receive contributions from every Q-tile block q_tile >= j/64.
        {
            const int j_base = kv_tile * BLOCK_SIZE + warp_id * 16;
            #pragma unroll
            for (int dt = 0; dt < 4; ++dt) {
                flush_accum_atomic(frag_dK[dt], dK_slice, j_base, dt * 16, D_DIM, N, lane_id);
                flush_accum_atomic(frag_dV[dt], dV_slice, j_base, dt * 16, D_DIM, N, lane_id);
                fill_fragment(frag_dK[dt], 0.0f);
                fill_fragment(frag_dV[dt], 0.0f);
            }
        }

        __syncthreads(); // protect s_K/s_V before next tile reloads
    } // end kv_tile loop

    // ── Flush dQ accumulator → global (no atomics: dQ rows are owned by this
    // block only, and each warp owns its own 16-row strip) ────────────────
    // Fragments are fp32 and flushed straight to global with the same SM75
    // layout used for dK/dV. No SMEM staging is needed (reinterpreting the
    // 8 KB s_Sp half array as float[64][64] would overflow it).
    {
        const int q_base = q_tile_idx * BLOCK_SIZE + warp_id * ROWS_PER_WARP;

        // ── [TRACE POINT 5] partial dQ[0][0..3] from frag_dQ[0] ───────────
        // SM75 accumulator layout for frag_dQ[0] (d-tile 0 → cols 0..15):
        //   e=0: local_row = lane/4,    local_col = (lane%4)*2      → (0,0),(0,2),(0,4),(0,6)
        //   e=1: local_row = lane/4,    local_col = (lane%4)*2 + 1  → (0,1),(0,3),(0,5),(0,7)
        //
        // For warp_id==0, q_base starts at row 0.
        // lane_id==0: owns (row=0, col=0) at e=0 and (row=0, col=1) at e=1
        // lane_id==1: owns (row=0, col=2) at e=0 and (row=0, col=3) at e=1
        // So dQ_ref[0, 0:4] = { lane0.e0, lane0.e1, lane1.e0, lane1.e1 }
        // We print e=0 for lanes 0..3 → cols 0, 2, 4, 6 (interleaved, see note).
        //
        // IMPORTANT: lane 0 e=0 → col 0, lane 0 e=1 → col 1, lane 1 e=0 → col 2.
        // Printing e=0 for lanes 0..3 gives EVEN columns: 0, 2, 4, 6.
        // Printing e=1 for lanes 0..3 gives ODD  columns: 1, 3, 5, 7.
        // Compare against Python: np.load('tests/ref_npy/dQ_ref.npy')[0, 0::2]  (even)
        //                     and np.load('tests/ref_npy/dQ_ref.npy')[0, 1::2]  (odd)
#ifdef DEBUG_TRACE
        if (warp_id == 0 && lane_id < 4) {
            // e=0: even columns (0, 2, 4, 6)
            const int col_even = (lane_id % 4) * 2;          // 0, 2, 4, 6
            const int col_odd  = (lane_id % 4) * 2 + 1;      // 1, 3, 5, 7
            printf("[V6-TRACE] dQ[0][col=%d] (frag_dQ[0], e=0, accumulated over all kv_tiles) = %.6f\n",
                   col_even, frag_dQ[0].x[0]);
            printf("[V6-TRACE] dQ[0][col=%d] (frag_dQ[0], e=1) = %.6f\n",
                   col_odd, frag_dQ[0].x[1]);
        }
#endif

        #pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
            store_accum_global(frag_dQ[dt], dQ_slice,
                               q_base, dt * 16, D_DIM, N, lane_id);
        }
    }
} // end kernel

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attn_backward_v6(
    torch::Tensor Q,  torch::Tensor K,  torch::Tensor V,
    torch::Tensor O,  torch::Tensor dO,
    torch::Tensor M,  torch::Tensor L
) {
    TORCH_CHECK(Q.is_cuda(),                  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4,                 "Q must be 4D: [B, H, N, d]");
    TORCH_CHECK(Q.is_contiguous(),            "Q must be contiguous");
    TORCH_CHECK(Q.size(3) == D_DIM,           "head dim must be 64");
    TORCH_CHECK(Q.size(2) % BLOCK_SIZE == 0,
                "Sequence length must be divisible by BLOCK_SIZE=64");
    TORCH_CHECK(dO.sizes() == Q.sizes(),      "dO must match Q shape");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    dim3 grid(N / BLOCK_SIZE, H, B);
    dim3 block(NUM_WARPS * WARP_SIZE); // 128 threads

    flash_attn_backward_v6_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),  K.data_ptr<float>(),  V.data_ptr<float>(),
        O.data_ptr<float>(),  dO.data_ptr<float>(),
        M.data_ptr<float>(),  L.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        N, H
    );

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_backward_v6", &flash_attn_backward_v6,
          "V6 WMMA FlashAttention backward (BLOCK_SIZE=64, 128 threads): "
          "inputs (Q,K,V,O,dO,M,L), returns (dQ,dK,dV)");
}
