/*
 * flash_attn_backward_v5.cu — FlashAttention V5 Backward Pass
 *
 * Matches V4 forward: BLOCK_SIZE=64, 128 threads (4 warps × 32 lanes).
 * Recomputes P on-the-fly from saved (M, L) — O(N) memory, not O(N²).
 *
 * Math (causal, scale = 1/sqrt(D)):
 *   S[i,j]  = dot(Q[i], K[j]) * scale          -- recomputed
 *   P[i,j]  = exp(S[i,j] - M[i]) / L[i]        -- recomputed from saved stats
 *   D_i     = sum_d( dO[i,d] * O[i,d] )         -- correction scalar
 *   dV[j]  += P[i,j] * dO[i]
 *   dP[i,j] = dot(dO[i], V[j])
 *   dS[i,j] = P[i,j] * (dP[i,j] - D_i) * scale   -- scale propagated back through S=QK^T*scale
 *   dQ[i]  += dS[i,j] * K[j]                    -- local, no atomics
 *   dK[j]  += dS[i,j] * Q[i]                    -- atomicAdd (multiple Q-tiles)
 *
 * Grid: dim3(N/64, H, B) — mirrors V4 forward.
 * Block: 128 threads (4 warps). Each warp owns 16 rows of the Q-tile.
 *
 * Shared memory (D=64):
 *   s_Q  [64][64] float = 16 KB  — loaded once, fixed for all KV tiles
 *   s_dO [64][64] float = 16 KB  — loaded once, fixed for all KV tiles
 *   s_K  [64][64] float = 16 KB  — reloaded each KV tile
 *   s_V  [64][64] float = 16 KB  — reloaded each KV tile
 *   Total: 64 KB — exceeds 48 KB T4 limit.
 *
 * To fit within 48 KB: process Q/dO in two half-tile passes (32 rows each),
 * keeping s_K and s_V resident. Each pass uses:
 *   s_Q_half  [32][64] float = 8 KB
 *   s_dO_half [32][64] float = 8 KB
 *   s_K       [64][64] float = 16 KB
 *   s_V       [64][64] float = 16 KB
 *   Total: 48 KB exactly ✓
 *
 * Alternatively: use a single 64-row shared buffer for K/V and stream Q/dO
 * from registers. Since each thread owns exactly one row of Q/dO (its global
 * row), we can keep Q[i] and dO[i] in registers (D=64 floats each = 128 regs
 * per thread). This avoids the SMEM pressure entirely:
 *
 *   Registers: Q_reg[64], dO_reg[64] per thread (128 floats = 512 bytes/thread)
 *   SMEM: s_K[64][64] + s_V[64][64] = 32 KB ✓
 *
 * This is the approach used below. Each thread (one per Q-row in its warp's
 * 16-row strip) holds its own Q and dO rows in registers throughout the KV loop.
 */

#include <torch/extension.h>
#include <float.h>

#define BLOCK_SIZE    64
#define NUM_WARPS     4
#define WARP_SIZE     32
#define ROWS_PER_WARP 16   // BLOCK_SIZE / NUM_WARPS
#define D_DIM         64

__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)
void flash_attn_backward_v5_kernel(
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
    // ── Shared memory: K and V tiles only (32 KB) ─────────────────────────────
    __shared__ float s_K[BLOCK_SIZE][D_DIM];
    __shared__ float s_V[BLOCK_SIZE][D_DIM];

    const int tx      = threadIdx.x;          // 0..127
    const int warp_id = tx / WARP_SIZE;        // 0..3
    const int lane_id = tx % WARP_SIZE;        // 0..31

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

    // ── Each lane in lanes 0..15 owns one Q-row in its warp's strip ──────────
    // lane_id 16..31 are helpers for the cooperative K/V load but don't write output.
    // We use lane_id < ROWS_PER_WARP to gate output writes.
    const int tr      = warp_id * ROWS_PER_WARP + lane_id;  // tile-local row (0..63)
    const int global_q_idx = q_tile_idx * BLOCK_SIZE + tr;   // global row index

    // ── Load Q[i] and dO[i] into registers (valid for lane_id < ROWS_PER_WARP) ─
    float Q_reg [D_DIM];
    float dO_reg[D_DIM];
    float O_reg [D_DIM];

    if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
        const float* qsrc  = Q_slice  + global_q_idx * D_DIM;
        const float* dosrc = dO_slice + global_q_idx * D_DIM;
        const float* osrc  = O_slice  + global_q_idx * D_DIM;
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d) {
            Q_reg [d] = __ldg(qsrc  + d);
            dO_reg[d] = __ldg(dosrc + d);
            O_reg [d] = __ldg(osrc  + d);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d) {
            Q_reg [d] = 0.0f;
            dO_reg[d] = 0.0f;
            O_reg [d] = 0.0f;
        }
    }

    // ── Load saved forward statistics ─────────────────────────────────────────
    const float m_i = (lane_id < ROWS_PER_WARP && global_q_idx < N)
                      ? __ldg(&M[ml_offset + global_q_idx]) : 0.0f;
    const float l_i = (lane_id < ROWS_PER_WARP && global_q_idx < N)
                      ? __ldg(&L[ml_offset + global_q_idx]) : 1.0f;

    // ── D_i = sum_d( dO[i,d] * O[i,d] ) ─────────────────────────────────────
    float D_i = 0.0f;
    if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d)
            D_i += dO_reg[d] * O_reg[d];
    }

    // ── dQ accumulator (register, no atomics) ─────────────────────────────────
    float dQ_acc[D_DIM];
    #pragma unroll
    for (int d = 0; d < D_DIM; ++d) dQ_acc[d] = 0.0f;

    // ── KV loop ───────────────────────────────────────────────────────────────
    for (int kv_tile = 0; kv_tile <= q_tile_idx; ++kv_tile) {

        // Cooperative load of s_K and s_V (all 128 threads)
        {
            const int total = BLOCK_SIZE * D_DIM;
            for (int i = tx; i < total; i += blockDim.x) {
                const int row = i / D_DIM;
                const int col = i % D_DIM;
                const int gr  = kv_tile * BLOCK_SIZE + row;
                s_K[row][col] = (gr < N) ? __ldg(K_slice + gr * D_DIM + col) : 0.0f;
                s_V[row][col] = (gr < N) ? __ldg(V_slice + gr * D_DIM + col) : 0.0f;
            }
        }
        __syncthreads();

        // Each active lane (lane_id < ROWS_PER_WARP) processes its Q-row
        // against all 64 KV positions in this tile.
        if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                const int actual_kv = kv_tile * BLOCK_SIZE + jj;

                // Recompute S[i, jj]
                float S_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < D_DIM; ++d)
                    S_ij += Q_reg[d] * s_K[jj][d];
                S_ij *= scale;

                // Causal mask
                if (actual_kv > global_q_idx) S_ij = -1e20f;

                // Recompute P[i, jj]
                const float P_ij = expf(S_ij - m_i) / l_i;

                // dV[jj] += P_ij * dO[i]
                if (actual_kv < N) {
                    #pragma unroll
                    for (int d = 0; d < D_DIM; ++d)
                        atomicAdd(&dV_slice[actual_kv * D_DIM + d], P_ij * dO_reg[d]);
                }

                // dP[i, jj] = dot(dO[i], V[jj])
                float dP_ij = 0.0f;
                #pragma unroll
                for (int d = 0; d < D_DIM; ++d)
                    dP_ij += dO_reg[d] * s_V[jj][d];

                // dS[i, jj] = P_ij * (dP_ij - D_i)
                // dQ[i] += dS_ij * dS/dQ = dS_ij * K[jj] * scale
                // dK[jj] += dS_ij * dS/dK = dS_ij * Q[i] * scale
                const float dS_ij = P_ij * (dP_ij - D_i) * scale;

                // dQ[i] += dS_ij * K[jj]  (local, no atomics)
                #pragma unroll
                for (int d = 0; d < D_DIM; ++d)
                    dQ_acc[d] += dS_ij * s_K[jj][d];

                // dK[jj] += dS_ij * Q[i]  (atomicAdd: multiple Q-tiles write same jj)
                if (actual_kv < N) {
                    #pragma unroll
                    for (int d = 0; d < D_DIM; ++d)
                        atomicAdd(&dK_slice[actual_kv * D_DIM + d], dS_ij * Q_reg[d]);
                }
            }
        }

        __syncthreads();  // done with s_K/s_V before next tile reloads them
    }

    // ── Write dQ ──────────────────────────────────────────────────────────────
    if (lane_id < ROWS_PER_WARP && global_q_idx < N) {
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d)
            dQ_slice[global_q_idx * D_DIM + d] = dQ_acc[d];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python binding
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attn_backward_v5(
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
    dim3 block(NUM_WARPS * WARP_SIZE);  // 128 threads

    flash_attn_backward_v5_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),  K.data_ptr<float>(),  V.data_ptr<float>(),
        O.data_ptr<float>(),  dO.data_ptr<float>(),
        M.data_ptr<float>(),  L.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        N, H
    );

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_backward_v5", &flash_attn_backward_v5,
          "V5 FlashAttention backward (BLOCK_SIZE=64, 128 threads): "
          "inputs (Q,K,V,O,dO,M,L), returns (dQ,dK,dV)");
}
