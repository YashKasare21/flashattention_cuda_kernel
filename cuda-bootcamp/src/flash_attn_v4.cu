/*
 * flash_attn_v4.cu — FlashAttention V4: Multi-Warp Block Design
 *
 * Key upgrade over V3:
 *   V3: 1 warp (32 threads) per block, BLOCK_SIZE=32, ~20KB SMEM
 *   V4: 4 warps (128 threads) per block, BLOCK_SIZE=64, ~48KB SMEM
 *
 * Occupancy analysis (T4, SM75, 48KB SMEM/SM):
 *   V3: 1 warp/block × (48KB / 20KB ≈ 2 blocks/SM) = 2 warps/SM  (~6%)
 *   V4: 4 warps/block × (48KB / 48KB = 1 block/SM) = 4 warps/SM (~12.5%)
 *   Additionally, 4× more threads per block provides 4× more instruction-level
 *   parallelism, better hiding memory latency within each block.
 *
 * Warp data partitioning (all 4 warps are structurally identical):
 *   BLOCK_SIZE=64 rows split across 4 warps → 16 rows per warp.
 *   warp_id ∈ {0,1,2,3} owns rows [warp_id*16 .. warp_id*16+15].
 *
 *   Phase 1 — Load Q: all 128 threads cooperate (flat strided load).
 *   Phase 2 — Load K,V: all 128 threads cooperate.
 *   Phase 3 — wmma QKᵀ: each warp computes its 16-row strip of the 64×64
 *              score matrix (1 row-tile × 4 col-tiles of 16×16 fragments).
 *   Phase 4 — Softmax + PV: each warp processes its 16 rows independently.
 *
 * Shared memory budget (D=64):
 *   s_Q:  __half [64][64]  = 8192  bytes
 *   s_K:  __half [64][64]  = 8192  bytes
 *   s_V:  float  [64][64]  = 16384 bytes  (no padding — accept minor bank conflicts)
 *   s_S:  float  [64][64]  = 16384 bytes  (float for wmma store + softmax)
 *   Total: 49152 bytes = exactly 48 KB  ✓
 *
 * Note on s_V bank conflicts: removing the +1 padding reintroduces minor bank
 * conflicts on s_V stores, but this is acceptable given the 4× warp gain.
 */

#include <torch/extension.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda::wmma;

#define BLOCK_SIZE    64   // rows per tile
#define NUM_WARPS     4    // warps per block
#define WARP_SIZE     32   // threads per warp
#define ROWS_PER_WARP 16   // BLOCK_SIZE / NUM_WARPS

// ─────────────────────────────────────────────────────────────────────────────
// V4 kernel — templated on D_DIM (must be multiple of 16)
// ─────────────────────────────────────────────────────────────────────────────
template <int D_DIM>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE, 1)
void flash_attn_v4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ M_out,
    float* __restrict__ L_out,
    const int N, const int H
) {
    // ── Shared memory ─────────────────────────────────────────────────────────
    // No padding on __half arrays: wmma requires 32-byte alignment at each
    // 16-row boundary. D_DIM=64 → 128 bytes/row ✓, D_DIM=128 → 256 bytes/row ✓
    // s_S uses __align__(32) so wmma store_matrix_sync pointer is aligned.
    __shared__ __half  s_Q[BLOCK_SIZE][D_DIM];
    __shared__ __half  s_K[BLOCK_SIZE][D_DIM];
    __shared__ float   s_V[BLOCK_SIZE][D_DIM];
    __shared__ __align__(32) float s_S[BLOCK_SIZE][BLOCK_SIZE];

    // ── Thread/warp indices ───────────────────────────────────────────────────
    const int tx      = threadIdx.x;          // 0..127
    const int warp_id = tx / WARP_SIZE;        // 0..3
    const int lane_id = tx % WARP_SIZE;        // 0..31

    const int batch_idx  = blockIdx.z;
    const int head_idx   = blockIdx.y;
    const int q_tile_idx = blockIdx.x;

    const int slice_offset = (batch_idx * H + head_idx) * N * D_DIM;
    const float* Q_slice = Q + slice_offset;
    const float* K_slice = K + slice_offset;
    const float* V_slice = V + slice_offset;
    float*       O_slice = O + slice_offset;

    const float scale = 1.0f / sqrtf((float)D_DIM);

    // ── Per-thread accumulators ───────────────────────────────────────────────
    // Each warp owns ROWS_PER_WARP=16 rows. Lanes 0..15 each own 1 row.
    // Lanes 16..31 are "helpers" for the wmma load phase but don't write output.
    // tile_row for lane_id < ROWS_PER_WARP: warp_id*16 + lane_id
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_i[D_DIM];
    #pragma unroll
    for (int k = 0; k < D_DIM; ++k) O_i[k] = 0.0f;

    // ── Phase 1: Load Q tile (all 128 threads cooperate) ─────────────────────
    {
        const int total = BLOCK_SIZE * D_DIM;
        for (int i = tx; i < total; i += blockDim.x) {
            int row = i / D_DIM;
            int col = i % D_DIM;
            int gr  = q_tile_idx * BLOCK_SIZE + row;
            float val = (gr < N) ? __ldg(Q_slice + gr * D_DIM + col) : 0.0f;
            s_Q[row][col] = __float2half(val);
        }
    }
    __syncthreads();

    const int num_kv_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_blocks; ++kv_tile) {
        // Tile-level causal skip
        if (kv_tile > q_tile_idx) break;

        // ── Phase 2: Load K and V tiles ───────────────────────────────────────
        {
            const int total = BLOCK_SIZE * D_DIM;
            for (int i = tx; i < total; i += blockDim.x) {
                int row = i / D_DIM;
                int col = i % D_DIM;
                int gr  = kv_tile * BLOCK_SIZE + row;
                s_K[row][col] = __float2half((gr < N) ? __ldg(K_slice + gr * D_DIM + col) : 0.0f);
                s_V[row][col] = (gr < N) ? __ldg(V_slice + gr * D_DIM + col) : 0.0f;
            }
        }
        __syncthreads();

        // ── Phase 3: wmma QKᵀ → s_S[64][64] ─────────────────────────────────
        // Each warp computes 1 row-tile (16 rows) × 4 col-tiles.
        // warp_id → row_tile = warp_id (rows warp_id*16 .. warp_id*16+15)
        {
            fragment<matrix_a,    16, 16, 16, __half, row_major> frag_Q;
            fragment<matrix_b,    16, 16, 16, __half, col_major> frag_K;
            fragment<accumulator, 16, 16, 16, float>             frag_S[4];

            for (int ct = 0; ct < 4; ++ct)
                fill_fragment(frag_S[ct], 0.0f);

            for (int k = 0; k < D_DIM; k += 16) {
                load_matrix_sync(frag_Q, &s_Q[warp_id * 16][k], D_DIM);
                for (int ct = 0; ct < 4; ++ct) {
                    load_matrix_sync(frag_K, &s_K[ct * 16][k], D_DIM);
                    mma_sync(frag_S[ct], frag_Q, frag_K, frag_S[ct]);
                }
            }

            for (int ct = 0; ct < 4; ++ct)
                store_matrix_sync(&s_S[warp_id * 16][ct * 16],
                                  frag_S[ct], BLOCK_SIZE, mem_row_major);
        }
        __syncthreads();

        // ── Phase 4: Online softmax + PV accumulation ─────────────────────────
        // Only lanes 0..15 per warp process their row.
        if (lane_id < ROWS_PER_WARP) {
            const int tr  = warp_id * ROWS_PER_WARP + lane_id;
            const int gqr = q_tile_idx * BLOCK_SIZE + tr;

            float m_j = -FLT_MAX;
            float S_row[BLOCK_SIZE];

            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                float val = s_S[tr][jj] * scale;
                int gkj = kv_tile * BLOCK_SIZE + jj;
                if (gkj > gqr) val = -1e20f;
                S_row[jj] = val;
                if (val > m_j) m_j = val;
            }

            const float m_new     = fmaxf(m_i, m_j);
            const float exp_scale = expf(m_i - m_new);

            float exp_S[BLOCK_SIZE];
            float l_j = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                exp_S[jj] = expf(S_row[jj] - m_new);
                l_j += exp_S[jj];
            }

            const float l_new = exp_scale * l_i + l_j;

            #pragma unroll
            for (int k = 0; k < D_DIM; ++k)
                O_i[k] *= exp_scale;

            #pragma unroll
            for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
                #pragma unroll
                for (int k = 0; k < D_DIM; ++k)
                    O_i[k] += exp_S[jj] * s_V[jj][k];
            }

            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────────
    if (lane_id < ROWS_PER_WARP) {
        const int tr  = warp_id * ROWS_PER_WARP + lane_id;
        const int gqr = q_tile_idx * BLOCK_SIZE + tr;

        if (gqr < N) {
            const int ml_offset = (batch_idx * H + head_idx) * N;
            M_out[ml_offset + gqr] = m_i;
            L_out[ml_offset + gqr] = l_i;
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k)
                O_slice[gqr * D_DIM + k] = O_i[k] / l_i;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiation (D=64 only; D=128 exceeds 48KB SMEM with BLOCK_SIZE=64)
// ─────────────────────────────────────────────────────────────────────────────
template __global__ void flash_attn_v4_kernel<64>(
    const float*, const float*, const float*, float*, float*, float*, const int, const int);

// ─────────────────────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────────────────────
void dispatch_flash_attn_v4(
    float* Q, float* K, float* V, float* O,
    float* M_out, float* L_out,
    int B, int H, int N, int D
) {
    dim3 grid(N / BLOCK_SIZE, H, B);
    dim3 block(NUM_WARPS * WARP_SIZE);  // 128 threads

    switch (D) {
        case 64:
            flash_attn_v4_kernel<64><<<grid, block>>>(Q, K, V, O, M_out, L_out, N, H);
            break;
        case 128:
            // D=128 with BLOCK_SIZE=64: s_Q(16KB)+s_K(16KB)+s_V(32KB)+s_S(16KB)=80KB
            // exceeds the T4 48KB SMEM limit. Use V3 for D=128.
            TORCH_CHECK(false,
                "V4 does not support D=128 with BLOCK_SIZE=64 (80KB SMEM > 48KB limit). "
                "Use flash_attn_v3_forward for D=128.");
            break;
        default:
            TORCH_CHECK(false, "V4: Unsupported head dim: ", D, ". Supported: 64.");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Python binding
// ─────────────────────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attn_v4_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(),                  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4,                 "Q must be 4D: [B, H, N, d]");
    TORCH_CHECK(Q.is_contiguous(),            "Q must be contiguous");
    TORCH_CHECK(Q.size(2) % BLOCK_SIZE == 0,
                "Sequence length must be divisible by BLOCK_SIZE=64");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto O = torch::zeros_like(Q);
    auto M = torch::empty({B, H, N}, Q.options());
    auto L = torch::empty({B, H, N}, Q.options());

    dispatch_flash_attn_v4(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        M.data_ptr<float>(),
        L.data_ptr<float>(),
        B, H, N, D
    );

    return {O, M, L};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v4_forward", &flash_attn_v4_forward,
          "V4 multi-warp FlashAttention: returns (O, M, L)");
}
