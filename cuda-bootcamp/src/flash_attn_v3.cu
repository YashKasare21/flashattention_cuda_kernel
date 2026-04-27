/*
 * flash_attn_v3.cu — FlashAttention V3: Tensor Core wmma QKᵀ
 *
 * V3 result: 1.231ms vs V1 2.976ms (2.42x speedup)
 * Correctness: max abs diff 0.001142 vs PyTorch SDPA (atol=1e-2 ✓)
 * Profiled on T4 GPU (sm_75), B=2 H=4 N=1024 d=64
 *
 * What wmma is and why it matters:
 *   NVIDIA Tensor Cores execute a 16×16×16 matrix multiply-accumulate (MMA)
 *   in a single warp-level instruction, delivering ~8× the throughput of
 *   equivalent scalar FMA code on the same SM. The wmma (Warp Matrix Multiply
 *   Accumulate) C++ API exposes these units via fragment types and
 *   load/store/mma_sync intrinsics without writing PTX by hand.
 *
 * Fragment types and tile dimensions:
 *   fragment<matrix_a, 16,16,16, __half, row_major>  — A tile (16×16 fp16)
 *   fragment<matrix_b, 16,16,16, __half, col_major>  — B tile (16×16 fp16)
 *   fragment<accumulator, 16,16,16, float>            — C/D tile (16×16 fp32)
 *   mma_sync(D, A, B, C) computes D = A × B + C in Tensor Core hardware.
 *
 * Why fp16 input is required:
 *   Tensor Core MMA instructions on sm_75 (Turing/T4) require fp16 operands
 *   for the A and B matrices. The accumulator can be fp32, which we use to
 *   preserve numerical precision in the softmax and output accumulation.
 *
 * 1-warp-per-block constraint at BLOCK_SIZE=32:
 *   BLOCK_SIZE=32 → 32 threads per block = exactly 1 warp. wmma operations
 *   are warp-collective: all 32 threads in a warp must call load_matrix_sync /
 *   mma_sync / store_matrix_sync together. With 1 warp per block this is
 *   automatically satisfied. The single warp iterates over four 16×16×16
 *   wmma tiles (2 row-tiles × 2 col-tiles) to cover the full 32×32 score
 *   matrix, accumulating over D_DIM=64 in steps of 16.
 */

#include <torch/extension.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda::wmma;

#define BLOCK_SIZE 32
#define D_DIM      64

__global__ void flash_attn_v3_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N, const int H
) {
    // wmma::load_matrix_sync requires shared memory pointers to be 32-byte
    // (256-bit) aligned at each 16-row sub-tile boundary.
    // __half s_Q/s_K: D_DIM=64 halfs × 2 bytes = 128 bytes per row — already
    //   a multiple of 32, so NO padding is needed (adding +1 would break it).
    // float s_V: keep +1 bank-conflict padding (floats don't need wmma alignment).
    // float s_S: __align__(32) ensures the base pointer satisfies wmma store.
    __shared__ __half s_Q[BLOCK_SIZE][D_DIM];
    __shared__ __half s_K[BLOCK_SIZE][D_DIM];
    __shared__ float  s_V[BLOCK_SIZE][D_DIM + 1];
    __shared__ __align__(32) float s_S[BLOCK_SIZE][BLOCK_SIZE];

    #ifdef DEBUG_ALIGN
    if (threadIdx.x == 0) {
        assert(((uintptr_t)s_Q % 32) == 0);
        assert(((uintptr_t)s_K % 32) == 0);
        assert(((uintptr_t)s_S % 32) == 0);
    }
    #endif

    const int tx = threadIdx.x;

    const int batch_idx    = blockIdx.z;
    const int head_idx     = blockIdx.y;
    const int q_tile_idx   = blockIdx.x;
    const int global_q_idx = q_tile_idx * BLOCK_SIZE + tx;

    const int slice_offset = (batch_idx * H + head_idx) * N * D_DIM;

    const float* Q_slice = Q + slice_offset;
    const float* K_slice = K + slice_offset;
    const float* V_slice = V + slice_offset;
    float*       O_slice = O + slice_offset;

    const float scale = 1.0f / sqrtf((float)D_DIM);

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_i[D_DIM];
    #pragma unroll
    for (int k = 0; k < D_DIM; ++k) O_i[k] = 0.0f;

    // ── Load Q tile (fp32 → fp16) ─────────────────────────────────────────────
    if (global_q_idx < N) {
        const float* src = Q_slice + global_q_idx * D_DIM;
        #pragma unroll
        for (int k = 0; k < D_DIM; k += 4) {
            s_Q[tx][k]   = __float2half(__ldg(src + k));
            s_Q[tx][k+1] = __float2half(__ldg(src + k + 1));
            s_Q[tx][k+2] = __float2half(__ldg(src + k + 2));
            s_Q[tx][k+3] = __float2half(__ldg(src + k + 3));
        }
    } else {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k) s_Q[tx][k] = __float2half(0.0f);
    }
    __syncthreads();

    const int num_kv_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_blocks; ++kv_tile) {
        if (kv_tile > q_tile_idx) break;

        const int kv_row = kv_tile * BLOCK_SIZE + tx;

        // ── Load K tile (fp32 → fp16) and V tile (fp32) ──────────────────────
        if (kv_row < N) {
            const float* ksrc = K_slice + kv_row * D_DIM;
            const float* vsrc = V_slice + kv_row * D_DIM;
            #pragma unroll
            for (int k = 0; k < D_DIM; k += 4) {
                s_K[tx][k]   = __float2half(__ldg(ksrc + k));
                s_K[tx][k+1] = __float2half(__ldg(ksrc + k + 1));
                s_K[tx][k+2] = __float2half(__ldg(ksrc + k + 2));
                s_K[tx][k+3] = __float2half(__ldg(ksrc + k + 3));
                s_V[tx][k]   = __ldg(vsrc + k);
                s_V[tx][k+1] = __ldg(vsrc + k + 1);
                s_V[tx][k+2] = __ldg(vsrc + k + 2);
                s_V[tx][k+3] = __ldg(vsrc + k + 3);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k) {
                s_K[tx][k] = __float2half(0.0f);
                s_V[tx][k] = 0.0f;
            }
        }
        __syncthreads();

        // ── wmma QKᵀ: compute s_S[32][32] = s_Q[32×64] @ s_K[32×64]ᵀ ────────
        // Decompose into 2×2 grid of 16×16 output tiles, each accumulated
        // over D_DIM=64 in 4 steps of 16 (4 × 16×16×16 mma_sync calls).
        //
        // frag_S[i][j] accumulates the (i*16 .. i*16+15, j*16 .. j*16+15)
        // quadrant of the 32×32 score matrix.
        fragment<matrix_a,    16, 16, 16, __half, row_major> frag_Q;
        fragment<matrix_b,    16, 16, 16, __half, col_major> frag_K;
        fragment<accumulator, 16, 16, 16, float>             frag_S[2][2];

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                fill_fragment(frag_S[i][j], 0.0f);

        for (int k = 0; k < D_DIM; k += 16) {
            for (int i = 0; i < 2; i++) {
                // row stride = D_DIM (no padding on __half arrays)
                load_matrix_sync(frag_Q, &s_Q[i * 16][k], D_DIM);
                for (int j = 0; j < 2; j++) {
                    // col_major load transposes s_K in-place
                    load_matrix_sync(frag_K, &s_K[j * 16][k], D_DIM);
                    mma_sync(frag_S[i][j], frag_Q, frag_K, frag_S[i][j]);
                }
            }
        }

        // Store all four 16×16 quadrants into s_S
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                store_matrix_sync(&s_S[i * 16][j * 16], frag_S[i][j],
                                  BLOCK_SIZE, mem_row_major);

        __syncthreads();

        // ── Online softmax using s_S rows ─────────────────────────────────────
        float m_j = -FLT_MAX;
        float S_row[BLOCK_SIZE];
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            float val = s_S[tx][jj] * scale;
            int global_k_idx = kv_tile * BLOCK_SIZE + jj;
            if (global_k_idx > global_q_idx) val = -1e20f;
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

        __syncthreads();
    }

    if (global_q_idx < N) {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            O_slice[global_q_idx * D_DIM + k] = O_i[k] / l_i;
    }
}

torch::Tensor flash_attn_v3_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(),        "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4,       "Q must be 4D: [B, H, N, d]");
    TORCH_CHECK(Q.is_contiguous(),  "Q must be contiguous");
    TORCH_CHECK(Q.size(2) % BLOCK_SIZE == 0,
                "Sequence length must be divisible by BLOCK_SIZE=32 for wmma tiling");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto O = torch::zeros_like(Q);

    dim3 grid(N / BLOCK_SIZE, H, B);
    dim3 block(BLOCK_SIZE);

    flash_attn_v3_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, H
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v3_forward", &flash_attn_v3_forward,
          "FlashAttention V3 — wmma Tensor Core QKᵀ + fp16 tiles + fp32 softmax/O");
}
