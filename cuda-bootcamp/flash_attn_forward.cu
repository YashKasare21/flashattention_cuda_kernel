#include <torch/extension.h>
#include <float.h>

#define BLOCK_SIZE 32
#define D_DIM      64

// Each thread block handles one Q tile (BLOCK_SIZE rows).
// Grid dim = N / BLOCK_SIZE covers the outer "loop" over Q blocks.
// The explicit inner loop sweeps all K/V tiles, applying online softmax.
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N
) {
    __shared__ float s_Q[BLOCK_SIZE][D_DIM];
    __shared__ float s_K[BLOCK_SIZE][D_DIM];
    __shared__ float s_V[BLOCK_SIZE][D_DIM];

    const int tx    = threadIdx.x;
    const int q_row = blockIdx.x * BLOCK_SIZE + tx;
    const float scale = 1.0f / sqrtf((float)D_DIM);

    // Per-thread registers: running max, running l-sum, unnormalized output
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_i[D_DIM];
    #pragma unroll
    for (int k = 0; k < D_DIM; ++k) O_i[k] = 0.0f;

    // ── Outer loop: load this thread block's Q tile once ─────────────────────
    if (q_row < N) {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            s_Q[tx][k] = Q[q_row * D_DIM + k];
    } else {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            s_Q[tx][k] = 0.0f;
    }
    __syncthreads();

    // ── Inner loop: sweep all K/V tiles ──────────────────────────────────────
    const int num_kv_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_blocks; ++kv_tile) {
        const int kv_row = kv_tile * BLOCK_SIZE + tx;

        if (kv_row < N) {
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k) {
                s_K[tx][k] = K[kv_row * D_DIM + k];
                s_V[tx][k] = V[kv_row * D_DIM + k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k) {
                s_K[tx][k] = 0.0f;
                s_V[tx][k] = 0.0f;
            }
        }
        __syncthreads();  // tile fully loaded before compute

        // S_ij = s_Q[tx] · s_K[jj]^T * scale; track local max
        float S_ij[BLOCK_SIZE];
        float m_j = -FLT_MAX;
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k)
                dot += s_Q[tx][k] * s_K[jj][k];
            S_ij[jj] = dot * scale;
            if (S_ij[jj] > m_j) m_j = S_ij[jj];
        }

        // ── Online softmax update ─────────────────────────────────────────────
        const float m_new     = fmaxf(m_i, m_j);
        const float exp_scale = expf(m_i - m_new);   // rescales old accumulated stats

        float exp_S[BLOCK_SIZE];
        float l_j = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            exp_S[jj] = expf(S_ij[jj] - m_new);
            l_j += exp_S[jj];
        }

        const float l_new = exp_scale * l_i + l_j;

        // Rescale old unnormalized O_i, then add new V contribution
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

        __syncthreads();  // guard before next tile load
    }

    // Single final division — normalize and write to global memory
    if (q_row < N) {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            O[q_row * D_DIM + k] = O_i[k] / l_i;
    }
}

torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(),  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    const int N = Q.size(0);
    auto O = torch::zeros({N, D_DIM}, Q.options());

    flash_attn_forward_kernel<<<
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward,
          "FlashAttention forward pass — tiled shared memory + online softmax");
}
