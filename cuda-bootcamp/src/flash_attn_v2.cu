#include <torch/extension.h>
#include <float.h>

#define BLOCK_SIZE 32
#define D_DIM      64

// PERF CHANGE 2: +1 padding on the innermost dim eliminates 32-way shared memory
// bank conflicts. With D_DIM=64 floats per row, consecutive threads hitting
// s_Q[tx][k] for the same k map to the same bank (64 % 32 == 0). Adding 1
// shifts each row by one float, spreading accesses across all 32 banks.
__global__ void flash_attn_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N, const int H
) {
    __shared__ float s_Q[BLOCK_SIZE][D_DIM + 1];  // +1 = bank-conflict padding
    __shared__ float s_K[BLOCK_SIZE][D_DIM + 1];
    __shared__ float s_V[BLOCK_SIZE][D_DIM + 1];

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

    // ── Load Q tile ───────────────────────────────────────────────────────────
    // PERF CHANGE 1: float4 vectorized loads. D_DIM=64, 64/4=16 float4 per row.
    // Each thread issues 16 wide 128-bit loads instead of 64 scalar 32-bit loads,
    // cutting load-instruction count 4x and maximising memory-bus utilisation.
    if (global_q_idx < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Q_slice + global_q_idx * D_DIM);
        float4* sQ4      = reinterpret_cast<float4*>(s_Q[tx]);
        #pragma unroll
        for (int k = 0; k < D_DIM / 4; ++k)
            sQ4[k] = Q4[k];
    } else {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k) s_Q[tx][k] = 0.0f;
    }
    __syncthreads();

    const int num_kv_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int kv_tile = 0; kv_tile < num_kv_blocks; ++kv_tile) {
        if (kv_tile > q_tile_idx) break;

        const int kv_row = kv_tile * BLOCK_SIZE + tx;

        // PERF CHANGE 1 (cont.): same float4 load pattern for K and V tiles.
        if (kv_row < N) {
            const float4* K4 = reinterpret_cast<const float4*>(K_slice + kv_row * D_DIM);
            const float4* V4 = reinterpret_cast<const float4*>(V_slice + kv_row * D_DIM);
            float4* sK4      = reinterpret_cast<float4*>(s_K[tx]);
            float4* sV4      = reinterpret_cast<float4*>(s_V[tx]);
            #pragma unroll
            for (int k = 0; k < D_DIM / 4; ++k) {
                sK4[k] = K4[k];
                sV4[k] = V4[k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k) {
                s_K[tx][k] = 0.0f;
                s_V[tx][k] = 0.0f;
            }
        }
        __syncthreads();

        float S_ij[BLOCK_SIZE];
        float m_j = -FLT_MAX;
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k)
                dot += s_Q[tx][k] * s_K[jj][k];
            S_ij[jj] = dot * scale;

            int global_k_idx = kv_tile * BLOCK_SIZE + jj;
            if (global_k_idx > global_q_idx) S_ij[jj] = -1e20f;

            if (S_ij[jj] > m_j) m_j = S_ij[jj];
        }

        const float m_new     = fmaxf(m_i, m_j);
        const float exp_scale = expf(m_i - m_new);

        float exp_S[BLOCK_SIZE];
        float l_j = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            exp_S[jj] = expf(S_ij[jj] - m_new);
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

torch::Tensor flash_attn_v2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(),  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D: [B, H, N, d]");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto O = torch::zeros_like(Q);

    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, H, B);
    dim3 block(BLOCK_SIZE);

    flash_attn_v2_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, H
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_v2_forward", &flash_attn_v2_forward,
          "FlashAttention V2 — float4 vectorized loads + shared memory bank padding");
}
