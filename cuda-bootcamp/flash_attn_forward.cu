#include <torch/extension.h>
#include <float.h>

#define BLOCK_SIZE 32
#define D_DIM      64

// Grid: (N/BLOCK_SIZE, H, B)
//   blockIdx.x → Q tile within sequence
//   blockIdx.y → head index
//   blockIdx.z → batch index
// Each thread block owns one (batch, head, Q-tile) and sweeps all KV tiles.
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q,   // [B, H, N, d]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N, const int H
) {
    __shared__ float s_Q[BLOCK_SIZE][D_DIM];
    __shared__ float s_K[BLOCK_SIZE][D_DIM];
    __shared__ float s_V[BLOCK_SIZE][D_DIM];

    const int tx = threadIdx.x;

    // Decode 3D grid position
    const int batch_idx = blockIdx.z;
    const int head_idx  = blockIdx.y;
    const int q_tile    = blockIdx.x;
    const int q_row     = q_tile * BLOCK_SIZE + tx;   // local sequence row

    // Base pointer offset for this (batch, head) slice: [B, H, N, d] row-major
    const int slice_offset = (batch_idx * H + head_idx) * N * D_DIM;

    const float* Q_slice = Q + slice_offset;
    const float* K_slice = K + slice_offset;
    const float* V_slice = V + slice_offset;
    float*       O_slice = O + slice_offset;

    const float scale = 1.0f / sqrtf((float)D_DIM);

    // Per-thread registers
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float O_i[D_DIM];
    #pragma unroll
    for (int k = 0; k < D_DIM; ++k) O_i[k] = 0.0f;

    // ── Load Q tile for this block (once) ────────────────────────────────────
    if (q_row < N) {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            s_Q[tx][k] = Q_slice[q_row * D_DIM + k];
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
                s_K[tx][k] = K_slice[kv_row * D_DIM + k];
                s_V[tx][k] = V_slice[kv_row * D_DIM + k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < D_DIM; ++k) {
                s_K[tx][k] = 0.0f;
                s_V[tx][k] = 0.0f;
            }
        }
        __syncthreads();

        // S_ij = s_Q[tx] · s_K[jj]^T * scale
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

        // ── Online softmax update (unchanged from 2D version) ─────────────────
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

    // Normalize and write output
    if (q_row < N) {
        #pragma unroll
        for (int k = 0; k < D_DIM; ++k)
            O_slice[q_row * D_DIM + k] = O_i[k] / l_i;
    }
}

torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(),  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D: [B, H, N, d]");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto O = torch::zeros_like(Q);

    // Grid: (N/BLOCK_SIZE, H, B)
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, H, B);
    dim3 block(BLOCK_SIZE);

    flash_attn_forward_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, H
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward,
          "FlashAttention forward pass — 4D [B,H,N,d], tiled shared memory + online softmax");
}
