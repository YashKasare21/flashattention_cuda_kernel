/*
 * flash_attn_backward.cu — FlashAttention backward pass
 *
 * Computes dQ, dK, dV given (Q, K, V, O, dO, M, L) from the forward pass.
 * M[b,h,i] = final running max m_i after all KV tiles (saved by forward).
 * L[b,h,i] = final unnormalized sum of exp weights l_i (NOT divided by anything).
 *
 * Math (causal, scale = 1/sqrt(D)):
 *   S[i,j]  = dot(Q[i], K[j]) * scale          -- recomputed
 *   P[i,j]  = exp(S[i,j] - M[i]) / L[i]        -- recomputed from saved stats
 *   D_i     = sum_d( dO[i,d] * O[i,d] )         -- correction scalar
 *   dV[j]  += P[i,j] * dO[i]
 *   dP[i,j] = dot(dO[i], V[j])
 *   dS[i,j] = P[i,j] * (dP[i,j] - D_i)
 *   dQ[i]  += dS[i,j] * K[j]                    -- local accumulation, no atomics
 *   dK[j]  += dS[i,j] * Q[i]                    -- atomicAdd (multiple Q-tiles)
 *
 * Grid/block mirrors the forward: dim3 grid(N/32, H, B), dim3 block(32).
 * Each block owns one Q-tile and iterates over KV tiles 0..q_tile_idx (causal).
 *
 * Shared memory (32KB total, well within 48KB T4 limit):
 *   s_Q  [32][64] float  — loaded once before KV loop
 *   s_dO [32][64] float  — loaded once before KV loop
 *   s_K  [32][64] float  — reloaded each KV iteration
 *   s_V  [32][64] float  — reloaded each KV iteration
 */

#include <torch/extension.h>
#include <float.h>

#define BLOCK_SIZE 32
#define D_DIM      64

__global__ void flash_attn_backward_kernel(
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
    __shared__ float s_Q [BLOCK_SIZE][D_DIM];
    __shared__ float s_dO[BLOCK_SIZE][D_DIM];
    __shared__ float s_K [BLOCK_SIZE][D_DIM];
    __shared__ float s_V [BLOCK_SIZE][D_DIM];

    const int tx = threadIdx.x;

    const int batch_idx    = blockIdx.z;
    const int head_idx     = blockIdx.y;
    const int q_tile_idx   = blockIdx.x;
    const int global_q_idx = q_tile_idx * BLOCK_SIZE + tx;

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

    // ── Load s_Q and s_dO once — they stay fixed for all KV tiles ────────────
    if (global_q_idx < N) {
        const float* qsrc  = Q_slice  + global_q_idx * D_DIM;
        const float* dosrc = dO_slice + global_q_idx * D_DIM;
        #pragma unroll
        for (int d = 0; d < D_DIM; d += 4) {
            s_Q [tx][d]   = __ldg(qsrc  + d);
            s_Q [tx][d+1] = __ldg(qsrc  + d + 1);
            s_Q [tx][d+2] = __ldg(qsrc  + d + 2);
            s_Q [tx][d+3] = __ldg(qsrc  + d + 3);
            s_dO[tx][d]   = __ldg(dosrc + d);
            s_dO[tx][d+1] = __ldg(dosrc + d + 1);
            s_dO[tx][d+2] = __ldg(dosrc + d + 2);
            s_dO[tx][d+3] = __ldg(dosrc + d + 3);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d) {
            s_Q [tx][d] = 0.0f;
            s_dO[tx][d] = 0.0f;
        }
    }
    __syncthreads();

    // ── Load saved forward statistics ─────────────────────────────────────────
    const float m_i_saved = (global_q_idx < N) ? __ldg(&M[ml_offset + global_q_idx]) : 0.0f;
    const float l_i_saved = (global_q_idx < N) ? __ldg(&L[ml_offset + global_q_idx]) : 1.0f;

    // ── Compute D_i = sum_d( dO[i,d] * O[i,d] ) ──────────────────────────────
    // O is read from global memory with __ldg (no SMEM needed).
    float D_i = 0.0f;
    if (global_q_idx < N) {
        const float* osrc = O_slice + global_q_idx * D_DIM;
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d)
            D_i += s_dO[tx][d] * __ldg(osrc + d);
    }

    // ── dQ accumulator — no atomics needed (each thread owns its row) ─────────
    float dQ_acc[D_DIM];
    #pragma unroll
    for (int d = 0; d < D_DIM; ++d) dQ_acc[d] = 0.0f;

    // ── KV loop: iterate over all causal KV tiles ─────────────────────────────
    for (int kv_tile = 0; kv_tile <= q_tile_idx; ++kv_tile) {
        const int kv_row = kv_tile * BLOCK_SIZE + tx;

        // Load s_K and s_V for this tile
        if (kv_row < N) {
            const float* ksrc = K_slice + kv_row * D_DIM;
            const float* vsrc = V_slice + kv_row * D_DIM;
            #pragma unroll
            for (int d = 0; d < D_DIM; d += 4) {
                s_K[tx][d]   = __ldg(ksrc + d);
                s_K[tx][d+1] = __ldg(ksrc + d + 1);
                s_K[tx][d+2] = __ldg(ksrc + d + 2);
                s_K[tx][d+3] = __ldg(ksrc + d + 3);
                s_V[tx][d]   = __ldg(vsrc + d);
                s_V[tx][d+1] = __ldg(vsrc + d + 1);
                s_V[tx][d+2] = __ldg(vsrc + d + 2);
                s_V[tx][d+3] = __ldg(vsrc + d + 3);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < D_DIM; ++d) {
                s_K[tx][d] = 0.0f;
                s_V[tx][d] = 0.0f;
            }
        }
        __syncthreads();  // sync 1: s_K and s_V are ready

        // Iterate over each KV position jj within this tile
        #pragma unroll
        for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
            const int actual_kv_row = kv_tile * BLOCK_SIZE + jj;

            // Recompute S[tx, jj]
            float S_ij = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_DIM; ++d)
                S_ij += s_Q[tx][d] * s_K[jj][d];
            S_ij *= scale;

            // Causal mask
            if (actual_kv_row > global_q_idx) S_ij = -1e20f;

            // Recompute P[tx, jj] from saved forward statistics
            // l_i_saved is the raw unnormalized sum — used directly as denominator
            const float P_ij = expf(S_ij - m_i_saved) / l_i_saved;

            // dV[jj] += P_ij * dO[tx]
            if (actual_kv_row < N) {
                #pragma unroll
                for (int d = 0; d < D_DIM; ++d)
                    atomicAdd(&dV_slice[actual_kv_row * D_DIM + d], P_ij * s_dO[tx][d]);
            }

            // dP[tx, jj] = dot(dO[tx], V[jj])
            float dP_ij = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_DIM; ++d)
                dP_ij += s_dO[tx][d] * s_V[jj][d];

            // dS[tx, jj] = P_ij * (dP_ij - D_i)
            const float dS_ij = P_ij * (dP_ij - D_i);

            // dQ[tx] += dS_ij * K[jj]  — local, no atomics
            #pragma unroll
            for (int d = 0; d < D_DIM; ++d)
                dQ_acc[d] += dS_ij * s_K[jj][d];

            // dK[jj] += dS_ij * Q[tx]  — multiple Q-tiles write same jj row
            if (actual_kv_row < N) {
                #pragma unroll
                for (int d = 0; d < D_DIM; ++d)
                    atomicAdd(&dK_slice[actual_kv_row * D_DIM + d], dS_ij * s_Q[tx][d]);
            }
        }

        __syncthreads();  // sync 2: done with s_K/s_V before next iteration reloads them
    }

    // ── Write dQ ──────────────────────────────────────────────────────────────
    if (global_q_idx < N) {
        #pragma unroll
        for (int d = 0; d < D_DIM; ++d)
            dQ_slice[global_q_idx * D_DIM + d] = dQ_acc[d];
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attn_backward(
    torch::Tensor Q,  torch::Tensor K,  torch::Tensor V,
    torch::Tensor O,  torch::Tensor dO,
    torch::Tensor M,  torch::Tensor L
) {
    TORCH_CHECK(Q.is_cuda(),                  "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(Q.dim() == 4,                 "Q must be 4D: [B, H, N, d]");
    TORCH_CHECK(Q.is_contiguous(),            "Q must be contiguous");
    TORCH_CHECK(Q.size(2) % BLOCK_SIZE == 0,
                "Sequence length must be divisible by BLOCK_SIZE=32");
    TORCH_CHECK(dO.sizes() == Q.sizes(),      "dO must have the same shape as Q");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    // Must be zeros_like — atomicAdd accumulates into dK and dV
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    dim3 grid(N / BLOCK_SIZE, H, B);
    dim3 block(BLOCK_SIZE);

    flash_attn_backward_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),  K.data_ptr<float>(),  V.data_ptr<float>(),
        O.data_ptr<float>(),  dO.data_ptr<float>(),
        M.data_ptr<float>(),  L.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        N, H
    );

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_backward", &flash_attn_backward,
          "FlashAttention backward — inputs (Q,K,V,O,dO,M,L), returns (dQ,dK,dV)");
}
