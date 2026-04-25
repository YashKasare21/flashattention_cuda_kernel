import torch
import math

torch.manual_seed(42)

B, H, S, D = 1, 1, 2048, 64
BLOCK_SIZE = 256
scale = 1.0 / math.sqrt(D)

Q = torch.randn(B, H, S, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)

# ── Reference: standard attention ────────────────────────────────────────────
scores = (Q @ K.transpose(-2, -1)) * scale   # [B, H, S, S]
O_ref  = torch.softmax(scores, dim=-1) @ V   # [B, H, S, D]


# ── Online (block-wise) attention ─────────────────────────────────────────────
# Math recap for one query row attending to all KV pairs processed in blocks:
#
#   After processing block j, with running max m and running sum l:
#     m_new = max(m, max(s_j))
#     l_new = exp(m - m_new) * l  +  sum(exp(s_j - m_new))
#     O_new = [ exp(m - m_new) * l * O  +  exp(s_j - m_new) @ V_j ] / l_new
#
#   Rescaling O by exp(m - m_new) corrects for the shift in the running max.
#   At the last block, O = softmax(all scores) @ V exactly.

def online_attention_python(Q, K, V, block_size):
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    m = torch.full((B, H, S), float('-inf'))   # running max,  shape [B,H,S]
    l = torch.zeros(B, H, S)                   # running l-sum, shape [B,H,S]
    O = torch.zeros(B, H, S, D)               # accumulator,  shape [B,H,S,D]

    num_blocks = (S + block_size - 1) // block_size

    for j in range(num_blocks):
        start = j * block_size
        end   = min(start + block_size, S)

        K_j = K[:, :, start:end, :]            # [B, H, blk, D]
        V_j = V[:, :, start:end, :]            # [B, H, blk, D]

        s_j   = (Q @ K_j.transpose(-2, -1)) * scale   # [B, H, S, blk]
        m_j   = s_j.max(dim=-1).values                # [B, H, S]
        m_new = torch.maximum(m, m_j)                 # [B, H, S]

        exp_sj = torch.exp(s_j - m_new.unsqueeze(-1)) # [B, H, S, blk]
        l_j    = exp_sj.sum(dim=-1)                    # [B, H, S]
        l_new  = torch.exp(m - m_new) * l + l_j       # [B, H, S]

        # rescale old accumulator, add new block's contribution, normalise
        rescale = (torch.exp(m - m_new) * l).unsqueeze(-1)  # [B, H, S, 1]
        O = (rescale * O + exp_sj @ V_j) / l_new.unsqueeze(-1)

        m, l = m_new, l_new

    return O


O_out = online_attention_python(Q, K, V, BLOCK_SIZE)

match = torch.allclose(O_ref, O_out, atol=1e-4, rtol=1e-4)
max_diff = (O_ref - O_out).abs().max().item()

print(f"Shape  : Q/K/V = {list(Q.shape)},  block_size = {BLOCK_SIZE}")
print(f"Max abs diff vs reference: {max_diff:.2e}")
print(f"Results match (atol=1e-4): {match}")

if match:
    print("Online Softmax Math is Correct!")
else:
    print("MISMATCH — check implementation.")
