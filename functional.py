"""
functional.py — PyTorch autograd wrappers for FlashAttention V4+V5 and V4+V6.

Usage:
    from functional import flash_attention        # V4 fwd + V5 bwd (stable)
    from functional import flash_attention_v6     # V4 fwd + V6 bwd (WMMA Tensor Cores)
"""

import torch
import custom_flash_attn_v4
import custom_flash_attn_v5
import custom_flash_attn_v6


# ---------------------------------------------------------------------------
# V5 backward path (original, scalar fp32) — kept intact for comparison
# ---------------------------------------------------------------------------
class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, M, L = custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, M, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = custom_flash_attn_v5.flash_attn_backward_v5(
            Q, K, V, O, dO, M, L
        )
        return dQ, dK, dV


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Causal FlashAttention with V5 (scalar fp32) backward.

    Args:
        Q, K, V: float32 tensors of shape [B, H, N, d] on CUDA.
                 N must be divisible by 64, d must be 64.
    Returns:
        O: attention output, same shape as Q.
    """
    return FlashAttentionFunc.apply(Q, K, V)


# ---------------------------------------------------------------------------
# V6 backward path (WMMA Tensor Core fp16) — optional, for benchmarking/comparison
# ---------------------------------------------------------------------------
class FlashAttentionV6Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, M, L = custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)
        ctx.save_for_backward(Q, K, V, O, M, L)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = custom_flash_attn_v6.flash_attn_backward_v6(
            Q, K, V, O, dO, M, L
        )
        return dQ, dK, dV


def flash_attention_v6(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Causal FlashAttention with V6 (WMMA Tensor Core fp16) backward.

    Uses the same V4 forward pass as flash_attention().  The backward uses
    fp16 Tensor Cores for all five matmuls (S=QKᵀ, dP, dQ, dK, dV) and
    reduces atomicAdd calls from O(N²·D) per block to O(64·D) per block.

    Precision: S and P stored fp16 → dQ tolerance ~2e-2, dK/dV ~5e-2.

    Args:
        Q, K, V: float32 tensors of shape [B, H, N, d] on CUDA.
                 N must be divisible by 64, d must be 64.
    Returns:
        O: attention output, same shape as Q.
    """
    return FlashAttentionV6Func.apply(Q, K, V)
