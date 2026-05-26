"""
functional.py — PyTorch autograd wrapper for FlashAttention V4+V5.

Usage:
    from functional import flash_attention
    O = flash_attention(Q, K, V)   # differentiable, causal
"""

import torch
import custom_flash_attn_v4
import custom_flash_attn_v5


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
    """Causal FlashAttention with autograd support.

    Args:
        Q, K, V: float32 tensors of shape [B, H, N, d] on CUDA.
                 N must be divisible by 64, d must be 64.
    Returns:
        O: attention output, same shape as Q.
    """
    return FlashAttentionFunc.apply(Q, K, V)
