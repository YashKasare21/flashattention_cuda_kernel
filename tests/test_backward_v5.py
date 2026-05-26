"""
test_backward_v5.py — Correctness tests for FlashAttention V5 backward pass.

Tests:
  1. Direct kernel: flash_attn_backward_v5 vs PyTorch SDPA backward
  2. Autograd: FlashAttentionFunc.backward via loss.backward()
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import custom_flash_attn_v4
import custom_flash_attn_v5
from functional import flash_attention


def test_backward_kernel(B=2, H=4, N=512, D=64, seed=42):
    """Compare dQ/dK/dV from V5 kernel against PyTorch SDPA backward."""
    torch.manual_seed(seed)
    device = 'cuda'

    Q  = torch.randn(B, H, N, D, device=device)
    K  = torch.randn(B, H, N, D, device=device)
    V  = torch.randn(B, H, N, D, device=device)
    dO = torch.randn(B, H, N, D, device=device)

    # V4 forward → (O, M, L)
    O, M, L = custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)

    # V5 backward
    dQ_c, dK_c, dV_c = custom_flash_attn_v5.flash_attn_backward_v5(
        Q, K, V, O, dO, M, L
    )

    # PyTorch reference
    Qr = Q.detach().requires_grad_(True)
    Kr = K.detach().requires_grad_(True)
    Vr = V.detach().requires_grad_(True)
    F.scaled_dot_product_attention(Qr, Kr, Vr, is_causal=True).backward(dO)

    diffs = {
        'dQ': (dQ_c - Qr.grad).abs().max().item(),
        'dK': (dK_c - Kr.grad).abs().max().item(),
        'dV': (dV_c - Vr.grad).abs().max().item(),
    }
    # dQ: local accumulation, tight tolerance
    # dK/dV: atomicAdd, slightly looser
    thresholds = {'dQ': 1e-2, 'dK': 5e-2, 'dV': 5e-2}

    print(f"\n[test_backward_kernel] B={B} H={H} N={N} D={D}")
    all_pass = True
    for name, diff in diffs.items():
        ok = diff < thresholds[name]
        all_pass = all_pass and ok
        print(f"  {name}: max_diff={diff:.2e}  thr={thresholds[name]:.0e}  {'✓' if ok else '✗ FAIL'}")
    return all_pass


def test_autograd(B=1, H=2, N=128, D=64, seed=7):
    """Verify FlashAttentionFunc integrates correctly with autograd."""
    torch.manual_seed(seed)
    device = 'cuda'

    Q = torch.randn(B, H, N, D, device=device, requires_grad=True)
    K = torch.randn(B, H, N, D, device=device, requires_grad=True)
    V = torch.randn(B, H, N, D, device=device, requires_grad=True)

    # Custom autograd path
    O_custom = flash_attention(Q, K, V)
    loss_c = O_custom.sum()
    loss_c.backward()
    dQ_c, dK_c, dV_c = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    # PyTorch reference
    Q.grad = K.grad = V.grad = None
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    O_ref.sum().backward()
    dQ_r, dK_r, dV_r = Q.grad.clone(), K.grad.clone(), V.grad.clone()

    diffs = {
        'dQ': (dQ_c - dQ_r).abs().max().item(),
        'dK': (dK_c - dK_r).abs().max().item(),
        'dV': (dV_c - dV_r).abs().max().item(),
    }
    thresholds = {'dQ': 1e-2, 'dK': 5e-2, 'dV': 5e-2}

    print(f"\n[test_autograd] B={B} H={H} N={N} D={D}")
    all_pass = True
    for name, diff in diffs.items():
        ok = diff < thresholds[name]
        all_pass = all_pass and ok
        print(f"  {name}: max_diff={diff:.2e}  thr={thresholds[name]:.0e}  {'✓' if ok else '✗ FAIL'}")
    return all_pass


def test_multiple_shapes():
    """Run backward correctness across several shapes."""
    configs = [
        (1, 1, 64,  64),
        (2, 4, 256, 64),
        (2, 4, 512, 64),
        (1, 8, 1024, 64),
    ]
    all_pass = True
    for B, H, N, D in configs:
        ok = test_backward_kernel(B, H, N, D)
        all_pass = all_pass and ok
    return all_pass


if __name__ == '__main__':
    results = {
        'kernel':  test_backward_kernel(),
        'autograd': test_autograd(),
        'shapes':  test_multiple_shapes(),
    }
    print("\n" + "=" * 40)
    overall = all(results.values())
    for name, ok in results.items():
        print(f"  {name:<12}: {'PASSED' if ok else 'FAILED'}")
    print("=" * 40)
    print(f"  Overall: {'PASSED ✓' if overall else 'FAILED ✗'}")
    sys.exit(0 if overall else 1)
