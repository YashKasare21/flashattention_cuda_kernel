import torch
import torch.nn.functional as F
import custom_flash_attn_v3
import custom_flash_attn_backward


def test_backward(B=2, H=4, N=512, D=64, seed=42):
    torch.manual_seed(seed)
    Q  = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    K  = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V  = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    # Custom forward (returns O, M, L)
    O, M, L = custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V)

    # Custom backward
    dQ_c, dK_c, dV_c = custom_flash_attn_backward.flash_attn_backward(
        Q, K, V, O, dO, M, L)

    # PyTorch reference
    Qr = Q.detach().requires_grad_(True)
    Kr = K.detach().requires_grad_(True)
    Vr = V.detach().requires_grad_(True)
    F.scaled_dot_product_attention(Qr, Kr, Vr, is_causal=True).backward(dO)

    results = {
        'dQ': (dQ_c - Qr.grad).abs().max().item(),
        'dK': (dK_c - Kr.grad).abs().max().item(),
        'dV': (dV_c - Vr.grad).abs().max().item(),
    }
    # dQ has no atomics — tighter tolerance
    # dK, dV use atomicAdd — non-deterministic ordering, looser tolerance
    thresholds = {'dQ': 1e-2, 'dK': 5e-2, 'dV': 5e-2}

    print(f"\nBackward correctness [{B},{H},{N},{D}]:")
    all_pass = True
    for name, diff in results.items():
        ok = diff < thresholds[name]
        all_pass = all_pass and ok
        print(f"  {name}: max_diff={diff:.2e}  threshold={thresholds[name]:.0e}  {'✓' if ok else '✗ FAIL'}")

    return all_pass


if __name__ == '__main__':
    ok = test_backward()
    print(f"\nResult: {'PASSED' if ok else 'FAILED'}")
