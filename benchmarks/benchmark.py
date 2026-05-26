import torch
import torch.nn.functional as F
import custom_flash_attn
import custom_flash_attn_v2
import custom_flash_attn_v3
import custom_flash_attn_v4


def benchmark_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    assert torch.cuda.is_available(), "CUDA not available"

    B, H, N, D = 2, 4, 1024, 64
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device='cuda')
    K = torch.randn(B, H, N, D, device='cuda')
    V = torch.randn(B, H, N, D, device='cuda')

    kernels = [
        ("V1 (baseline)",  lambda: custom_flash_attn.flash_attn_forward(Q, K, V)),
        ("V2 (__ldg+pad)", lambda: custom_flash_attn_v2.flash_attn_v2_forward(Q, K, V)),
        ("V3 (wmma, 1w)",  lambda: custom_flash_attn_v3.flash_attn_v3_forward(Q, K, V)),
        ("V4 (wmma, 4w)",  lambda: custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)),
        ("PyTorch SDPA",   lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=True)),
    ]

    print(f"\nB={B} H={H} N={N} D={D}\n")
    print(f"{'Kernel':<18} {'ms':>8} {'vs V1':>8}")
    print("-" * 38)

    v1_ms = None
    for name, fn in kernels:
        ms = benchmark_fn(fn)
        if v1_ms is None:
            v1_ms = ms
        print(f"{name:<18} {ms:>8.3f} {v1_ms/ms:>8.2f}x")


if __name__ == '__main__':
    main()
