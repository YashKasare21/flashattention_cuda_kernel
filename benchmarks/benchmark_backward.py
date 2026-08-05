"""Forward+backward benchmark: custom (V4 fwd + V5 bwd) vs PyTorch SDPA.

Reports end-to-end latency, throughput, and peak GPU memory (proxy for the
O(N) vs O(N^2) memory argument). The backward pass (V5) supports D=64 only.

Usage:
    python benchmarks/benchmark_backward.py --output results/backward_YYYYMMDD.json
    python benchmarks/benchmark_backward.py --output results/backward.json --seq-lens 256 1024 4096
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for _p in (SCRIPT_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from bench_utils import (backward_flops, bench, device_metadata, forward_flops,
                         make_record, peak_mem_mb, save_results)
from functional import flash_attention

DEFAULT_SEQ_LENS = [256, 512, 1024, 2048, 4096]


def run(B, H, seq_lens, warmup, iters, repeats, seed=0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — cannot benchmark")
    D = 64  # V5 backward is D=64 only
    records = []

    for N in seq_lens:
        torch.manual_seed(seed)
        Q = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
        K = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
        V = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
        dO = torch.randn(B, H, N, D, device="cuda")

        def custom_fwd_bwd():
            Q.grad = K.grad = V.grad = None
            flash_attention(Q, K, V).backward(dO)

        def sdpa_fwd_bwd():
            Q.grad = K.grad = V.grad = None
            F.scaled_dot_product_attention(Q, K, V, is_causal=True).backward(dO)

        flops = (forward_flops(B, H, N, D, causal=True) +
                 backward_flops(B, H, N, D, causal=True))

        for version, fn in [("v4+v5", custom_fwd_bwd), ("sdpa", sdpa_fwd_bwd)]:
            mean_ms, std_ms = bench(fn, warmup, iters, repeats)
            rec = make_record(version, N, B, H, D, True, mean_ms, std_ms, flops)
            rec["peak_mem_mb"] = peak_mem_mb(fn)
            records.append(rec)
            print(f"  N={N:>5} {version:<6} {mean_ms:>9.3f} ± {std_ms:>6.3f} ms  "
                  f"{rec['tflops']:>7.1f} TFLOPS  {rec['peak_mem_mb']:>8.1f} MB")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/backward_bench.json")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--seq-lens", type=int, nargs="*", default=DEFAULT_SEQ_LENS,
                   help="sequence lengths (default: 256 512 1024 2048 4096)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10, help="launches per timing round")
    p.add_argument("--repeats", type=int, default=5, help="rounds for mean/std")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = run(B=args.B, H=args.H, seq_lens=args.seq_lens, warmup=args.warmup,
                  iters=args.iters, repeats=args.repeats, seed=args.seed)

    meta = {
        "kind": "forward+backward",
        "command": " ".join(sys.argv),
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"B": args.B, "H": args.H, "seq_lens": args.seq_lens,
                   "warmup": args.warmup, "iters": args.iters,
                   "repeats": args.repeats, "seed": args.seed},
        "flops_model": "fwd(2*B*H*N^2*D causal) + bwd(4*B*H*N^2*D causal)",
        "gpu": device_metadata(),
    }
    save_results(records, args.output, meta)


if __name__ == "__main__":
    main()