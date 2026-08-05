"""Forward-pass benchmark: V1-V4 vs PyTorch SDPA, mean +/- std, JSON output.

Merges the functionality of the archived run_benchmarks.py (V1..V4 scaling)
and run_benchmarks_v4.py (V3 vs V4 scaling) into one active script.

Usage:
    python benchmarks/benchmark.py --output results/forward_YYYYMMDD.json
    python benchmarks/benchmark.py --output results/forward.json \
        --seq-lens 256 1024 4096 --d-values 64 128

Notes:
    * Timing uses torch.cuda.Event; each config runs `--repeats` rounds of
      `--iters` launches; mean and std across rounds are reported.
    * D=128 is benchmarked only for kernels that support it (V3, SDPA);
      V1/V2/V4 are D=64-only kernels.
    * Colab: run `!pip install -e .` first, then execute this file. Kernels
      that are not built are skipped with a warning (no hardcoded paths).
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from bench_utils import bench, device_metadata, forward_flops, make_record, save_results

DEFAULT_SEQ_LENS = [256, 512, 1024, 2048, 4096]
DEFAULT_D_VALUES = [64, 128]


def sdpa(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def make_kernels():
    """(version, callable(Q,K,V)->O, supported_dims) for every available module."""
    def wrap(module, entry):
        fn = getattr(module, entry)
        def call(Q, K, V):
            out = fn(Q, K, V)
            return out[0] if isinstance(out, tuple) else out
        return call

    kernels = [("sdpa", sdpa, {64, 128})]
    specs = [
        ("custom_flash_attn",       "flash_attn_forward",    "v1", {64}),
        ("custom_flash_attn_v2",    "flash_attn_v2_forward", "v2", {64}),
        ("custom_flash_attn_v3",    "flash_attn_v3_forward", "v3", {64, 128}),
        ("custom_flash_attn_v4",    "flash_attn_v4_forward", "v4", {64}),
    ]
    for module_name, entry, version, dims in specs:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            print(f"[warn] {module_name} not built (pip install -e .) — skipping {version}: {exc}")
            continue
        kernels.append((version, wrap(module, entry), dims))
    return kernels


def run(B, H, seq_lens, d_values, warmup, iters, repeats, seed=0, verify=True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — cannot benchmark")
    kernels = make_kernels()
    records = []

    for D in d_values:
        for N in seq_lens:
            torch.manual_seed(seed)
            Q = torch.randn(B, H, N, D, device="cuda")
            K = torch.randn(B, H, N, D, device="cuda")
            V = torch.randn(B, H, N, D, device="cuda")

            ref = None
            for version, fn, dims in kernels:
                if D not in dims:
                    continue

                max_diff = None
                if verify and version != "sdpa":
                    if ref is None:
                        ref = sdpa(Q, K, V)
                    max_diff = (fn(Q, K, V) - ref).abs().max().item()

                mean_ms, std_ms = bench(lambda: fn(Q, K, V), warmup, iters, repeats)
                flops = forward_flops(B, H, N, D, causal=True)
                rec = make_record(version, N, B, H, D, True, mean_ms, std_ms, flops)
                if max_diff is not None:
                    rec["max_diff_vs_sdpa"] = max_diff
                records.append(rec)
                diff_s = f"  diff={max_diff:.2e}" if max_diff is not None else ""
                print(f"  D={D:>3} N={N:>5} {version:<5} "
                      f"{mean_ms:>9.3f} ± {std_ms:>6.3f} ms  "
                      f"{rec['tflops']:>7.1f} TFLOPS{diff_s}")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/forward_bench.json",
                   help="output JSON path (default: results/forward_bench.json)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--seq-lens", type=int, nargs="*", default=DEFAULT_SEQ_LENS,
                   help="sequence lengths (default: 256 512 1024 2048 4096)")
    p.add_argument("--d-values", type=int, nargs="*", default=DEFAULT_D_VALUES,
                   help="head dims (default: 64 128; only V3/SDPA run at 128)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10, help="launches per timing round")
    p.add_argument("--repeats", type=int, default=5, help="rounds for mean/std")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-verify", action="store_true",
                   help="skip max_diff check vs SDPA")
    args = p.parse_args()

    records = run(B=args.B, H=args.H, seq_lens=args.seq_lens,
                  d_values=args.d_values, warmup=args.warmup,
                  iters=args.iters, repeats=args.repeats,
                  seed=args.seed, verify=not args.no_verify)

    meta = {
        "kind": "forward",
        "command": " ".join(sys.argv),
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"B": args.B, "H": args.H, "seq_lens": args.seq_lens,
                   "d_values": args.d_values, "warmup": args.warmup,
                   "iters": args.iters, "repeats": args.repeats, "seed": args.seed},
        "flops_model": "forward_flops (causal: 2*B*H*N^2*D; non-causal: 4*B*H*N^2*D)",
        "gpu": device_metadata(),
    }
    save_results(records, args.output, meta)


if __name__ == "__main__":
    main()
