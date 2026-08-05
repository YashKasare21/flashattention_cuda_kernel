"""Single entry point: run all forward + backward benchmarks, save one JSON.

Usage (Colab-friendly, no hardcoded paths):
    python benchmarks/run_all_benchmarks.py                                   # results/bench_YYYYMMDD.json
    python benchmarks/run_all_benchmarks.py --output results/bench.json --seq-lens 1024 4096
    python benchmarks/run_all_benchmarks.py --iters 20 --repeats 5 --d-values 64

    # On Colab, first build the extensions:
    #   !pip install -e .
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benchmark as fwd_bench
import benchmark_backward as bwd_bench

from bench_utils import device_metadata, save_results

DEFAULT_SEQ_LENS = [256, 512, 1024, 2048, 4096]
DEFAULT_D_VALUES = [64, 128]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=None,
                   help="output JSON path (default: results/bench_YYYYMMDD.json)")
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--seq-lens", type=int, nargs="*", default=DEFAULT_SEQ_LENS)
    p.add_argument("--d-values", type=int, nargs="*", default=DEFAULT_D_VALUES,
                   help="forward head dims (default: 64 128); backward always D=64")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10, help="launches per timing round")
    p.add_argument("--repeats", type=int, default=5, help="rounds for mean/std")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.output or os.path.join(
        "results", "bench_" + time.strftime("%Y%m%d") + ".json")

    print("==== Forward benchmark ====")
    fwd_records = fwd_bench.run(B=args.B, H=args.H, seq_lens=args.seq_lens,
                                d_values=args.d_values, warmup=args.warmup,
                                iters=args.iters, repeats=args.repeats,
                                seed=args.seed, verify=True)

    print("\n==== Forward+Backward benchmark (D=64) ====")
    bwd_records = bwd_bench.run(B=args.B, H=args.H, seq_lens=args.seq_lens,
                                warmup=args.warmup, iters=args.iters,
                                repeats=args.repeats, seed=args.seed)

    meta = {
        "kind": "flash-attention-cuda",
        "command": " ".join(sys.argv),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"B": args.B, "H": args.H, "seq_lens": args.seq_lens,
                   "d_values": args.d_values, "warmup": args.warmup,
                   "iters": args.iters, "repeats": args.repeats, "seed": args.seed},
        "gpu": device_metadata(),
    }
    save_results(fwd_records + bwd_records, out, meta)


if __name__ == "__main__":
    main()