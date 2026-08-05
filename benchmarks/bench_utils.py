"""Shared benchmarking utilities for the FlashAttention CUDA kernels.

Colab-friendly: no hardcoded paths; everything is derived from __file__.
Provides:
    bench()          -> (mean_ms, std_ms) using CUDA events over N repeats
    forward_flops()  -> FLOP count for the forward pass (causal-aware)
    backward_flops() -> FLOP count for the backward pass (causal-aware)
    tflops()         -> FLOPs / time / 1e12
    make_record()    -> one JSON-serializable result dict
    peak_mem_mb()    -> peak GPU memory allocated by a callable
    save_results()   -> write {meta, results} to a JSON file
    device_metadata()-> GPU info for the JSON meta block
"""

import json
import os
import statistics
import time

import torch


def device_metadata():
    if not torch.cuda.is_available():
        return {"device_name": "cpu"}
    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
    }


def bench(fn, warmup=5, iters=10, repeats=5):
    """Mean +/- std of per-launch latency (ms) across `repeats` timing rounds.

    Each round times `iters` consecutive launches with torch.cuda.Event. The
    std across rounds captures run-to-run GPU noise (clock throttling, thermal
    state, contention) — a single round is not sufficient for a paper.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    round_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        round_ms.append(start.elapsed_time(end) / iters)

    mean_ms = sum(round_ms) / len(round_ms)
    std_ms = statistics.stdev(round_ms) if len(round_ms) > 1 else 0.0
    return mean_ms, std_ms


def forward_flops(B, H, N, D, causal=True):
    """QK^T (2*N*N*D) + PV (2*N*N*D) per (B, H) slice; causal ~half."""
    flops = 4.0 * B * H * N * N * D
    return flops * (0.5 if causal else 1.0)


def backward_flops(B, H, N, D, causal=True):
    """dP (dO*V^T) + dV (P^T*dO) + dQ (dS*K) + dK (dS^T*Q), each 2*N*N*D."""
    flops = 8.0 * B * H * N * N * D
    return flops * (0.5 if causal else 1.0)


def tflops(flops, mean_ms):
    return flops / (mean_ms * 1e-3) / 1e12


def make_record(version, N, B, H, D, causal, mean_ms, std_ms, flops, peak_mem_mb=None):
    record = {
        "version": version,
        "N": int(N),
        "B": int(B),
        "H": int(H),
        "D": int(D),
        "causal": bool(causal),
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "tflops": round(tflops(flops, mean_ms), 2),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if peak_mem_mb is not None:
        record["peak_mem_mb"] = round(peak_mem_mb, 2)
    return record


def peak_mem_mb(fn):
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def save_results(records, out_path, meta=None):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {"meta": meta if meta is not None else {}, "results": records}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved {len(records)} records -> {out_path}")
    return out_path
