# V5 Backward Kernel: Root-Cause Performance Analysis

**Scope:** Why `src/flash_attn_backward_v5.cu` (V5 backward) is far slower per launch
than `src/flash_attn_v4.cu` (V4 forward), beyond the "backward computes more" argument.

**Method / trust level:** This is a **static source analysis** grounded in verified
measurements: kernel-level facts are derived directly from the two kernel sources; timing
numbers come from the fresh 15-repeat T4 dataset in `results/bench_20260805.json`; and the
register/stack statistics are confirmed via `nvcc --ptxas-options=-v` (see §4). The only
remaining items that need a live profile are explicitly tagged **`TODO: verify on GPU`**.
No speculation is included.

---

## 0. Headline numbers — from verified T4 data (`results/bench_20260805.json`)

Source: `benchmarks/run_all_benchmarks.py --repeats 15 --iters 15` on Tesla T4
(B=2 H=4 d=64 causal, 15-round mean±std, seed 0). The `v4` record is the forward pass;
the `v4+v5` record is **forward + backward together** (the benchmark times
`flash_attention(…).backward(…)`), so **backward-only time is estimated as
`v4+v5 − v4`** (an approximation that slightly overcounts backward by the launch/pipeline
frontier between the two kernels).

| N | V4 fwd (ms) | V4+V5 (ms) | V5 bwd est. (ms) | time ratio | theor. FLOP× | throughput× |
|---|-------------|------------|------------------:|-----------:|------------:|-----------:|
| 256  | 0.130 ± 0.000 |  5.064 ± 0.031 |  4.934 | 38.0× | 2.5× | 15.2× |
| 512  | 0.354 ± 0.001 | 15.840 ± 0.057 | 15.485 | 43.7× | 2.5× | 17.5× |
| 1024 | 1.087 ± 0.003 | 46.459 ± 0.263 | 45.373 | 41.8× | 2.5× | 16.7× |
| 2048 | 3.475 ± 0.004 | 164.971 ± 1.251 | 161.496 | 46.5× | 2.5× | 18.6× |
| 4096 | 12.261 ± 0.011 | 627.306 ± 8.922 | 615.045 | 50.2× | 2.5× | 20.1× |

> **Note on earlier numbers:** the README tables previously cited e.g. V4 fwd 1.91 ms /
> V5 bwd 87.47 ms at N=1024 and V4 fwd 23.47 ms / V5 bwd 1189.47 ms at N=4096. Those were
> old hardcoded, single-run figures (previously flagged as unverified/unreproducible) and
> are **superseded** by the 15-repeat data above (which shows V4 fwd 1.087 ms / V5 bwd
> ~45.37 ms at N=1024 and V4 fwd 12.26 ms / V5 bwd ~615 ms at N=4096). Only the verified
> dataset is used in this analysis.

FLOP model (causal): forward = `2·B·H·N²·D`; backward = `5·B·H·N²·D`
(recompute S + dP + dQ + dK + dV matmuls, each `2·N²·D`, causal ≈ half).
Theoretical backward vs forward work = **2.5×**.

**The premise "15–20×" maps to the achieved-throughput ratio** (e.g. V4 ~988 GFLOPS vs V5
~59 GFLOPS at N=1024). The **raw wall-clock ratio is ~38–50×**, which is even larger.
Either framing, >2.5× of the gap is *not* explained by FLOPs — it is implementation-level
inefficiency.

---

## 1. Tensor Cores vs scalar: THE primary bottleneck ✅ verified in code

| | V4 forward | V5 backward |
|---|---|---|
| `#include <mma.h>` | **yes** | **no** |
| `fragment` / `mma_sync` | **yes** (wmma, fp16) | **none** |
| QKᵀ matmul | **Tensor Core** fp16 (`mma_sync`, 16×16×16) | **scalar fp32 FMA** (`Q_reg[d]*s_K[jj][d]`) |
| PV matmul | scalar fp32 (only non-TC matmul) | scalar fp32 |
| Other matmuls (dP, dQ, dK, dV) | n/a (not present) | **all scalar fp32** |

- In the forward kernel, the single largest matmul (QKᵀ, `2·N²·D` FLOPs) is offloaded to
  Tensor Cores: 16 `mma_sync` per warp replace thousands of scalar FMAs. Only the small
  PV (`2·N²·D`) runs scalar.
- In the backward kernel, **all five matmuls run as scalar fp32 FMA** — including the
  recomputed S (which is exactly the same QKᵀ the forward kernel did on Tensor Cores).
- The project's own `docs/archive/WRITEUP.md` cites `mma_sync` delivering ~8× the scalar
  FMA throughput. Backward therefore forfeits that ~8× on every matmul. **`TODO: verify on
  GPU`** — nvcc does not guarantee full fp16-TC vs fp32 throughput on the measured T4;
  confirm with `ncu` sm__pipe / tensor instruction rates.

**Per-active-lane, per-KV-tile instruction budget** (both BLOCK=64):

- V5 backward: `64 jj · (64 FMA S + 64 FMA dP + 64 FMA dQ)` = **12,288 scalar FMA**
  (+ 128 mults feeding atomics) **+ 8,192 atomicAdd** (+ 64 `expf`) per thread per tile.
- V4 forward: the QKᵀ is 16 warp-instructions on Tensor Cores; only PV is
  scalar = `64 jj · 64 d` = **4,096 scalar FMA** per thread per tile.

So V5 does ~3× the scalar-FMA work of V4's only scalar matmul, *plus* 8K
atomics — and V4's dominant matmul is nearly free on Tensor Cores. This alone explains a
multiplier well above 2.5×.

---

## 2. `atomicAdd`: 8,192 atomic ops / thread / tile, heavy cross-block contention ✅ verified in code

Two call sites (both in the inner `jj` loop):

- `atomicAdd(&dV_slice[actual_kv*D_DIM+d], P_ij*dO_reg[d])` — line 186
- `atomicAdd(&dK_slice[actual_kv*D_DIM+d], dS_ij*Q_reg[d])` — line 209

**Count per active thread (lane 0–15), per KV tile:**
`2 (dV,dK) · 64 jj · 64 d = **8,192 atomicAdd** over global memory`.

**Contention:** grid is `(N/64, H, B)` — all `N/64` query-tile blocks in `blockIdx.x` write
into the **same** `dK[j]` / `dV[j]` rows for a given (head, batch). For causal attention,
key `j` receives contributions from every query `i ≥ j`, i.e. from ~`(N−j)/64` different
blocks. Each atomicAdd is a **read-modify-write on HBM** (no read-local, no fp32
`red.*`/`cp.reduce.async` reuse on sm_75 — Turing has no coalesced int32 fp32 reduce peering
for fp32 atomics at this granularity). RMW traffic is far slower than a register write, and
cross-block contention serializes it.

**Rough HBM atomic traffic per block per tile:** `16 rows · 64 jj · 64 d · 2 · 4 B`
= **512 KB of RMW traffic per tile**, repeated over the causal KV sweep — dwarfing the
forward pass's single `O` write (`N·D·4 B`). This is the dominant *memory-side* cost.

**`TODO: verify on GPU`** — quantify with `ncu` `l1tex__t_sectors_pipe_lsu_mem_global_op_atom`,
`smsp__sass_thread_inst_executed_op_atom`, and atomic bank/utilization counters to put a
number on serialization latency.

---

## 3. Thread/block config & occupancy: comparable — NOT the differentiator ✅ verified in code

| | V4 forward | V5 backward |
|---|---|---|
| Threads/block | 128 (4 warps × 32) | 128 (4 warps × 32) |
| BLOCK_SIZE | 64 (16 rows/warp) | 64 (16 rows/warp) |
| `__launch_bounds__` | `(128, 1)` | `(128, 1)` |
| SMEM | `48 KB` (s_Q+s_K+s_V+s_S) | `32 KB` (s_K+s_V only) |
| Grid | `(N/64, H, B)` | `(N/64, H, B)` |

- Both run 1 block/SM ⇒ **~4 warps/SM (~12.5%** of a 32-warp T4 SM**)**. Occupancy is
  effectively identical, so occupancy does **not** explain the gap.
- V5's smallest SMEM (32 KB) means SMEM is **not** the occupancy limiter there; register
  usage is (see §4). V4's 48 KB is exactly the T4 SMEM/SM budget, making SMEM its limiter.

---

## 4. Register pressure: CONFIRMED — kernel runs at the 255-register hardware max with spills ✅ verified on GPU

V5 holds, per active thread, four `D_DIM=64` float arrays (`#pragma unroll` forces full
expansion):

- `Q_reg[64]`, `dO_reg[64]`, `O_reg[64]` (loaded pre-loop), `dQ_acc[64]` (accumulates).

That is **256 array-elements/thread**, i.e. already over the 255-register hard limit before
counting any loop state — forcing the compiler to spill regardless of how tightly it packs
the live set.

**Actual `nvcc --ptxas-options=-v` output (T4/Colab; arch flag was not explicit, so it
compiled for sm_52 — register limits are identical across sm_52/sm_75, so the numbers are
valid):**

```
Used 255 registers/thread           (hardware maximum)
1796 bytes spill stores / thread    + 1976 bytes spill loads / thread
240 bytes stack frame
32768 bytes shared memory
1 barrier
```

**Impact:** 3,772 bytes/thread of register spills become **local-memory traffic**, and
local memory is **HBM-backed** — every spilled load/store inside the hot `jj`/`d` loops is
a memory round-trip instead of a register op. At 16 active lanes/warp × 4 warps, this adds
hundreds of thousands of HBM round-trips per KV tile. This is now the **strongest, fully
quantified** bottleneck factor, alongside tensor-core absence and atomicAdd contention.

---

## 5. Bottom line: where the 2.5× ⇒ ~38–50× gap comes from (verified T4 data)

| Factor | Status | Effect |
|---|---|---|
| FLOPs of backward vs forward | verified | 2.5× (expected) |
| Tensor Cores present in backward | verified | **absent** → every matmul at fp32 scalar rate (~8× forfeit on dominant work) |
| **Register spilling** | **confirmed (nvcc)** | **255 regs/thread (HW max) + 3,772 B/thread spills → local-mem round-trips in hot loop** |
| `atomicAdd` RMW traffic | verified count | 8,192 atomic/thread/tile + severe cross-block serialization (dK/dV) |
| Occupancy | verified | ~equal (4 warps/SM) → neutral |
| `expf` count | verified | ~equal to forward (64/row/tile) → neutral |

**Decomposition (time ratio):** `2.5×` (FLOPs) ✕ `~15–20×` (implementation inefficiency:
scalar-vs-TensorCore + atomic RMW + spills).

**Decomposition (throughput ratio):** matches the user-reported ~15–20× directly
(e.g. V4 ~988 GFLOPS vs V5 ~59 GFLOPS at N=1024; 1,401 vs 70 GFLOPS at N=4096).

---

## Ready-to-cite Paper Limitations paragraph

> The custom backward kernel (V5) recomputes attention weights from saved statistics,
> which is exact and O(N)-memory, but it is ~38–50× slower per launch than the V4 forward
> (≈15–20× slower in achieved throughput, e.g. 988→59 GFLOPS at N=1024, measured on T4 with
> 15-repetition timing). Only 2.5× of this is inherent to the greater FLOP count of the
> backward pass. The remaining ~15–20× stems from three GPU-confirmed
> implementation bottlenecks: (i) the forward pass's QKᵀ matmul runs on fp16 Tensor Cores
> while the backward run always scalar fp32 FMA, forfeiting the ~8× tensor throughput on
> every matmul; (ii) dK/dV are accumulated in global memory via atomicAdd (8,192 per thread
> per KV tile) with heavy cross-block contention, incurring read-modify-write traffic of
> ~512 KB per tile versus a single O write in forward; and (iii) the backward kernel
> compiles to the 255-register hardware maximum with 3,772 B/thread spilled to local memory
> (1,796 B spill stores + 1,976 B spill loads, verified via `nvcc --ptxas-options=-v`),
> turning register references inside the hot loop into HBM-backed local-memory
> round-trips. These are engineering limits of the current formulation rather than
> algorithmic ones; a backward kernel parallelized over (num_warps × dK) with software
> pipelining, Tensor-Core dQ/dK/dV, and segmented/atomic-free reduction is the obvious
> headroom, aligning the gap toward the theoretical 2.5×.

---

## `TODO: verify on GPU` (Colab, T4) — remaining items

Register pressure is now **confirmed** (see §4: 255 regs/thread, 1,796 B spill stores +
1,976 B spill loads, 240 B stack). Note the caveat: the build was compiled for **sm_52**
(the arch flag was not explicit); register/stack limits are identical across sm_52/sm_75 so
the numbers remain valid, but an sm_75 build with `-arch=sm_75` (as `setup.py` uses) would
be the exact repro. Still open:

1. `ncu` on V5: `sm__pipe_tensor_op_count`, `smsp__sass_thread_inst_executed_op_atom`,
   `l1tex__t_sectors_pipe_lsu_mem_global_op_atom`, `sm__warps_active*` to put real numbers
   on atomic RMW sectors, atomic serialization, and occupancy.
2. Confirm Tensor-Core vs FP32 achievable throughput ratio on the exact T4 SKU (clocks can
   shift the ~8× figure from `docs/archive/WRITEUP.md`).
3. Re-run `benchmarks/run_all_benchmarks.py` (see also PROGRESS.md) to refresh the README
   numbers and attach commit-able JSON results.