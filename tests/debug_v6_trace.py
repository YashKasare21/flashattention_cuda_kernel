"""
debug_v6_trace.py — Systematic numerical tracing tool for FlashAttention V6 backward.

PURPOSE
-------
After 5 rounds of bug-fixes (accumulator flush, dK/dV overcount, fp16 quantization,
type mismatch, OOB write) the V6 backward still produces values 100-500x off from
SDPA. This script generates a ground-truth numerical reference for the SMALLEST
possible failing case (B=1, H=1, N=64, D=64 — single KV-tile, single Q-tile) so
the kernel printf output can be compared at each intermediate step:

    Step 1: S  = Q @ K.T * scale          shape [N, N]
    Step 2: P  = causal_softmax(S)         shape [N, N]
    Step 3: D  = sum(dO * O, dim=-1)       shape [N]
    Step 4: dP = dO @ V.T                  shape [N, N]
    Step 5: dS = P * (dP - D[:,None]) * scale  shape [N, N]
    Step 6: dQ = dS @ K                    shape [N, D]
    Step 7: dK = dS.T @ Q                  shape [N, D]
    Step 8: dV = P.T @ dO                  shape [N, D]

Every intermediate is saved as a .npy file under tests/ref_npy/ so you can compare
specific elements against the kernel printf output character-by-character.

The kernel printf output (when DEBUG_TRACE is defined) prints:
    [V6-TRACE] S[0][0..3]  : val0 val1 val2 val3
    [V6-TRACE] P[0][0..3]  : val0 val1 val2 val3
    [V6-TRACE] dP[0][0..3] : val0 val1 val2 val3
    [V6-TRACE] dS[0][0..3] : val0 val1 val2 val3
    [V6-TRACE] dQ[0][0..3] (partial) : val0 val1 val2 val3

DIVERGENCE DIAGNOSIS METHODOLOGY
---------------------------------
Compare kernel output vs Python reference in ORDER:

1. S[0][0..3]:
   kernel_val  vs  ref_S[0, 0:4]
   -> If mismatch: QK^T WMMA is wrong (fragment layout bug, transpose bug,
      or scale applied at wrong step).

2. P[0][0..3]:
   kernel_val  vs  ref_P[0, 0:4]
   -> If S matched but P mismatches: softmax step is wrong (m_i/l_i loaded
      from wrong offset, or causal mask applied wrong — e.g., >= vs > ).

3. dP[0][0..3]:
   kernel_val  vs  ref_dP[0, 0:4]
   -> If mismatch: dO*V^T WMMA is wrong. Since dO and V are both loaded from
      SMEM, check if s_dO is loaded correctly (row offset, col offset).

4. dS[0][0..3]:
   kernel_val  vs  ref_dS[0, 0:4]
   -> If dP matched but dS mismatches: D_i is wrong (O loaded incorrectly,
      or dO*O dot product is computed over wrong range), or scale applied
      incorrectly (missing or double-applied).

5. partial dQ[0][0..3] (BEFORE final write):
   kernel_val  vs  ref_dQ[0, 0:4]
   -> If dS matched but dQ mismatches: dS*K WMMA layout is wrong, or K is
      loaded with wrong row/column mapping (note: this is the accumulated
      value AFTER all kv_tiles, which for N=64 is just kv_tile=0).

FIRST MISMATCH = ROOT CAUSE. Do not look further.

HOW TO RUN ON COLAB
-------------------
See the printed guide at the bottom of this script's output (run it first!).

USAGE
-----
# Step 1: Run this script to generate reference .npy files
python tests/debug_v6_trace.py

# Step 2: Build and run the kernel with printf tracing enabled
pip install -e . --quiet
# Then see the Colab instructions printed by this script.
"""

import os
import sys
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REF_DIR     = os.path.join(SCRIPT_DIR, "ref_npy")
os.makedirs(REF_DIR, exist_ok=True)


def save(name: str, tensor: torch.Tensor) -> np.ndarray:
    """Convert to float32 numpy and save as .npy. Returns the numpy array."""
    arr = tensor.detach().float().cpu().numpy()
    path = os.path.join(REF_DIR, f"{name}.npy")
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# 1. Fixed-seed inputs  (B=1, H=1, N=64, D=64)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

B, H, N, D = 1, 1, 64, 64
device = "cpu"   # CPU is fine — we only need reference values, not GPU speed

# Raw tensors (float32 on CPU so arithmetic is exact)
Q  = torch.randn(B, H, N, D)   # [1,1,64,64]
K  = torch.randn(B, H, N, D)
V  = torch.randn(B, H, N, D)
dO = torch.randn(B, H, N, D)

# The V4 forward pass computes O, M, L; we replicate it here manually so we
# don't depend on the CUDA extension being built just to generate the reference.
scale = 1.0 / (D ** 0.5)

# For the forward reference we need O, M (row-max), L (softmax denominator).
# Squeeze to [N, D] for the single (b=0, h=0) slice.
q = Q[0, 0]   # [64, 64]
k = K[0, 0]
v = V[0, 0]
do = dO[0, 0]

# ── Forward pass (manual, causal) ──────────────────────────────────────────
S_fwd = (q @ k.T) * scale                              # [N, N]

# Causal mask: positions j > i are -inf
causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
S_fwd_masked = S_fwd.masked_fill(causal_mask, float('-inf'))

# Row-wise softmax with explicit m (row max) and l (sum of exp)
m_vec = S_fwd_masked.max(dim=-1).values                # [N]   saved M
exp_S = torch.exp(S_fwd_masked - m_vec[:, None])       # [N,N]
l_vec = exp_S.sum(dim=-1)                              # [N]   saved L
P_fwd = exp_S / l_vec[:, None]                         # [N,N] attention probs

O_fwd = P_fwd @ v                                      # [N,D] forward output

# Save forward intermediates (needed to reconstruct for backward)
save("Q",  q)
save("K",  k)
save("V",  v)
save("dO", do)
save("O",  O_fwd)
save("M",  m_vec)
save("L",  l_vec)
save("S_fwd",  S_fwd)          # raw S (before mask), useful for checking scale
save("P_fwd",  P_fwd)          # attention probs after causal softmax

# ── Backward pass (manual, step-by-step) ───────────────────────────────────
#
# This is the EXACT math the V6 kernel is supposed to implement.
# Each step is numbered to match the kernel comments and the kernel printf labels.
#

# Step 1: S = Q @ K.T * scale  (same as S_fwd, recomputed in backward)
S = (q @ k.T) * scale
# Apply causal mask (using -inf so that exp(-inf) = 0)
S_masked = S.masked_fill(causal_mask, float('-inf'))

# Step 2: P = exp(S - m) / l   (causal softmax recomputed from saved M, L)
#   NOTE: mask is implicit here because exp(-inf) = 0
P = torch.exp(S_masked - m_vec[:, None]) / l_vec[:, None]

# Step 3: D_i = sum_d( dO[i,d] * O[i,d] )   (softmax backward correction)
D_vec = (do * O_fwd).sum(dim=-1)   # [N]

# Step 4: dP = dO @ V.T
dP = do @ v.T                      # [N, N]

# Step 5: dS = P * (dP - D_i) * scale
dS = P * (dP - D_vec[:, None]) * scale   # [N, N]

# Step 6: dQ = dS @ K          (sum over all j: no atomics, each i-row is independent)
dQ_ref = dS @ k                    # [N, D]

# Step 7: dK = dS.T @ Q        (sum over all i: atomics in kernel)
dK_ref = dS.T @ q                  # [N, D]

# Step 8: dV = P.T @ dO        (sum over all i: atomics in kernel)
dV_ref = P.T @ do                  # [N, D]

# Save all backward intermediates
save("S",      S)         # raw S (before mask, after scale) — same as S_fwd
save("P",      P)         # [N,N] recomputed attention probs
save("D_vec",  D_vec)     # [N]   correction scalars D_i
save("dP",     dP)        # [N,N]
save("dS",     dS)        # [N,N]
save("dQ_ref", dQ_ref)    # [N,D] ground-truth dQ
save("dK_ref", dK_ref)    # [N,D] ground-truth dK
save("dV_ref", dV_ref)    # [N,D] ground-truth dV


# ---------------------------------------------------------------------------
# 2. Print reference values for row 0 (the ones the kernel will printf)
# ---------------------------------------------------------------------------
SEP = "─" * 70

print(f"\n{SEP}")
print("  debug_v6_trace.py — Reference values for V6 printf comparison")
print(f"  Config: B={B}, H={H}, N={N}, D={D}, seed=0, scale={scale:.6f}")
print(f"{SEP}\n")

def show_row(label: str, arr: np.ndarray, row: int = 0, cols: int = 8) -> None:
    """Print first `cols` values of a 2-D array's given row."""
    vals = arr[row, :cols]
    formatted = "  ".join(f"{v:+.6f}" for v in vals)
    print(f"  {label}[{row}][0..{cols-1}]:")
    print(f"    {formatted}")
    print()

def show_vec(label: str, arr: np.ndarray, cols: int = 8) -> None:
    """Print first `cols` values of a 1-D array."""
    vals = arr[:cols]
    formatted = "  ".join(f"{v:+.6f}" for v in vals)
    print(f"  {label}[0..{cols-1}]:")
    print(f"    {formatted}")
    print()

# Load back from .npy so we confirm save/load roundtrip is clean
ref = {name: np.load(os.path.join(REF_DIR, f"{name}.npy"))
       for name in ["S", "P", "D_vec", "dP", "dS", "dQ_ref", "dK_ref", "dV_ref"]}

print("── KERNEL COMPARISON TABLE (row i=0) ──\n")
print("  Copy these values. Compare against [V6-TRACE] printf output.\n")
print("  STEP 1 — S = Q@K^T * scale (after mask, BEFORE exp)")
show_row("S (masked)", ref["S"])

print("  STEP 2 — P = exp(S - m) / l  (softmax probs)")
show_row("P",  ref["P"])

print("  STEP 3 — D_i = sum(dO * O)  (correction scalar, 1-D)")
show_vec("D_vec", ref["D_vec"])

print("  STEP 4 — dP = dO @ V^T")
show_row("dP", ref["dP"])

print("  STEP 5 — dS = P * (dP - D_i) * scale")
show_row("dS", ref["dS"])

print("  STEP 6 — dQ = dS @ K  (after summing all kv_tiles — for N=64 just tile 0)")
show_row("dQ_ref", ref["dQ_ref"])

print("  STEP 7 — dK = dS^T @ Q")
show_row("dK_ref", ref["dK_ref"])

print("  STEP 8 — dV = P^T @ dO")
show_row("dV_ref", ref["dV_ref"])

# ---------------------------------------------------------------------------
# 3. Sanity check: cross-validate against PyTorch autograd
# ---------------------------------------------------------------------------
print(f"{SEP}")
print("  SANITY CHECK — Python reference vs PyTorch autograd\n")

import torch.nn.functional as F

# Create fresh leaf tensors that require grad (outside any no_grad context)
Qr  = q.detach().clone().unsqueeze(0).unsqueeze(0).requires_grad_(True)   # [1,1,N,D]
Kr  = k.detach().clone().unsqueeze(0).unsqueeze(0).requires_grad_(True)
Vr  = v.detach().clone().unsqueeze(0).unsqueeze(0).requires_grad_(True)
dOr = do.detach().clone().unsqueeze(0).unsqueeze(0)

O_sdpa = F.scaled_dot_product_attention(Qr, Kr, Vr, is_causal=True)
O_sdpa.backward(dOr)
dQ_sdpa = Qr.grad[0, 0].detach()
dK_sdpa = Kr.grad[0, 0].detach()
dV_sdpa = Vr.grad[0, 0].detach()

diffs = {
    "O":  (O_fwd - O_sdpa.detach()[0,0]).abs().max().item(),
    "dQ": (dQ_ref - dQ_sdpa).abs().max().item(),
    "dK": (dK_ref - dK_sdpa).abs().max().item(),
    "dV": (dV_ref - dV_sdpa).abs().max().item(),
}
all_ok = True
for name, diff in diffs.items():
    thr = 1e-5
    ok  = diff < thr
    all_ok = all_ok and ok
    sym = "✓" if ok else "✗ FAIL"
    print(f"  {name}: max_diff={diff:.2e}  thr={thr:.0e}  {sym}")

if all_ok:
    print("\n  All reference values match PyTorch autograd ✓")
    print("  Safe to use as ground truth for kernel comparison.\n")
else:
    print("\n  WARNING: reference does NOT match PyTorch autograd.")
    print("  The manual formulas above have a bug — fix them before tracing.\n")

# ---------------------------------------------------------------------------
# 4. Also save the 4-element slices as text for easy copy-paste
# ---------------------------------------------------------------------------
COMPARE_FILE = os.path.join(REF_DIR, "compare_values.txt")
with open(COMPARE_FILE, "w") as f:
    f.write(f"# V6 debug reference values — B={B} H={H} N={N} D={D} seed=0\n")
    f.write(f"# scale = {scale}\n\n")
    for step, (name, label) in enumerate([
        ("S",      "S[0][0..3]  after scale+mask  (STEP 1)"),
        ("P",      "P[0][0..3]  after softmax      (STEP 2)"),
        ("dP",     "dP[0][0..3] = dO@V^T           (STEP 4)"),
        ("dS",     "dS[0][0..3] = P*(dP-D)*scale   (STEP 5)"),
        ("dQ_ref", "dQ[0][0..3] = dS@K final       (STEP 6)"),
    ], 1):
        arr  = ref[name] if name != "S" else np.load(os.path.join(REF_DIR, "S.npy"))
        vals = arr[0, :4]
        f.write(f"[V6-REF] {label}\n")
        f.write(f"  {vals[0]:.6f}  {vals[1]:.6f}  {vals[2]:.6f}  {vals[3]:.6f}\n\n")

print(f"  Comparison values also saved to: {COMPARE_FILE}\n")

# ---------------------------------------------------------------------------
# 5. Colab-runnable instructions
# ---------------------------------------------------------------------------
print(SEP)
print("  COLAB INSTRUCTIONS — printf tracing workflow")
print(SEP)
print("""
OVERVIEW
  You have two files:
    tests/debug_v6_trace.py          ← this script (Python reference)
    src/flash_attn_backward_v6.cu    ← V6 kernel (with DEBUG_TRACE printfs)

STEP 1 — Upload / clone the project in Colab
  !git clone https://github.com/YashKasare21/flashattention_cuda_kernel.git
  %cd flashattention_cuda_kernel

  OR upload via Files panel if working from a local copy.

STEP 2 — Generate Python reference values
  !python tests/debug_v6_trace.py

  This creates tests/ref_npy/ with .npy files and prints the reference table.
  KEEP THIS OUTPUT VISIBLE — you will compare against kernel printfs.

STEP 3 — Build with DEBUG_TRACE enabled
  The kernel printf guard is:
      #define DEBUG_TRACE   ← ADD THIS LINE at top of the .cu file
  (Already present but commented out — see "DEBUG_TRACE" in the kernel.)

  Build:
      !pip install -e . --quiet

  If build fails (type error, etc.) you may need to ensure CUDA 12+ is loaded:
      !nvcc --version
      !python -c "import torch; print(torch.cuda.is_available())"

STEP 4 — Run the trace kernel
  !python - << 'EOF'
  import sys, os
  sys.path.insert(0, '.')
  import torch
  import numpy as np
  import custom_flash_attn_v4
  import custom_flash_attn_v6

  torch.manual_seed(0)
  B, H, N, D = 1, 1, 64, 64
  device = 'cuda'

  Q  = torch.randn(B, H, N, D, device=device)
  K  = torch.randn(B, H, N, D, device=device)
  V  = torch.randn(B, H, N, D, device=device)
  dO = torch.randn(B, H, N, D, device=device)

  O, M, L = custom_flash_attn_v4.flash_attn_v4_forward(Q, K, V)
  dQ, dK, dV = custom_flash_attn_v6.flash_attn_backward_v6(Q, K, V, O, dO, M, L)
  torch.cuda.synchronize()  # flush printf buffer
  print("Kernel run complete.")
  EOF

  The [V6-TRACE] lines will appear in the Colab output (stdout).

STEP 5 — Compare
  Find the FIRST step where kernel and Python reference diverge:

  Python reference (from Step 2 output):
    [V6-REF] S[0][0..3]  : a b c d
    [V6-REF] P[0][0..3]  : a b c d
    [V6-REF] dP[0][0..3] : a b c d
    [V6-REF] dS[0][0..3] : a b c d
    [V6-REF] dQ[0][0..3] (partial) : a b c d

  Kernel output (from Step 4):
    [V6-TRACE] S[0][0..3]  : a' b' c' d'   ← compare with Python's S
    [V6-TRACE] P[0][0..3]  : a' b' c' d'   ← compare with Python's P
    ... etc.

  DIAGNOSIS:
    If S mismatches  → QK^T WMMA layout or scale is wrong
    If P mismatches  → softmax: wrong m_i/l_i indexing, or causal mask off-by-one
    If dP mismatches → dO@V^T WMMA layout or s_dO loading is wrong
    If dS mismatches → D_i computation is wrong (O loaded incorrectly)
    If dQ mismatches → dS@K WMMA layout or K loading is wrong

  The FIRST mismatch is the root cause. Fix only that. Rerun.

STEP 6 — Remove printf before committing
  Once root cause is identified and fixed, remove or comment out:
      #define DEBUG_TRACE
  Then rebuild and run tests/test_backward_v6.py.

TIPS
  - torch.cuda.synchronize() is ESSENTIAL — without it printf may not flush.
  - printf in CUDA goes to stdout. In Colab it appears inline after the kernel.
  - If you see NO [V6-TRACE] output: check that DEBUG_TRACE is defined and
    the build was not cached (delete build/ directory and reinstall).
  - For N=64, there is exactly 1 Q-tile (blockIdx.x=0) and 1 KV-tile (kv_tile=0).
    All 128 threads run, but printf is gated to warp_id==0 && lane_id < 4 — so
    you get exactly 4 lines per step, one per lane.

""")

print(SEP)
print(f"  .npy files saved to: {REF_DIR}/")
print(f"  Files: Q, K, V, dO, O, M, L, S_fwd, P_fwd,")
print(f"         S, P, D_vec, dP, dS, dQ_ref, dK_ref, dV_ref,")
print(f"         compare_values.txt")
print(SEP)
print()
