"""
generate_figures.py
-------------------
Generates three IEEE-conference-quality figures from results/bench_20260805.json.

Target format: IEEEtran 2-column, single-column width (3.5 in)
  • 300 DPI, vector PDF + PNG fallback
  • Serif font (DejaVu Serif / Times New Roman fallback → Computer Modern feel)
  • 8–9 pt font, colorblind-safe palette
  • Clean, minimal chrome (no top/right spines, tight layout)

Figures produced in docs/figures/:
  1. fig_forward_scaling.pdf/.png   — V1..V4 + SDPA latency vs N (log-y, error bars)
  2. fig_memory_comparison.pdf/.png — V4+V5 vs SDPA peak memory, grouped bar per N
  3. fig_backward_bottleneck.pdf/.png — Theoretical vs achieved wall-clock ratio vs N

Usage:
    python generate_figures.py [--json PATH] [--outdir PATH]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
# Use non-interactive backend — safe on headless servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# IEEE / typography configuration
# ─────────────────────────────────────────────────────────────────────────────

# Prefer Times New Roman (present on most Linux/Mac); fall back to DejaVu Serif
_SERIF_CANDIDATES = ["Times New Roman", "Times", "DejaVu Serif", "serif"]

def _best_serif() -> str:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _SERIF_CANDIDATES:
        if name in available:
            return name
    return "serif"

SERIF = _best_serif()

# Colorblind-safe palette (Wong 2011, 8-color):
#   black, orange, sky-blue, green, yellow, blue, vermilion, reddish-purple
CB_PALETTE = [
    "#000000",  # black     → V1
    "#E69F00",  # orange    → V2
    "#56B4E9",  # sky-blue  → V3
    "#009E73",  # green     → V4
    "#CC79A7",  # mauve     → SDPA / reference lines
]

# Per-version styling for the forward scaling plot
VERSION_STYLE = {
    "v1":   dict(color=CB_PALETTE[0], marker="o", linestyle="-",  label="V1 Baseline",   zorder=3),
    "v2":   dict(color=CB_PALETTE[1], marker="s", linestyle="-",  label="V2 (__ldg)",    zorder=3),
    "v3":   dict(color=CB_PALETTE[2], marker="^", linestyle="-",  label="V3 (Tensor Cores)", zorder=3),
    "v4":   dict(color=CB_PALETTE[3], marker="D", linestyle="-",  label="V4 (Multi-Warp)", zorder=4),
    "sdpa": dict(color=CB_PALETTE[4], marker="*", linestyle="--", label="PyTorch SDPA",  zorder=5),
}

# IEEE single-column dimensions
IEEE_COL_W  = 3.54   # inches  (90 mm, IEEE single-column)
IEEE_COL_H  = 2.60   # inches  (golden-ratio-ish, keeps 8-pt labels readable)
DPI_SCREEN  = 150    # for in-script preview
DPI_SAVE    = 300    # IEEE minimum

FONTSIZE_AXIS   = 8    # pt — axis tick labels
FONTSIZE_LABEL  = 9    # pt — axis titles
FONTSIZE_LEGEND = 7.5  # pt — legend entries
FONTSIZE_ANNOT  = 7    # pt — annotation text
FONTSIZE_TITLE  = 9    # pt — figure title (used sparingly in IEEE)

LINE_W   = 1.2   # default line width
MARKER_S = 5     # marker size (pts)
CAP_SIZE = 3     # error-bar cap size

def _apply_ieee_rcparams():
    """Apply global rcParams for IEEE style."""
    plt.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         [SERIF, "DejaVu Serif", "Times", "serif"],
        "font.size":          FONTSIZE_AXIS,
        "axes.labelsize":     FONTSIZE_LABEL,
        "axes.titlesize":     FONTSIZE_TITLE,
        "xtick.labelsize":    FONTSIZE_AXIS,
        "ytick.labelsize":    FONTSIZE_AXIS,
        "legend.fontsize":    FONTSIZE_LEGEND,
        "legend.title_fontsize": FONTSIZE_LEGEND,
        # Lines / markers
        "lines.linewidth":    LINE_W,
        "lines.markersize":   MARKER_S,
        "errorbar.capsize":   CAP_SIZE,
        # Axes
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.which":    "both",
        "grid.linewidth":     0.4,
        "grid.linestyle":     ":",
        "grid.color":         "#cccccc",
        # Ticks
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Figure
        "figure.dpi":         DPI_SCREEN,
        "savefig.dpi":        DPI_SAVE,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        # PDF / PS font embedding (no Type-3)
        "pdf.fonttype":       42,   # TrueType embedded → IEEE-accepted
        "ps.fonttype":        42,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_forward(records: list) -> dict:
    """
    Returns {version: {N: (mean_ms, std_ms)}} for D=64 causal forward records
    (v1, v2, v3, v4, sdpa — only those without peak_mem_mb, i.e. the forward-only pass).
    """
    data = {v: {} for v in ("v1", "v2", "v3", "v4", "sdpa")}
    for r in records:
        ver = r.get("version")
        if ver not in data:
            continue
        if r.get("D") != 64 or not r.get("causal"):
            continue
        # Exclude backward-pass sdpa records (they have peak_mem_mb)
        if "peak_mem_mb" in r:
            continue
        N = r["N"]
        data[ver][N] = (r["mean_ms"], r["std_ms"])
    return data


def extract_memory(records: list) -> dict:
    """
    Returns {version: {N: peak_mem_mb}} for D=64 causal backward-pass records
    that carry peak_mem_mb (v4+v5 and sdpa backward).
    """
    data = {"v4+v5": {}, "sdpa": {}}
    for r in records:
        ver = r.get("version")
        if ver not in data:
            continue
        if r.get("D") != 64 or not r.get("causal"):
            continue
        if "peak_mem_mb" not in r:
            continue
        N = r["N"]
        data[ver][N] = r["peak_mem_mb"]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Forward scaling — latency vs N (log-y), error bars
# ─────────────────────────────────────────────────────────────────────────────

def fig_forward_scaling(fwd_data: dict, outdir: Path):
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_H))

    seq_lens = [256, 512, 1024, 2048, 4096]

    for ver in ("v1", "v2", "v3", "v4", "sdpa"):
        d = fwd_data[ver]
        ns     = sorted(d.keys())
        means  = np.array([d[n][0] for n in ns])
        stds   = np.array([d[n][1] for n in ns])
        style  = VERSION_STYLE[ver]
        ax.errorbar(
            ns, means, yerr=stds,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=LINE_W,
            markersize=MARKER_S,
            capsize=CAP_SIZE,
            capthick=0.8,
            elinewidth=0.8,
            zorder=style["zorder"],
        )

    # --- axes formatting ---
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(200, 5500)
    ax.set_xticks(seq_lens)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda y, _: f"{y:.0f}" if y >= 1 else f"{y:.2f}".rstrip("0").rstrip(".")
    ))

    ax.set_xlabel("Sequence Length $N$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Latency (ms, log scale)", fontsize=FONTSIZE_LABEL)

    # Legend inside — 2 columns to keep it compact
    leg = ax.legend(
        loc="upper left",
        ncol=1,
        framealpha=0.85,
        edgecolor="#aaaaaa",
        borderpad=0.4,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=0.8,
        fontsize=FONTSIZE_LEGEND,
    )
    leg.get_frame().set_linewidth(0.5)

    # Annotation: V4 speedup over V1 at N=4096
    v4_4096   = fwd_data["v4"][4096][0]
    v1_4096   = fwd_data["v1"][4096][0]
    sdpa_4096 = fwd_data["sdpa"][4096][0]
    ax.annotate(
        f"V4 = {v4_4096/sdpa_4096:.1f}× SDPA",
        xy=(4096, v4_4096),
        xytext=(2200, v4_4096 * 1.6),
        fontsize=FONTSIZE_ANNOT,
        arrowprops=dict(arrowstyle="-|>", color="#555555",
                        lw=0.7, mutation_scale=6),
        color="#333333",
    )

    ax.set_title(
        "Forward Pass Latency: V1–V4 vs. PyTorch SDPA\n"
        r"Tesla T4, $B{=}2$, $H{=}4$, $d{=}64$, causal",
        fontsize=FONTSIZE_TITLE - 0.5,
        pad=4,
    )

    fig.tight_layout(pad=0.3)
    _save(fig, outdir / "fig_forward_scaling")
    plt.close(fig)
    print("  ✓  fig_forward_scaling.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Memory comparison — grouped bar V4+V5 vs SDPA (backward pass)
# ─────────────────────────────────────────────────────────────────────────────

def fig_memory_comparison(mem_data: dict, outdir: Path):
    seq_lens = [256, 512, 1024, 2048, 4096]
    x_labels = ["256", "512", "1K", "2K", "4K"]

    v4v5_mem = np.array([mem_data["v4+v5"][n] for n in seq_lens])
    sdpa_mem = np.array([mem_data["sdpa"][n]   for n in seq_lens])
    savings  = (1.0 - v4v5_mem / sdpa_mem) * 100  # % reduction

    n_groups = len(seq_lens)
    x        = np.arange(n_groups)
    bar_w    = 0.35
    gap      = 0.06

    # Color scheme: V4+V5 = green (our kernel), SDPA = mauve (baseline)
    c_v4v5 = CB_PALETTE[3]  # green
    c_sdpa = CB_PALETTE[4]  # mauve

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_H))

    bars_v4v5 = ax.bar(
        x - bar_w / 2 - gap / 2, v4v5_mem,
        width=bar_w, label="V4+V5 (ours)", color=c_v4v5,
        edgecolor="white", linewidth=0.5, zorder=3,
    )
    bars_sdpa = ax.bar(
        x + bar_w / 2 + gap / 2, sdpa_mem,
        width=bar_w, label="PyTorch SDPA", color=c_sdpa,
        edgecolor="white", linewidth=0.5, zorder=3,
        alpha=0.85,
    )

    # Annotate savings % above each V4+V5 bar
    for i, (bar, pct) in enumerate(zip(bars_v4v5, savings)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sdpa_mem[i] * 0.02,
            f"−{pct:.0f}%",
            ha="center", va="bottom",
            fontsize=FONTSIZE_ANNOT,
            color=c_v4v5,
            fontweight="bold",
        )

    # Value labels on SDPA bars (so reader has absolute reference)
    for bar, val in zip(bars_sdpa, sdpa_mem):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sdpa_mem[-1] * 0.01,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=FONTSIZE_ANNOT - 0.5,
            color="#555555",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONTSIZE_AXIS)
    ax.set_xlabel("Sequence Length $N$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Peak Memory (MiB)", fontsize=FONTSIZE_LABEL)

    # Leave headroom for annotations
    ax.set_ylim(0, sdpa_mem.max() * 1.28)

    leg = ax.legend(
        loc="upper left",
        framealpha=0.85,
        edgecolor="#aaaaaa",
        borderpad=0.4,
        handlelength=1.2,
        fontsize=FONTSIZE_LEGEND,
    )
    leg.get_frame().set_linewidth(0.5)

    # Highlight O(N) scaling with a light annotation
    ax.annotate(
        "O(N) memory\n(V4+V5)",
        xy=(x[-1] - bar_w / 2 - gap / 2, v4v5_mem[-1]),
        xytext=(x[-1] - 1.8, v4v5_mem[-1] * 0.68),
        fontsize=FONTSIZE_ANNOT,
        color=c_v4v5,
        arrowprops=dict(arrowstyle="-|>", color=c_v4v5,
                        lw=0.7, mutation_scale=6),
        ha="center",
    )

    ax.set_title(
        "Peak Memory: V4+V5 (O(N)) vs. PyTorch SDPA (O(N²))\n"
        r"Tesla T4, $B{=}2$, $H{=}4$, $d{=}64$, causal, backward pass",
        fontsize=FONTSIZE_TITLE - 0.5,
        pad=4,
    )

    fig.tight_layout(pad=0.3)
    _save(fig, outdir / "fig_memory_comparison")
    plt.close(fig)
    print("  ✓  fig_memory_comparison.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Backward bottleneck — theoretical vs achieved wall-clock ratio
# ─────────────────────────────────────────────────────────────────────────────

def fig_backward_bottleneck(outdir: Path):
    """
    Grouped bar: for N ∈ {256, 512, 1024, 2048, 4096} show
      • Theoretical FLOP ratio (constant 2.5×) — dashed reference line
      • Achieved wall-clock ratio (V5 bwd est. / V4 fwd)
    Derived directly from verified bench_20260805.json data (BACKWARD_ANALYSIS.md §0).
    """
    # From bench_20260805.json (D=64, causal, B=2, H=4, 15-repeat mean):
    #   v4 fwd mean_ms; v4+v5 total mean_ms → V5 bwd est = total − fwd
    v4_fwd  = np.array([0.130,  0.3542,  1.0865,  3.4751, 12.2612])
    v4v5    = np.array([5.0637, 15.8396, 46.4592, 164.9713, 627.306])
    v5_bwd  = v4v5 - v4_fwd

    seq_lens    = [256, 512, 1024, 2048, 4096]
    x_labels    = ["256", "512", "1K", "2K", "4K"]
    wall_ratios = v5_bwd / v4_fwd          # achieved wall-clock ratio
    theo_ratio  = 2.5                      # constant theoretical FLOP ratio

    n_groups = len(seq_lens)
    x = np.arange(n_groups)

    c_achieved = CB_PALETTE[0]   # black (achieved — heavy emphasis)
    c_theo     = CB_PALETTE[4]   # mauve (theoretical — reference)

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_H))

    # Bars: achieved wall-clock ratio
    bars = ax.bar(
        x, wall_ratios,
        width=0.5,
        color=c_achieved, alpha=0.82,
        edgecolor="white", linewidth=0.5,
        label="Achieved wall-clock ratio",
        zorder=3,
    )

    # Reference line: theoretical FLOP ratio (2.5×)
    ax.axhline(
        theo_ratio,
        color=c_theo, linestyle="--", linewidth=1.4,
        label=f"Theoretical FLOP ratio ({theo_ratio}×)",
        zorder=4,
    )

    # Shade the "implementation gap" region
    ax.fill_between(
        [-0.5, n_groups - 0.5],
        [theo_ratio, theo_ratio],
        [wall_ratios.max() * 1.12, wall_ratios.max() * 1.12],
        color=CB_PALETTE[2], alpha=0.07, zorder=1,
    )

    # Value labels on bars
    for bar, ratio in zip(bars, wall_ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + wall_ratios.max() * 0.012,
            f"{ratio:.1f}×",
            ha="center", va="bottom",
            fontsize=FONTSIZE_ANNOT,
            color=c_achieved,
            fontweight="bold",
        )

    # Gap annotation (at N=4096, largest gap)
    gap_4096 = wall_ratios[-1] - theo_ratio
    ax.annotate(
        f"Impl. gap\n({wall_ratios[-1]:.1f}× − {theo_ratio}× = {gap_4096:.1f}×)",
        xy=(x[-1], wall_ratios[-1] / 2 + theo_ratio / 2),
        xytext=(x[-1] - 1.5, wall_ratios[-1] * 0.72),
        fontsize=FONTSIZE_ANNOT,
        color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#555555",
                        lw=0.7, mutation_scale=6),
        ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=FONTSIZE_AXIS)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(0, wall_ratios.max() * 1.22)
    ax.set_xlabel("Sequence Length $N$", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(r"Slowdown Ratio ($t_\mathrm{bwd}$ / $t_\mathrm{fwd}$)", fontsize=FONTSIZE_LABEL)

    # Theoretical ratio label on the dashed line
    ax.text(
        -0.42, theo_ratio + wall_ratios.max() * 0.015,
        f"{theo_ratio}× (theory)",
        fontsize=FONTSIZE_ANNOT,
        color=c_theo,
        va="bottom",
    )

    leg = ax.legend(
        loc="upper left",
        framealpha=0.85,
        edgecolor="#aaaaaa",
        borderpad=0.4,
        handlelength=1.5,
        fontsize=FONTSIZE_LEGEND,
    )
    leg.get_frame().set_linewidth(0.5)

    ax.set_title(
        "Backward Bottleneck: Achieved vs. Theoretical Slowdown\n"
        r"V5 bwd / V4 fwd — Tesla T4, $B{=}2$, $H{=}4$, $d{=}64$, causal",
        fontsize=FONTSIZE_TITLE - 0.5,
        pad=4,
    )

    fig.tight_layout(pad=0.3)
    _save(fig, outdir / "fig_backward_bottleneck")
    plt.close(fig)
    print("  ✓  fig_backward_bottleneck.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Save helper — writes both .pdf (vector) and .png (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, stem: Path):
    for ext in (".pdf", ".png"):
        path = stem.with_suffix(ext)
        fig.savefig(path, dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate IEEE-quality figures from FlashAttention benchmark JSON."
    )
    parser.add_argument(
        "--json", default="results/bench_20260805.json",
        help="Path to benchmark JSON file (default: results/bench_20260805.json)"
    )
    parser.add_argument(
        "--outdir", default="docs/figures",
        help="Output directory for figures (default: docs/figures)"
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    outdir    = Path(args.outdir)

    if not json_path.exists():
        sys.exit(f"ERROR: JSON not found: {json_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark data from: {json_path}")
    bench  = load_json(json_path)
    records = bench["results"]

    _apply_ieee_rcparams()

    print("Generating figures …")

    # --- Fig 1: forward scaling ---
    fwd_data = extract_forward(records)
    # Sanity check
    for ver in ("v1", "v2", "v3", "v4", "sdpa"):
        n_pts = len(fwd_data[ver])
        if n_pts < 5:
            warnings.warn(f"  WARNING: only {n_pts}/5 seq_len points found for '{ver}'")
    fig_forward_scaling(fwd_data, outdir)

    # --- Fig 2: memory comparison ---
    mem_data = extract_memory(records)
    for ver in ("v4+v5", "sdpa"):
        n_pts = len(mem_data[ver])
        if n_pts < 5:
            warnings.warn(f"  WARNING: only {n_pts}/5 seq_len points found for memory '{ver}'")
    fig_memory_comparison(mem_data, outdir)

    # --- Fig 3: backward bottleneck ---
    fig_backward_bottleneck(outdir)

    print(f"\nAll figures saved to: {outdir.resolve()}")
    for f in sorted(outdir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40s}  {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
