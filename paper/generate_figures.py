#!/usr/bin/env python3
"""Generate all four vector figures (SVG + PDF) for the BitFSAI paper.

All plotted CPU and GPU arrays (Figures 1, 2, 3, and 4) are computed directly
from the raw CSVs in ``../benchmarks/results/*.csv`` (paired medians are
medians of ``reset_ms + solve_ms`` within each timed trial).
"""

import csv
import statistics as st
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
})

C_BLUE   = "#1f5a99"
C_TEAL   = "#0e8a7d"
C_ORANGE = "#d96b27"
C_RED    = "#c0392b"
C_SLATE  = "#5c6b73"
C_PURPLE = "#6c4f99"
C_NAVY   = "#102a43"

CASES_ORDER = [
    "grid128",
    "grid192",
    "tet48",
    "elastic28",
    "rail_79841_neg.mtx",
    "bcsstk14.mtx",
    "bcsstk15.mtx",
    "bcsstk16.mtx",
]


def load_timed_rows(filename: str) -> list[dict[str, str]]:
    with open(RESULTS / filename, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("phase") == "timed"]


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig1() -> None:
    """Figure 1: CPU smoother quantization cliff & coarse Galerkin dynamic-range floor."""
    with open(RESULTS / "cpu_precision_sweep.csv", newline="", encoding="utf-8") as fh:
        cpu_rows = list(csv.DictReader(fh))
    exp_a = {r["fmt"]: r for r in cpu_rows if r["experiment"] == "A"}
    exp_b = {r["label"]: r for r in cpu_rows if r["experiment"] == "B"}

    fig = plt.figure(figsize=(10.6, 3.80))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.25],
        height_ratios=[0.27, 0.73],
        hspace=0.11,
        wspace=0.31,
    )
    ax1 = fig.add_subplot(gs[:, 0])
    ax2_top = fig.add_subplot(gs[0, 1])
    ax2_bot = fig.add_subplot(gs[1, 1], sharex=ax2_top)

    # Panel (a): Smoother quantization cliff from cpu_precision_sweep.csv
    fmts = ["FP64", "FP16", "INT4", "INT3", "INT2"]
    x = np.arange(len(fmts))
    ic_lmax = [float(exp_a[f]["ic_lmax"]) for f in fmts]
    fs_lmax = [float(exp_a[f]["fs_lmax"]) for f in fmts]
    ic_int3_it = int(exp_a["INT3"]["ic_estw_iters"])
    ic_int2_it = int(exp_a["INT2"]["ic_estw_iters"])
    fs_int3_it = int(exp_a["INT3"]["fs_w072_iters"])
    ratio_cliff = ic_lmax[3] / ic_lmax[0]

    ax1.plot(x, ic_lmax, marker="o", lw=2.2, ms=6.3, color=C_RED,
             label="IC(0) diag-inclusive ($L_q^{-T}L_q^{-1}$)")
    ax1.plot(x, fs_lmax, marker="s", lw=2.2, ms=6.3, color=C_TEAL,
             label="BitFSAI split Gram ($G_q^T G_q$)")
    ax1.axhline(2.0, color=C_SLATE, ls=":", lw=1.2,
                label="Undamped $\\omega=1$ limit ($\\lambda_{\\max}=2$)")
    ax1.annotate(
        f"{ratio_cliff:.1f}$\\times$ inflation\n($\\omega=1$ breakdown;\nest. $\\omega$: {ic_int3_it}, {ic_int2_it} it)",
        xy=(3, ic_lmax[3]), xytext=(0.12, 6.20),
        fontsize=7.6, color=C_RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88),
        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.1),
    )
    ax1.annotate(
        f"Near {fs_lmax[3]:.2f} ({fs_int3_it} it at $\\omega=0.72$)",
        xy=(3, fs_lmax[3]), xytext=(1.75, 3.65),
        fontsize=7.6, color=C_TEAL, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.88),
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.1),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(fmts)
    ax1.set_ylabel(r"Estimated spectral radius $\hat{\lambda}_{\max}(M_q A)$")
    ax1.set_ylim(0, 14.3)
    ax1.set_title("(a) Smoother Quantization Cliff (3D $20^3$, Two-Grid)", fontsize=9.4)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Coarse-grid PCG from cpu_precision_sweep.csv (broken linear y-axis)
    bar_spec = [
        ("FP64\n(exact)", "FP64 exact", None, C_SLATE, ""),
        ("FP16\n(naive)", "FP16 (naive)", None, C_RED, "///"),
        ("FP16 +\nrow-sum", "FP16 + row-sum (FP64)", "FP16 + row-sum (FP32)", C_BLUE, ""),
        ("INT8 +\nrow-sum", "INT8 + row-sum (FP64)", "INT8 + row-sum (FP32)", C_ORANGE, "..."),
        ("FP8 E5M2\n(naive)", "FP8 E5M2 (naive)", None, C_RED, "///"),
        ("FP8 E5M2\n+ row-sum", "FP8 E5M2 + row-sum (FP64)", "FP8 E5M2 + row-sum (FP32)", C_TEAL, ""),
        ("FP8 E4M3\n+ row-sum", "FP8 E4M3 + row-sum (FP64)", "FP8 E4M3 + row-sum (FP32)", C_TEAL, ""),
    ]
    labels = [s[0] for s in bar_spec]
    iters = [int(exp_b[s[1]]["coarse_iters"]) for s in bar_spec]
    colors = [s[3] for s in bar_spec]
    hatches = [s[4] for s in bar_spec]

    xb = np.arange(len(labels))
    for ax_sub in (ax2_top, ax2_bot):
        bars = ax_sub.bar(xb, iters, width=0.58, color=colors, edgecolor="#222222", linewidth=0.6)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

    ax2_top.axhline(400, color=C_ORANGE, ls="--", lw=0.9, alpha=0.75)
    ax2_top.set_ylim(360, 475)
    ax2_top.set_yticks([380, 400])
    ax2_bot.set_ylim(0, 26)
    ax2_bot.set_yticks([0, 5, 10, 15, 20, 25])

    ax2_top.spines["bottom"].set_visible(False)
    ax2_top.tick_params(bottom=False, labelbottom=False)
    ax2_top.set_title("(b) Coarse-Grid PCG ($A_c$ Dynamic-Range Floor)", fontsize=9.5)

    # Diagonal axis-break marks on left spine
    d = 0.014
    kwargs_top = dict(transform=ax2_top.transAxes, color="#222222", clip_on=False, lw=0.9)
    ax2_top.plot((-d, +d), (-3.0 * d, +3.0 * d), **kwargs_top)
    kwargs_bot = dict(transform=ax2_bot.transAxes, color="#222222", clip_on=False, lw=0.9)
    ax2_bot.plot((-d, +d), (1.0 - d, 1.0 + d), **kwargs_bot)

    texts = []
    for _, key64, key32, _, _ in bar_spec:
        r64 = exp_b[key64]
        it64 = int(r64["coarse_iters"])
        st64 = r64["coarse_status"]
        if st64 == "NON-SPD BREAKDOWN":
            texts.append(f"Iter {it64}\nBREAK")
        elif st64 == "MAX_ITERS":
            texts.append(f"{it64} it\nMAX ITERS")
        elif key32 is not None and int(exp_b[key32]["coarse_iters"]) != it64:
            it32 = int(exp_b[key32]["coarse_iters"])
            texts.append(f"{it64} FP64-d\n({it32} FP32-d)")
        elif key32 is not None:
            texts.append(f"{it64} it\n(FP64/32-d)")
        else:
            texts.append(f"{it64} it\n(PASS)")

    for xi, val, txt, col in zip(xb, iters, texts, colors):
        if val > 100:
            ax2_top.text(
                xi, val + 8, txt, ha="center", va="bottom", fontsize=7.3,
                fontweight="bold", color="#9a4a12",
            )
        else:
            ax2_bot.text(
                xi, val + 0.7, txt, ha="center", va="bottom", fontsize=6.9,
                fontweight="bold" if col in (C_TEAL, C_RED) else "normal",
                color=C_RED if "BREAK" in txt else ("#085c53" if col == C_TEAL else "#1a1a1a"),
            )

    ax2_bot.set_xticks(xb)
    ax2_bot.set_xticklabels(labels, fontsize=6.5)
    ax2_bot.set_ylabel("PCG Iterations to $\\leq 10^{-8}$ (broken axis; 400-it cap)")
    ax2_bot.yaxis.set_label_coords(-0.105, 0.70)

    save(fig, "fig1_quant_cliff_and_coarse")


def make_fig2() -> None:
    """Figure 2: Matched-dFFI AmgX and four-engine GPU pure-solve speedup computed from CSVs."""
    amgx_rows = load_timed_rows("amgx_8cases.csv")
    prior_rows = load_timed_rows("prior_art_8cases.csv")

    def amgx_median_solve(case: str, method: str) -> float:
        vals = [
            float(r["solve_ms"]) for r in amgx_rows
            if r["case"] == case and r["method"] == method and int(r["pass"]) == 1
        ]
        if not vals:
            raise ValueError(f"Missing passing AmgX rows for {case} {method}")
        return float(st.median(vals))

    def prior_best_solve(case: str, engine: str, methods: tuple[str, ...]) -> float:
        best = float("nan")
        for m in methods:
            m_rows = [r for r in prior_rows if r["case"] == case and r["engine"] == engine and r["method"] == m]
            if m_rows and all(int(r["pass"]) == 1 for r in m_rows):
                med = float(st.median(float(r["solve_ms"]) for r in m_rows))
                if np.isnan(best) or med < best:
                    best = med
        return best

    # Verify Ginkgo 3-P outcomes directly from prior_art_8cases.csv
    gko_3p_timeouts = 0
    gko_3p_fails = 0
    for c in CASES_ORDER:
        r3p = [r for r in prior_rows if r["case"] == c and r["engine"] == "ginkgo" and r["method"] == "mg_isai_3p"]
        check(bool(r3p) and all(int(r["pass"]) == 0 for r in r3p), f"Unexpected pass in mg_isai_3p for {c}")
        if any(r.get("note") == "timeout" for r in r3p):
            gko_3p_timeouts += 1
        else:
            gko_3p_fails += 1
    check(gko_3p_timeouts == 2 and gko_3p_fails == 6, f"Unexpected mg_isai_3p counts: {gko_3p_timeouts}/{gko_3p_fails}")

    # Matched dFFI baseline: AmgX JACOBI_L1 in dFFI (FP32 inner V-cycle, FP64 outer PCG)
    amgx_mix_jac = np.array([amgx_median_solve(c, "mix_jacobi") for c in CASES_ORDER])
    bitfsai_int4 = np.array([amgx_median_solve(c, "mix_int4_rep") for c in CASES_ORDER])
    bitfsai_fp16 = np.array([amgx_median_solve(c, "mix_fp16_rep") for c in CASES_ORDER])
    hypre_best = np.array([prior_best_solve(c, "hypre", ("l1_jacobi", "fsai")) for c in CASES_ORDER])
    ginkgo_best = np.array([
        prior_best_solve(c, "ginkgo", ("mg_jacobi_f32", "mg_isai_f32", "mg_isai_3p"))
        for c in CASES_ORDER
    ])

    sp_int4 = amgx_mix_jac / bitfsai_int4
    sp_fp16 = amgx_mix_jac / bitfsai_fp16
    sp_hypre = amgx_mix_jac / hypre_best
    sp_ginkgo = amgx_mix_jac / ginkgo_best

    cases_labels = [
        "128³\nGrid", "192³\nGrid", "P1 Tet\n48³", "Elast.\n28³",
        "rail\n79841", "bcsstk\n14", "bcsstk\n15", "bcsstk\n16",
    ]

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    x = np.arange(len(cases_labels))
    w = 0.20

    ax.bar(x - 1.5 * w, sp_int4, width=w, color=C_TEAL, edgecolor="#1c1c1c", lw=0.6,
           label="BitFSAI INT4-spl in AmgX (matched dFFI, 8/8 PASS)")
    ax.bar(x - 0.5 * w, sp_fp16, width=w, color=C_PURPLE, edgecolor="#1c1c1c", lw=0.6,
           label="BitFSAI FP16 in AmgX (matched dFFI, 8/8 PASS)")
    ax.bar(x + 0.5 * w, sp_hypre, width=w, color=C_BLUE, edgecolor="#1c1c1c", lw=0.6,
           label="LLNL hypre Best (BoomerAMG l1-Jacobi / GPU FSAI, 8/8 PASS)")
    ax.bar(
        x + 1.5 * w, np.nan_to_num(sp_ginkgo, nan=0.0), width=w, color=C_ORANGE, edgecolor="#1c1c1c", lw=0.6,
        label=f"Ginkgo develop (2.0.0-dev) Best (FP32 PGM: 6/8 PASS; 3-P: {gko_3p_timeouts} TIMEOUT, {gko_3p_fails} NaN FAIL)",
    )

    ax.axhline(1.0, color=C_SLATE, ls="--", lw=1.2,
               label="Matched dFFI AmgX JACOBI_L1 baseline (1.0$\\times$)")
    ax.set_yscale("log")
    ax.set_ylim(0.06, 17.0)
    ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    for idx, (xi, v) in enumerate(zip(x - 1.5 * w, sp_int4)):
        ax.text(xi, v * 1.09, f"{v:.2f}$\\times$", ha="center", va="bottom",
                fontsize=6.9, fontweight="bold", color="#085c53")
    for xi, v in zip(x + 1.5 * w, sp_ginkgo):
        if np.isnan(v):
            ax.text(xi, 0.072, "FAIL\n(0/5)", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color=C_RED)

    ax.set_xticks(x)
    ax.set_xticklabels(cases_labels)
    ax.set_ylabel("Pure Solve Speedup vs. Matched dFFI AmgX JACOBI_L1 (log)")
    ax.set_title("Matched-Precision dFFI AmgX & Four-Engine GPU Pure Solve Speedup (RTX Spark N1X)")
    ax.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout()
    save(fig, "fig2_four_engine_solve")


def make_fig3() -> None:
    """Figure 3: Conventional vs. mirrored factor memory reduction & matched-dFFI AmgX ablation."""
    s8_rows = load_timed_rows("standalone_8cases.csv")
    amgx_rows = load_timed_rows("amgx_8cases.csv")

    def weight_and_levels(case: str) -> tuple[int, int, int, int]:
        wb = {}
        lvls = None
        for m in ("double", "half", "int4"):
            rs = [r for r in s8_rows if r["case"] == case and r["method"] == m]
            vals = {int(r["weight_bytes"]) for r in rs}
            check(len(vals) == 1, f"Inconsistent weight_bytes for {case}/{m}")
            wb[m] = next(iter(vals))
            if m == "int4":
                lvls = int(rs[0]["factored_levels"].split("/")[1])
        return wb["double"], wb["half"], wb["int4"], lvls

    with open(RESULTS / "pattern_storage.csv", newline="", encoding="utf-8") as fh:
        pattern_rows = {r["case_label"]: r for r in csv.DictReader(fh)}
    check(len(pattern_rows) == len(CASES_ORDER), "pattern_storage.csv must cover all 8 workloads")

    r_conv64_val = []
    r_conv64_vi = []
    r_mir64 = []
    r_mir16 = []
    for c in CASES_ORDER:
        w64, w16, wi4, lvls = weight_and_levels(c)
        pr = pattern_rows[c]
        check(int(pr["int4_weight_bytes"]) == wi4, f"pattern_storage int4 mismatch on {c}")
        check(int(pr["fp64_mirrored_bytes"]) == w64, f"pattern_storage fp64 mismatch on {c}")
        check(int(pr["fp16_mirrored_bytes"]) == w16, f"pattern_storage fp16 mismatch on {c}")
        check(int(pr["levels"]) == lvls, f"pattern_storage level count mismatch on {c}")
        nnz = w64 // 8
        n_rows = (w16 - 2 * nnz) // 8
        nnz_l = (nnz + n_rows) // 2
        check(int(pr["full_lower_nnz"]) == (int(pr["nnz"]) + int(pr["rows"])) // 2,
              f"pattern_storage full_lower_nnz identity mismatch on {c}")
        conv_val = 8 * nnz_l
        conv_vi = 12 * nnz_l + 4 * (n_rows + lvls)
        r_conv64_val.append(conv_val / wi4)
        r_conv64_vi.append(conv_vi / wi4)
        r_mir64.append(w64 / wi4)
        r_mir16.append(w16 / wi4)

    r_conv64_val = np.array(r_conv64_val)
    r_conv64_vi = np.array(r_conv64_vi)
    r_mir64 = np.array(r_mir64)
    r_mir16 = np.array(r_mir16)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 3.85), gridspec_kw={"width_ratios": [1.38, 1.0]}
    )

    cases = ["128³", "192³", "Tet 48³", "Elast.", "rail", "bcsstk14", "bcsstk15", "bcsstk16"]
    x = np.arange(len(cases))
    w = 0.20

    ax1.bar(
        x - 1.5 * w, r_conv64_val, width=w, color=C_NAVY, edgecolor="#1c1c1c", lw=0.5,
        label=f"vs. full-pattern conv. FP64 val ({r_conv64_val.min():.2f}$\\times$–{r_conv64_val.max():.2f}$\\times$)",
    )
    ax1.bar(
        x - 0.5 * w, r_conv64_vi, width=w, color=C_BLUE, edgecolor="#1c1c1c", lw=0.5,
        label=f"vs. full-pattern conv. FP64 val+idx ({r_conv64_vi.min():.2f}$\\times$–{r_conv64_vi.max():.2f}$\\times$)",
    )
    bars_mir = ax1.bar(
        x + 0.5 * w, r_mir64, width=w, color=C_SLATE, edgecolor="#1c1c1c", lw=0.5,
        label=f"vs. mirrored FP64 val ({r_mir64.min():.2f}$\\times$–{r_mir64.max():.2f}$\\times$)",
    )
    for b in bars_mir:
        b.set_hatch("///")
    ax1.bar(
        x + 1.5 * w, r_mir16, width=w, color=C_TEAL, edgecolor="#1c1c1c", lw=0.5,
        label=f"vs. mirrored FP16 ({r_mir16.min():.2f}$\\times$–{r_mir16.max():.2f}$\\times$)",
    )
    for xi, v in zip(x - 0.5 * w, r_conv64_vi):
        ax1.text(xi, v + 0.25, f"{v:.1f}$\\times$", ha="center", va="bottom",
                 fontsize=6.6, fontweight="bold", color=C_BLUE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cases, fontsize=7.8)
    ax1.set_ylabel("Smoother Factor Storage Reduction Ratio")
    ax1.set_ylim(0, 17.5)
    ax1.set_title(
        "(a) Smoother Factor Memory Reduction\n(Full-Pattern Conv. FP64 vs. Mirrored, 0 B Extra Idx)",
        fontsize=8.9,
    )
    ax1.legend(loc="upper left", fontsize=6.7, frameon=True, facecolor="white", framealpha=0.94)

    # Panel (b): Matched-precision dFFI Theorem 3 re-projection ablation in AmgX
    stiff_cases = ["bcsstk14.mtx", "bcsstk15.mtx"]
    stiff_labels = ["bcsstk14\n(AmgX dFFI)", "bcsstk15\n(AmgX dFFI)"]

    def amgx_med(case: str, method: str) -> float:
        rows = [r for r in amgx_rows if r["case"] == case and r["method"] == method and int(r["pass"]) == 1]
        check(len(rows) == 5, f"Expected 5 passing AmgX rows for {case}/{method}, got {len(rows)}")
        return float(st.median(float(r["solve_ms"]) for r in rows))

    for c in stiff_cases:
        norep = [r for r in amgx_rows if r["case"] == c and r["method"] == "mix_int4_norep"]
        check(
            len(norep) == 5 and all(int(r["pass"]) == 0 and r["trigger"] == "rz" for r in norep),
            f"Unexpected mix_int4_norep rows for {c}",
        )

    jac_ms = [amgx_med(c, "mix_jacobi") for c in stiff_cases]
    fp16_ms = [amgx_med(c, "mix_fp16_rep") for c in stiff_cases]
    int4_rep_ms = [amgx_med(c, "mix_int4_rep") for c in stiff_cases]

    xs = np.arange(len(stiff_labels))
    w2 = 0.21
    ax2.bar(xs - 1.5 * w2, jac_ms, width=w2, color=C_SLATE, edgecolor="#1c1c1c", lw=0.6,
            label="JACOBI_L1 (dFFI: 5/5 passed)")
    failure_x = xs - 0.5 * w2
    ax2.plot(failure_x, [4, 4], marker="x", ms=7.5, mew=2, linestyle="None",
             color=C_RED, label="INT4-spl dFFI (no re-proj: 0/5 passed, rz guard)")
    ax2.bar(xs + 0.5 * w2, fp16_ms, width=w2, color=C_BLUE, edgecolor="#1c1c1c", lw=0.6,
            label="FP16 dFFI (with re-proj: 5/5 passed)")
    ax2.bar(xs + 1.5 * w2, int4_rep_ms, width=w2, color=C_TEAL, edgecolor="#1c1c1c", lw=0.6,
            label="INT4-spl dFFI (with re-proj: 5/5 passed)")

    callout_y = [66.0, 102.0]
    for i, (xj, xf, xi_val) in enumerate(zip(jac_ms, fp16_ms, int4_rep_ms)):
        ax2.text(xs[i] - 1.5 * w2, xj + 4, f"{xj:.1f}", ha="center", va="bottom", fontsize=7.3)
        ax2.annotate(
            "0/5 passed\n(rz guard)",
            xy=(failure_x[i], 9), xytext=(failure_x[i] + 0.08, callout_y[i]),
            ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=C_RED,
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9),
        )
        ax2.text(xs[i] + 0.5 * w2, xf + 4, f"{xf:.1f}", ha="center", va="bottom",
                 fontsize=7.3, color=C_BLUE)
        ax2.text(xs[i] + 1.5 * w2, xi_val + 4, f"{xi_val:.1f}\n({xj/xi_val:.1f}$\\times$)",
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#085c53")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(stiff_labels)
    ax2.set_ylabel("Pure Solve Time (ms, lower = faster)")
    ax2.set_ylim(0, 290)
    ax2.set_title("(b) Matched-dFFI Theorem 3 Re-Projection Ablation", fontsize=9.0)
    ax2.legend(loc="upper left", ncol=1, fontsize=6.9, frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout(w_pad=2.1)
    save(fig, "fig3_memory_and_reproj")


def make_fig4() -> None:
    """Figure 4: Multilevel FP8 E5M2 coarse operators & sparse-depth polynomial boost."""
    sm_rows = load_timed_rows("standalone_multilevel.csv")
    s8_rows = load_timed_rows("standalone_8cases.csv")

    cfg_keys = ["vcycle_fp32", "vcycle_fp8", "poly4_fp32", "poly4_shift2_fp32", "poly4_shift2_fp8"]
    cfgs = [
        "V-cycle\nFP32 $A_c$",
        "V-cycle\nFP8 E5M2",
        "Poly4\nFP32 $A_c$",
        "Poly4@2\nFP32 $A_c$",
        "Poly4@2 +\nFP8 E5M2",
    ]

    def ml_stats(grid: str, cfg: str) -> tuple[float, int, int, int, int]:
        rows = [r for r in sm_rows if r["case"] == grid and r["config"] == cfg and int(r["pass"]) == 1]
        check(bool(rows), f"Missing multilevel rows for {grid}/{cfg}")
        solve_med = float(st.median(float(r["solve_ms"]) for r in rows))
        iters_med = int(st.median(int(r["iters"]) for r in rows))
        wb = int(rows[0]["weight_bytes"])
        cb = int(rows[0]["coarse_bytes"])
        rb = int(rows[0]["resident_bytes"])
        return solve_med, iters_med, wb, cb, rb

    g128_stats = [ml_stats("grid128", k) for k in cfg_keys]
    g192_stats = [ml_stats("grid192", k) for k in cfg_keys]

    g128_ms = np.array([s[0] for s in g128_stats])
    g128_it = [s[1] for s in g128_stats]
    g192_ms = np.array([s[0] for s in g192_stats])
    g192_it = [s[1] for s in g192_stats]

    sp128 = g128_ms[0] / g128_ms
    sp192 = g192_ms[0] / g192_ms

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 3.75), gridspec_kw={"width_ratios": [1.30, 1.05]}
    )

    x = np.arange(len(cfgs))
    w = 0.36
    ax1.bar(x - w / 2, sp128, width=w, color=C_BLUE, edgecolor="#1c1c1c", lw=0.6,
            label="128³ Grid (2.1M DOFs, $10^4$ contrast)")
    ax1.bar(x + w / 2, sp192, width=w, color=C_TEAL, edgecolor="#1c1c1c", lw=0.6,
            label="192³ Grid (7.1M DOFs, $10^4$ contrast)")
    ax1.axhline(1.0, color=C_SLATE, ls="--", lw=1.1)

    for xi, v, it in zip(x - w / 2, sp128, g128_it):
        ax1.text(xi, v + 0.03, f"{v:.2f}$\\times$\n({it} it)", ha="center", va="bottom",
                 fontsize=7.0, color=C_BLUE, fontweight="bold" if v > 1.25 else "normal")
    for xi, v, it in zip(x + w / 2, sp192, g192_it):
        ax1.text(xi, v + 0.03, f"{v:.2f}$\\times$\n({it} it)", ha="center", va="bottom",
                 fontsize=7.0, color="#085c53", fontweight="bold" if v > 1.25 else "normal")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, fontsize=7.9)
    ax1.set_ylabel("GPU Solve Speedup vs. INT4-spl + FP32 $A_c$")
    ax1.set_ylim(0, 1.92)
    ax1.set_title("(a) FP8 E5M2 Coarse $A_c$ & Sparse-Depth Poly Boost (CUDA Graph)", fontsize=9.0)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92)

    # Panel (b): Active smoother & coarse-A_c value buffers on 192³ (MiB) directly from CSVs
    mib = float(1024 ** 2)
    w_fp64_mir_b = next(int(r["weight_bytes"]) for r in s8_rows if r["case"] == "grid192" and r["method"] == "double")
    w_fp16_mir_b = next(int(r["weight_bytes"]) for r in s8_rows if r["case"] == "grid192" and r["method"] == "half")
    r_i4_192 = next(r for r in s8_rows if r["case"] == "grid192" and r["method"] == "int4")
    lvls_192 = int(r_i4_192["factored_levels"].split("/")[1])
    nnz_192 = w_fp64_mir_b // 8
    n_192 = (w_fp16_mir_b - 2 * nnz_192) // 8
    nnz_l_192 = (nnz_192 + n_192) // 2
    w_fp64_conv_vi = (12 * nnz_l_192 + 4 * (n_192 + lvls_192)) / mib
    w_fp64_mir = w_fp64_mir_b / mib

    _, _, w_i4_b, c_fp32_b, res_fp32_b = g192_stats[0]
    _, _, _, c_fp8_b, res_fp8_b = g192_stats[1]
    check(w_i4_b + c_fp32_b == res_fp32_b, "resident_bytes != weight_bytes + coarse_bytes (FP32)")
    check(w_i4_b + c_fp8_b == res_fp8_b, "resident_bytes != weight_bytes + coarse_bytes (FP8)")

    w_i4 = w_i4_b / mib
    c_fp32 = c_fp32_b / mib
    c_fp8 = c_fp8_b / mib

    tiers = [
        "Mirr. FP64\n+ FP32 $A_c$",
        "Conv. FP64\n(v+i)+FP32 $A_c$",
        "INT4-spl\n+ FP32 $A_c$",
        "INT4-spl +\nFP8 E5M2 $A_c$",
    ]
    fac_mb = np.array([w_fp64_mir, w_fp64_conv_vi, w_i4, w_i4])
    ac_mb = np.array([c_fp32, c_fp32, c_fp32, c_fp8])
    xt = np.arange(len(tiers))
    ax2.bar(xt, fac_mb, width=0.56, color=C_BLUE, edgecolor="#1c1c1c", lw=0.6,
            label="Active smoother factors (0 B extra idx for INT4)")
    ax2.bar(xt, ac_mb, width=0.56, bottom=fac_mb, color=C_TEAL, edgecolor="#1c1c1c", lw=0.6,
            label="Active coarse $A_c$ values ($j \\geq 1$)")

    totals = fac_mb + ac_mb
    vram_saved = totals[2] - totals[3]
    vram_ratio = totals[2] / totals[3]
    for xi, tot in enumerate(totals):
        if xi == 0:
            lbl = f"{tot:.1f} MiB\n(mirr. FP64)"
        elif xi == 1:
            lbl = f"{tot:.1f} MiB\n(conv. val+idx)"
        elif xi == 2:
            lbl = f"{tot:.1f} MiB\n({totals[1]/tot:.2f}$\\times$ vs conv.)"
        else:
            lbl = f"{tot:.1f} MiB\n({totals[1]/tot:.2f}$\\times$ conv. / {totals[0]/tot:.2f}$\\times$ mirr.;\n{vram_ratio:.2f}$\\times$ / -{vram_saved:.0f} MiB)"
        ax2.text(
            xi + (0.04 if xi == 3 else 0.0), tot + 25, lbl, ha="center", va="bottom",
            fontsize=6.2 if xi == 3 else 6.5,
            fontweight="bold" if xi == 3 else "normal",
            color="#085c53" if xi == 3 else "#1a1a1a",
        )

    ax2.set_xticks(xt)
    ax2.set_xticklabels(tiers, fontsize=7.2)
    ax2.set_xlim(-0.55, 3.68)
    ax2.set_ylabel("Smoother + Coarse-$A_c$ Buffers on 192³ (MiB)")
    ax2.set_ylim(0, 1980)
    ax2.set_title("(b) Smoother & Coarse-$A_c$ Value Buffers (192³)", fontsize=9.0)
    ax2.legend(loc="upper right", fontsize=6.9, frameon=True, facecolor="white", framealpha=0.92)

    fig.tight_layout(w_pad=2.1)
    save(fig, "fig4_gpu_multilevel")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print(f"Generated 4 vector figures (SVG + PDF) in {OUT}")
