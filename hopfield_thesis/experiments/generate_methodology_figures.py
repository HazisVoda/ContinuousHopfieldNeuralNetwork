"""
Generate methodology pipeline diagrams for the thesis Work Methodology section.

Five PNG figures (matplotlib):
  methodology_fig1_chnn_retrieval.png     — CHNN update rule / retrieval mechanism
  methodology_fig2_onepixel_attack.png    — One-pixel attack protocol (WB attacker)
  methodology_fig3_twostage_protocol.png  — Two-stage experimental protocol & conditional success
  methodology_fig4_centering_pipeline.png — Global-mean centering preprocessing
  methodology_fig5_cross_dataset_map.png  — Cross-dataset vulnerability overview

Two Mermaid source files (auto-layout flowcharts, open in VS Code):
  methodology_fig2_onepixel_attack.mmd
  methodology_fig3_twostage_protocol.mmd
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── output dir ──────────────────────────────────────────────────────────────
FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────────────
BLUE    = "#1A5276"
LBLUE   = "#D6EAF8"
DBLUE   = "#154360"
RED     = "#B03A2E"
LRED    = "#FADBD8"
GREEN   = "#1E6B3C"
LGREEN  = "#D5F5E3"
ORANGE  = "#A04000"
LORANGE = "#FAE5D3"
PURPLE  = "#6C3483"
LPURPLE = "#E8DAEF"
GRAY    = "#2C3E50"
LGRAY   = "#F2F3F4"
MID     = "#5D6D7E"
WHITE   = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ── shared helpers ───────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, text, fc=LBLUE, ec=BLUE, fs=10, bold=False,
         color=GRAY, lw=1.5, va="center", ha="center"):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, clip_on=False,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha=ha, va=va, fontsize=fs, color=color,
            weight=weight, zorder=4, clip_on=False, multialignment="center")
    return patch


def _arr(ax, x1, y1, x2, y2, color=BLUE, lw=1.5, label="", lfs=9, ldy=0.04):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), zorder=2, annotation_clip=False,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12,
                        connectionstyle="arc3,rad=0"),
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + ldy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=lfs,
                color=MID, style="italic")


def _title(ax, text):
    ax.set_title(text, fontsize=12, fontweight="bold", color=GRAY, pad=10)


def _pattern_grid(ax, left, bottom, cell=0.018, rows=8, cols=8,
                  colors=None, rng=None, alpha=1.0):
    """Draw a schematic image as rows x cols coloured cells."""
    if rng is None:
        rng = np.random.default_rng(0)
    if colors is None:
        colors = rng.uniform(0.3, 0.95, (rows, cols))
    for r in range(rows):
        for c in range(cols):
            val = float(colors[r, c]) if colors.ndim == 2 else float(colors[0])
            fc = plt.cm.gray(val)
            rect = mpatches.Rectangle(
                (left + c * cell, bottom + (rows - 1 - r) * cell),
                cell * 0.92, cell * 0.92,
                linewidth=0, facecolor=fc, alpha=alpha, zorder=3,
            )
            ax.add_patch(rect)
    w = cols * cell
    h = rows * cell
    border = FancyBboxPatch(
        (left, bottom), w, h,
        boxstyle="square,pad=0", linewidth=1.2, edgecolor=BLUE,
        facecolor="none", zorder=4,
    )
    ax.add_patch(border)
    return w, h


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — CHNN Retrieval Mechanism  (redesigned: clean two-row layout)
# ═══════════════════════════════════════════════════════════════════════════

def fig_chnn_retrieval():
    W, H = 13, 5.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    _title(ax, "Figure M1: Continuous Modern Hopfield Network — Retrieval Mechanism")

    rng = np.random.default_rng(42)
    cell = 0.046

    # ────────────────────────────────────────────────────────────────────────
    # ROW 1  (y ≈ 3.8):  query ξ  +  5 stored patterns  +  retrieved ξ′
    # ────────────────────────────────────────────────────────────────────────

    # True pattern (index 2): digit-like vertical bar
    pat_true = np.full((8, 8), 0.12)
    pat_true[1:7, 3:5] = 0.92

    pat_colors = []
    for i in range(5):
        if i == 2:
            pat_colors.append(pat_true.copy())
        else:
            p = rng.uniform(0.1, 0.9, (8, 8))
            pat_colors.append(p)

    # Query = noisy version of true pattern
    noisy = pat_true + rng.normal(0, 0.25, (8, 8))
    noisy = np.clip(noisy, 0, 1)

    # y-baseline for the pattern row
    pat_y = 3.55
    pat_w = 8 * cell  # ≈ 0.368

    # Query ξ  (leftmost)
    q_x = 0.45
    _pattern_grid(ax, q_x, pat_y, cell=cell, rows=8, cols=8, colors=noisy)
    ax.text(q_x + pat_w / 2, pat_y - 0.24, "ξ  (partial query)",
            ha="center", va="top", fontsize=9.5, color=GRAY, weight="bold")

    # Arrow  query → patterns block
    _arr(ax, q_x + pat_w + 0.08, pat_y + pat_w / 2,
         2.55, pat_y + pat_w / 2, color=MID, lw=1.3)

    # Stored patterns X (5 patterns, centred around x=4.3)
    n_pat = 5
    gap = 0.12
    block_w = n_pat * pat_w + (n_pat - 1) * gap
    pat_xs = [2.62 + i * (pat_w + gap) for i in range(n_pat)]

    for i, (pl, pc) in enumerate(zip(pat_xs, pat_colors)):
        _pattern_grid(ax, pl, pat_y, cell=cell, rows=8, cols=8, colors=pc)
        if i == 2:
            # Highlight true pattern
            ax.add_patch(FancyBboxPatch(
                (pl - 0.03, pat_y - 0.03),
                pat_w + 0.06, pat_w + 0.06,
                boxstyle="square,pad=0", linewidth=2.3,
                edgecolor=RED, facecolor="none", zorder=5,
            ))
            ax.text(pl + pat_w / 2, pat_y - 0.24, "xₙ  (true)",
                    ha="center", va="top", fontsize=8.5, color=RED, style="italic")
        else:
            idx = i + 1 if i < 2 else i + 1
            ax.text(pl + pat_w / 2, pat_y - 0.22, f"x{idx}",
                    ha="center", va="top", fontsize=8, color=MID)

    # Bracket label above patterns
    blk_mid = pat_xs[0] + block_w / 2
    ax.annotate("", xy=(pat_xs[0], pat_y + pat_w + 0.26),
                xytext=(pat_xs[-1] + pat_w, pat_y + pat_w + 0.26),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="|-|", color=BLUE, lw=1.3,
                               mutation_scale=5))
    ax.text(blk_mid, pat_y + pat_w + 0.45,
            "X  =  [ x₁   x₂  …  xₙ  …  x_N ]        stored patterns matrix  (d × N)",
            ha="center", va="bottom", fontsize=10.5, color=GRAY, weight="bold")

    # Arrow  patterns block → retrieved
    _arr(ax, pat_xs[-1] + pat_w + 0.08, pat_y + pat_w / 2,
         8.52, pat_y + pat_w / 2, color=MID, lw=1.3)

    # Retrieved ξ′  (= same digit, clean)
    ret_x = 8.6
    _pattern_grid(ax, ret_x, pat_y, cell=cell, rows=8, cols=8, colors=pat_true)
    ax.add_patch(FancyBboxPatch(
        (ret_x - 0.03, pat_y - 0.03),
        pat_w + 0.06, pat_w + 0.06,
        boxstyle="square,pad=0", linewidth=2.3,
        edgecolor=GREEN, facecolor="none", zorder=5,
    ))
    ax.text(ret_x + pat_w / 2, pat_y - 0.24, "ξ′  (retrieved)",
            ha="center", va="top", fontsize=9.5, color=GREEN, weight="bold")

    # ────────────────────────────────────────────────────────────────────────
    # ROW 2  (y ≈ 1.65):  computation pipeline boxes
    # ────────────────────────────────────────────────────────────────────────
    y_pipe = 1.65
    bw, bh = 1.72, 0.82

    steps = [
        (1.55,  "① dot products\nβ · Xᵀ · ξ\n→ scores  (N×1)", LBLUE,   BLUE),
        (3.85,  "② softmax\n(column-wise)\n→ weights w (N×1)", LPURPLE, PURPLE),
        (6.15,  "③ combination\nξ′ = X · w\n→ (d×1)",          LGREEN,  GREEN),
        (8.45,  "④ class ID\nargmax cosine\n(ξ′, X cols)",      LORANGE, ORANGE),
        (10.75, "⑤ label\nRetrieved\npattern index",            LGREEN,  GREEN),
    ]

    for cx, txt, fc, ec in steps:
        _box(ax, cx, y_pipe, bw, bh, txt, fc=fc, ec=ec, fs=9.5)

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + bw / 2
        x2 = steps[i + 1][0] - bw / 2
        _arr(ax, x1, y_pipe, x2, y_pipe)

    # ────────────────────────────────────────────────────────────────────────
    # Vertical connectors:  pattern row  ↓  pipeline boxes
    # ────────────────────────────────────────────────────────────────────────
    # query ξ → step ①
    ax.plot([q_x + pat_w / 2, q_x + pat_w / 2, steps[0][0]],
            [pat_y, y_pipe + bh / 2 + 0.12, y_pipe + bh / 2 + 0.12],
            color=MID, lw=1.2, ls="--", zorder=2)
    ax.annotate("", xy=(steps[0][0], y_pipe + bh / 2),
                xytext=(steps[0][0], y_pipe + bh / 2 + 0.12),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.2,
                               mutation_scale=10))
    ax.text(steps[0][0] - 0.55, y_pipe + bh / 2 + 0.06, "ξ",
            ha="center", va="bottom", fontsize=10, color=MID, style="italic")

    # X → step ① (dot product with X^T)
    x1_pat = blk_mid - 1.0
    ax.plot([x1_pat, x1_pat, steps[0][0] + 0.3],
            [pat_y, y_pipe + bh / 2 + 0.36, y_pipe + bh / 2 + 0.36],
            color=BLUE, lw=1.2, ls="--", zorder=2)
    ax.annotate("", xy=(steps[0][0] + 0.3, y_pipe + bh / 2),
                xytext=(steps[0][0] + 0.3, y_pipe + bh / 2 + 0.36),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2,
                               mutation_scale=10))
    ax.text(steps[0][0] + 0.65, y_pipe + bh / 2 + 0.2, "X",
            ha="center", va="bottom", fontsize=10, color=BLUE, weight="bold")

    # X → step ③ (linear combination X·w)
    x3_pat = blk_mid + 0.8
    ax.plot([x3_pat, x3_pat, steps[2][0]],
            [pat_y, y_pipe + bh / 2 + 0.58, y_pipe + bh / 2 + 0.58],
            color=GREEN, lw=1.2, ls="--", zorder=2)
    ax.annotate("", xy=(steps[2][0], y_pipe + bh / 2),
                xytext=(steps[2][0], y_pipe + bh / 2 + 0.58),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2,
                               mutation_scale=10))
    ax.text(steps[2][0] + 0.45, y_pipe + bh / 2 + 0.35, "X",
            ha="center", va="bottom", fontsize=10, color=GREEN, weight="bold")

    # retrieved ξ′ → step ⑤
    ax.plot([ret_x + pat_w / 2, ret_x + pat_w / 2, steps[4][0]],
            [pat_y, y_pipe + bh / 2 + 0.12, y_pipe + bh / 2 + 0.12],
            color=MID, lw=1.2, ls="--", zorder=2)
    ax.annotate("", xy=(steps[4][0], y_pipe + bh / 2),
                xytext=(steps[4][0], y_pipe + bh / 2 + 0.12),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.2,
                               mutation_scale=10))

    # ────────────────────────────────────────────────────────────────────────
    # Formula banner
    # ────────────────────────────────────────────────────────────────────────
    ax.text(W / 2, 0.42,
            "ξ′  =  X · softmax( β · Xᵀ · ξ )        "
            "single step  |  β = 8.0  |  d = 784 (MNIST/FMNIST)  or  1024 (CIFAR)",
            ha="center", va="center", fontsize=10.5, color=DBLUE,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.32", fc=LBLUE, ec=BLUE, lw=1.5))

    fig.tight_layout(rect=[0, 0.0, 1, 1])
    out = FIGURES / "methodology_fig1_chnn_retrieval.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — One-Pixel Attack Protocol
# ═══════════════════════════════════════════════════════════════════════════

def fig_onepixel_attack():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _title(ax, "Figure M2: White-Box One-Pixel Attack Protocol")

    bw, bh = 3.2, 0.62

    # ── Main column (left-aligned, x=4.5 center) ─────────────────────────────
    CX = 4.5
    nodes = [
        (6.4,  "Original query  q\n(MNIST/FMNIST: ℝ⁷⁸⁴  |  CIFAR: ℝ¹⁰²⁴)", LBLUE,   BLUE,   True),
        (5.55, "Enumerate candidate queries\n"
               "d pixel locations  ×  5 values  {0, 0.25, 0.5, 0.75, 1}\n"
               "→  3,920 (MNIST)  or  5,120 (CIFAR) candidates",
               LGRAY,   GRAY,   False),
        (4.65, "Single batched retrieve() call\nretrieved  =  net.retrieve(candidates)",
               LBLUE,   BLUE,   False),
        (3.75, "Cosine similarity to true pattern xₙ\n"
               "for each candidate's retrieved vector",
               LPURPLE, PURPLE, False),
    ]
    for cy, txt, fc, ec, bold in nodes:
        _box(ax, CX, cy, bw, bh, txt, fc=fc, ec=ec, fs=9.5, bold=bold)

    # Arrows between them
    for i in range(len(nodes) - 1):
        y1 = nodes[i][0] - bh / 2
        y2 = nodes[i + 1][0] + bh / 2
        _arr(ax, CX, y1, CX, y2)

    # ── Diamond decision ─────────────────────────────────────────────────────
    dy = 2.9
    diamond = plt.Polygon(
        [[CX, dy + 0.38], [CX + 0.95, dy], [CX, dy - 0.38], [CX - 0.95, dy]],
        closed=True, linewidth=1.5, edgecolor=ORANGE, facecolor=LORANGE, zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(CX, dy, "Δcos < −1e‑4?", ha="center", va="center",
            fontsize=9.5, color=ORANGE, weight="bold", zorder=4)

    _arr(ax, CX, 3.75 - bh / 2, CX, dy + 0.38)

    # Yes branch (left)
    _arr(ax, CX - 0.95, dy, CX - 2.5, dy, color=GREEN, lw=1.3)
    ax.text(CX - 1.72, dy + 0.1, "Yes", fontsize=9, color=GREEN,
            weight="bold", ha="center")
    _box(ax, CX - 3.2, dy, 1.3, 0.55,
         "argmin Δcos\n→ pixel k*", fc=LGREEN, ec=GREEN, fs=9)

    # No branch (right)
    _arr(ax, CX + 0.95, dy, CX + 2.5, dy, color=RED, lw=1.3)
    ax.text(CX + 1.72, dy + 0.1, "No", fontsize=9, color=RED,
            weight="bold", ha="center")
    _box(ax, CX + 3.2, dy, 1.45, 0.65,
         "1st-order\nsensitivity\nfallback → k*", fc=LRED, ec=RED, fs=9)

    # Both branches merge down
    _arr(ax, CX - 3.2, dy - 0.28, CX - 0.05, 2.15, color=MID, lw=1.2)
    _arr(ax, CX + 3.2, dy - 0.33, CX + 0.05, 2.15, color=MID, lw=1.2)

    # ── Apply attack ──────────────────────────────────────────────────────────
    _box(ax, CX, 1.82, bw, 0.55,
         "Apply: set pixel k* to chosen value  →  q*",
         fc=LGRAY, ec=GRAY, fs=9.5)

    # ── Final decision ────────────────────────────────────────────────────────
    dy2 = 1.17
    diamond2 = plt.Polygon(
        [[CX, dy2 + 0.34], [CX + 0.9, dy2], [CX, dy2 - 0.34], [CX - 0.9, dy2]],
        closed=True, linewidth=1.5, edgecolor=GRAY, facecolor=LGRAY, zorder=3,
    )
    ax.add_patch(diamond2)
    ax.text(CX, dy2, "retrieved(q*) ≠ xₙ ?",
            ha="center", va="center", fontsize=9, color=GRAY, zorder=4)
    _arr(ax, CX, 1.82 - 0.28, CX, dy2 + 0.34)

    # Success / Fail
    _arr(ax, CX - 0.9, dy2, CX - 2.3, dy2, color=GREEN)
    ax.text(CX - 1.6, dy2 + 0.1, "Yes", fontsize=9, color=GREEN, weight="bold")
    _box(ax, CX - 3.0, dy2, 1.3, 0.52,
         "ATTACK\nSUCCESS", fc=LGREEN, ec=GREEN, fs=10, bold=True, color=GREEN)

    _arr(ax, CX + 0.9, dy2, CX + 2.3, dy2, color=RED)
    ax.text(CX + 1.6, dy2 + 0.1, "No", fontsize=9, color=RED, weight="bold")
    _box(ax, CX + 3.0, dy2, 1.3, 0.52,
         "ATTACK\nFAILS", fc=LRED, ec=RED, fs=10, bold=True, color=RED)

    # ── Candidate count annotation ────────────────────────────────────────────
    ax.text(8.5, 5.55,
            "3,920\ncandidates\nper probe",
            ha="center", va="center", fontsize=9, color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.3", fc=LPURPLE, ec=PURPLE, lw=1.2))

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = FIGURES / "methodology_fig2_onepixel_attack.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Two-Stage Experimental Protocol
# ═══════════════════════════════════════════════════════════════════════════

def fig_twostage_protocol():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    _title(ax, "Figure M3: Two-Stage Experimental Protocol — Conditional Success Definition")

    # ── Top: probe pool ───────────────────────────────────────────────────────
    _box(ax, 5, 5.85, 4.5, 0.65, "N probes  (100 per seed × 5 seeds  =  500 total)",
         fc=LBLUE, ec=BLUE, fs=11, bold=True)

    _arr(ax, 5, 5.52, 5, 5.0)

    _box(ax, 5, 4.68, 3.8, 0.55,
         "Stage 1 — Baseline retrieval  (no attack)",
         fc=LGRAY, ec=GRAY, fs=10)

    _arr(ax, 5, 4.40, 5, 3.9)

    # ── Split ─────────────────────────────────────────────────────────────────
    dy_split = 3.66
    diamond = plt.Polygon(
        [[5, dy_split + 0.36], [5 + 1.1, dy_split],
         [5, dy_split - 0.36], [5 - 1.1, dy_split]],
        closed=True, lw=1.5, edgecolor=GRAY, facecolor=LGRAY, zorder=3,
    )
    ax.add_patch(diamond)
    ax.text(5, dy_split, "Baseline correct?",
            ha="center", va="center", fontsize=9.5, color=GRAY, zorder=4)

    # Yes → right branch (attack branch)
    _arr(ax, 6.1, dy_split, 7.5, dy_split, color=GREEN, lw=1.5)
    ax.text(6.82, dy_split + 0.13, "Yes  ✓", fontsize=9.5,
            color=GREEN, weight="bold", ha="center")

    # No → left branch (excluded)
    _arr(ax, 3.9, dy_split, 2.3, dy_split, color=RED, lw=1.5)
    ax.text(3.1, dy_split + 0.13, "No  ✗", fontsize=9.5,
            color=RED, weight="bold", ha="center")

    # Excluded box
    _box(ax, 1.45, dy_split, 2.4, 0.8,
         "Excluded\nfrom conditional\nmetric",
         fc=LRED, ec=RED, fs=9, color=RED)
    ax.text(1.45, dy_split - 0.72,
            "MNIST: 59/500\nFMNIST: 408/500",
            ha="center", va="top", fontsize=8.5, color=RED, style="italic")

    # ── Attack branch ─────────────────────────────────────────────────────────
    _box(ax, 7.8, 3.0, 3.2, 0.58,
         "Stage 2 — Apply one-pixel attack\nretrieve from perturbed query q*",
         fc=LGRAY, ec=GRAY, fs=9.5)
    _arr(ax, 7.8, 3.66 - 0.36, 7.8, 3.29)

    # Second decision
    dy2 = 2.45
    diamond2 = plt.Polygon(
        [[7.8, dy2 + 0.34], [7.8 + 1.05, dy2],
         [7.8, dy2 - 0.34], [7.8 - 1.05, dy2]],
        closed=True, lw=1.5, edgecolor=GRAY, facecolor=LGRAY, zorder=3,
    )
    ax.add_patch(diamond2)
    ax.text(7.8, dy2, "Retrieval changed?",
            ha="center", va="center", fontsize=9.5, color=GRAY, zorder=4)
    _arr(ax, 7.8, 3.0 - 0.29, 7.8, dy2 + 0.34)

    # Conditional SUCCESS
    _arr(ax, 7.8 - 1.05, dy2, 6.0, dy2, color=GREEN, lw=1.5)
    ax.text(6.92, dy2 + 0.13, "Yes", fontsize=9.5,
            color=GREEN, weight="bold", ha="center")
    _box(ax, 5.1, dy2, 1.65, 0.7,
         "CONDITIONAL\nSUCCESS", fc=LGREEN, ec=GREEN, fs=10.5, bold=True,
         color=GREEN)
    ax.text(5.1, dy2 - 0.6,
            "MNIST: 16/441 = 3.63%\nFMNIST: 8/92  = 8.70%",
            ha="center", va="top", fontsize=8.5, color=GREEN, style="italic")

    # Attack fails
    _arr(ax, 7.8 + 1.05, dy2, 9.35, dy2, color=MID, lw=1.3)
    ax.text(8.62, dy2 + 0.13, "No", fontsize=9.5,
            color=MID, ha="center")
    _box(ax, 9.6, dy2, 0.72, 0.55,
         "Attack\nfails", fc=LGRAY, ec=MID, fs=9, color=MID)

    # ── Summary formula at bottom ─────────────────────────────────────────────
    ax.text(5, 0.95,
            "Conditional success rate  =  (# correct baseline probes where attack flips retrieval)\n"
            "                            ──────────────────────────────────────────────────────\n"
            "                                       (# probes with correct baseline retrieval)",
            ha="center", va="center", fontsize=9.5, color=DBLUE,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc=LBLUE, ec=BLUE, lw=1.5))

    # ── Per-seed annotation (right side) ──────────────────────────────────────
    ax.text(0.5, 5.6,
            "Seeds: 42–46\n5 independent\nexperiments",
            ha="center", va="center", fontsize=9, color=BLUE, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc=LBLUE, ec=BLUE, lw=1.2, alpha=0.6))

    fig.tight_layout(rect=[0, 0.0, 1, 1])
    out = FIGURES / "methodology_fig3_twostage_protocol.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Centering Preprocessing Pipeline  (redesigned)
# ═══════════════════════════════════════════════════════════════════════════

def fig_centering_pipeline():
    W, H = 14.0, 6.4
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    _title(ax, "Figure M4: Global-Mean Centering Preprocessing (Grayscale CIFAR)")

    rng  = np.random.default_rng(7)
    cell = 0.036                          # pixel cell size
    pw   = 8 * cell                       # pattern width = 0.288
    gap  = 0.09                           # gap between grids in a block
    blk  = 4 * pw + 3 * gap              # 4-pattern block width = 1.422
    bw   = 1.38                           # processing box width
    bh   = 0.72                           # processing box height

    # ── y levels (everything on exactly two horizontal rails) ────────────────
    Y_TOP = 4.15   # vertical centre of top row  (boxes and pattern midpoints)
    Y_BOT = 2.05   # vertical centre of bottom row
    Y_SEP = 3.08   # dashed separator
    # pattern grid bottom = Y_? - pw/2  (centred on rail)

    # ── TOP ROW: stored-pattern preprocessing ────────────────────────────────
    # 1. Raw pattern grids (left)
    rx0 = 0.22                            # left edge of raw block
    for i in range(4):
        pl = rx0 + i * (pw + gap)
        c  = 0.62 + rng.uniform(-0.10, 0.10, (8, 8))
        _pattern_grid(ax, pl, Y_TOP - pw / 2, cell=cell, rows=8, cols=8, colors=c)
    rx1 = rx0 + blk                       # right edge of raw block  ≈ 1.642

    ax.text(rx0 + blk / 2, Y_TOP - pw / 2 - 0.20,
            "Raw CIFAR  X\ncos ≈ 0.84",
            ha="center", va="top", fontsize=9, color=RED,
            bbox=dict(boxstyle="round,pad=0.16", fc=LRED, ec=RED, lw=1.1))

    # x positions of the three processing boxes (top row)
    bx = [2.58, 4.44, 6.30]   # box centres

    _arr(ax, rx1 + 0.07, Y_TOP, bx[0] - bw / 2, Y_TOP)
    _box(ax, bx[0], Y_TOP, bw, bh,
         "Compute  μ\nX.mean(dim=1)", fc=LORANGE, ec=ORANGE, fs=9.5)

    _arr(ax, bx[0] + bw / 2, Y_TOP, bx[1] - bw / 2, Y_TOP)
    _box(ax, bx[1], Y_TOP, bw, bh,
         "Subtract mean\nXc  =  X − μ", fc=LPURPLE, ec=PURPLE, fs=9.5)

    _arr(ax, bx[1] + bw / 2, Y_TOP, bx[2] - bw / 2, Y_TOP)
    _box(ax, bx[2], Y_TOP, bw, bh,
         "L2-normalise\nX̂  =  Xc / ‖Xc‖", fc=LBLUE, ec=BLUE, fs=9.5)

    # 5. Centered pattern grids (right)
    cx0 = bx[2] + bw / 2 + 0.28          # ≈ 7.27
    _arr(ax, bx[2] + bw / 2, Y_TOP, cx0 - 0.06, Y_TOP)
    for i in range(4):
        p  = rng.uniform(0.1, 0.9, (8, 8))
        p -= p.mean()
        p  = p / (np.abs(p).max() + 1e-6) * 0.42 + 0.5
        _pattern_grid(ax, cx0 + i * (pw + gap), Y_TOP - pw / 2,
                      cell=cell, rows=8, cols=8, colors=p)
    cx1 = cx0 + blk                       # ≈ 8.692

    ax.text(cx0 + blk / 2, Y_TOP - pw / 2 - 0.20,
            "Stored patterns  X̂\ncos ≈ −0.009",
            ha="center", va="top", fontsize=9, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.16", fc=LGREEN, ec=GREEN, lw=1.1))

    # "→ store in HN" note
    _arr(ax, cx1 + 0.08, Y_TOP, cx1 + 0.55, Y_TOP, color=GREEN)
    _box(ax, cx1 + 0.82, Y_TOP, 0.72, 0.62,
         "Store\nin HN", fc=LGREEN, ec=GREEN, fs=9, bold=True, color=GREEN)

    # Banner above top row
    ax.text(W / 2, Y_TOP + pw / 2 + 0.35,
            "hop = ContinuousHopfield( X̂,  beta = 8.0 )   ←  done once at experiment start",
            ha="center", va="bottom", fontsize=9.5, color=DBLUE,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.28", fc=LBLUE, ec=BLUE, lw=1.3))

    # ── SEPARATOR ────────────────────────────────────────────────────────────
    ax.axhline(y=Y_SEP, xmin=0.01, xmax=0.99,
               color=MID, lw=1.0, ls="--", alpha=0.55)
    ax.text(W / 2, Y_SEP - 0.10,
            "query-time centering  (applied to every probe)",
            ha="center", va="top", fontsize=9, color=MID, style="italic")

    # ── BOTTOM ROW: query pipeline ────────────────────────────────────────────
    qx0 = 0.22                            # left edge of query image
    q_raw = 0.62 + rng.uniform(-0.10, 0.10, (8, 8))
    _pattern_grid(ax, qx0, Y_BOT - pw / 2, cell=cell, rows=8, cols=8, colors=q_raw)
    qx1 = qx0 + pw                        # right edge of query image  ≈ 0.508

    ax.text(qx0 + pw / 2, Y_BOT - pw / 2 - 0.20,
            "Raw query  q\n(attacked pixel)",
            ha="center", va="top", fontsize=9, color=GRAY, style="italic")

    # x positions of bottom processing boxes (same bw, bh)
    bqx = [1.88, 3.64, 5.40]

    _arr(ax, qx1 + 0.07, Y_BOT, bqx[0] - bw / 2, Y_BOT)
    _box(ax, bqx[0], Y_BOT, bw, bh,
         "Subtract  μ\nq − μ", fc=LORANGE, ec=ORANGE, fs=9.5)

    _arr(ax, bqx[0] + bw / 2, Y_BOT, bqx[1] - bw / 2, Y_BOT)
    _box(ax, bqx[1], Y_BOT, bw, bh,
         "Unit-normalise\n(q−μ) / ‖q−μ‖", fc=LPURPLE, ec=PURPLE, fs=9.5)

    _arr(ax, bqx[1] + bw / 2, Y_BOT, bqx[2] - bw / 2, Y_BOT)
    _box(ax, bqx[2], Y_BOT, bw, bh,
         "Retrieve from  X̂\nnet.retrieve( q̂ )", fc=LBLUE, ec=BLUE, fs=9.5)

    # Result image
    rx0b = bqx[2] + bw / 2 + 0.28
    _arr(ax, bqx[2] + bw / 2, Y_BOT, rx0b - 0.06, Y_BOT, color=GREEN)
    q_cen = q_raw - q_raw.mean()
    q_cen = q_cen / (np.abs(q_cen).max() + 1e-6) * 0.42 + 0.5
    _pattern_grid(ax, rx0b, Y_BOT - pw / 2, cell=cell, rows=8, cols=8, colors=q_cen)
    ax.text(rx0b + pw / 2, Y_BOT - pw / 2 - 0.20,
            "Retrieved pattern",
            ha="center", va="top", fontsize=9, color=GREEN, style="italic")

    # ── μ REUSE: clean right-angle L-path ────────────────────────────────────
    # From bottom-centre of μ-box (top row) → down to separator midpoint →
    # left to above q-μ box → down to top of q-μ box (bottom row)
    mu_top_x  = bx[0]                     # 2.58
    mu_top_y1 = Y_TOP - bh / 2            # box bottom  ≈ 3.79
    qmu_x     = bqx[0]                    # 1.88
    qmu_y2    = Y_BOT + bh / 2            # box top     ≈ 2.41
    mid_y     = Y_SEP + 0.18              # just above separator  ≈ 3.26

    ax.plot([mu_top_x, mu_top_x, qmu_x],
            [mu_top_y1, mid_y, mid_y],
            color=ORANGE, lw=1.4, ls="--", zorder=2, clip_on=False)
    ax.annotate("", xy=(qmu_x, qmu_y2), xytext=(qmu_x, mid_y),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                               mutation_scale=10))
    ax.text((mu_top_x + qmu_x) / 2, mid_y + 0.10,
            "μ  (stored patterns mean)",
            ha="center", va="bottom", fontsize=8.5, color=ORANGE, style="italic")

    # ── Code block  (right side, vertically centred between the two rows) ────
    code = ("def center_and_normalise(X):  # X: (d, N)\n"
            "    mu  = X.mean(dim=1, keepdim=True)\n"
            "    Xc  = X - mu\n"
            "    nrm = Xc.norm(dim=0).clamp(min=1e-8)\n"
            "    return Xc / nrm,  mu\n"
            "\n"
            "def proc_query(q, mu):\n"
            "    qc = q - mu.squeeze()\n"
            "    return qc / qc.norm().clamp(min=1e-8)")
    ax.text(10.3, (Y_TOP + Y_BOT) / 2,
            code,
            ha="left", va="center", fontsize=8.2, color="#1a1a2e",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.38", fc="#F0F0FF", ec=BLUE, lw=1.3))

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = FIGURES / "methodology_fig4_centering_pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Cross-Dataset Vulnerability Map
# ═══════════════════════════════════════════════════════════════════════════

def fig_cross_dataset_map():
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5),
                              gridspec_kw={"width_ratios": [1.4, 1, 1]})
    fig.suptitle("Figure M5: Cross-Dataset Vulnerability Overview",
                 fontsize=12, fontweight="bold", color=GRAY, y=1.01)

    # ── Data ──────────────────────────────────────────────────────────────────
    datasets = [
        "MNIST\nN=100",
        "F-MNIST\nN=100",
        "CIFAR\nraw N=10",
        "CIFAR\ncentered\nN=100",
        "CIFAR\ncentered\nN=500",
    ]
    cos_vals  = [0.396, 0.595, 0.649, -0.009, -0.001]
    bl_fail   = [11.8,  81.6,  80.0,   0.0,   20.4]
    cond_atk  = [3.63,  8.70,  0.0,    0.0,    2.01]
    colors_ds = [BLUE, ORANGE, RED, GREEN, PURPLE]

    n = len(datasets)
    y_pos = np.arange(n)[::-1]  # top to bottom

    # ── Panel A: Pairwise cosine (crowding) ───────────────────────────────────
    ax = axes[0]
    ax.set_title("Pattern Crowding\n(mean pairwise cosine)", fontsize=10,
                 color=GRAY, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(MID)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, bottom=True)

    bars = ax.barh(y_pos, cos_vals, height=0.55, color=colors_ds, alpha=0.85,
                   edgecolor="white", linewidth=1.5)
    ax.axvline(x=0, color=MID, lw=1.0, ls="--")
    for i, (v, b) in enumerate(zip(cos_vals, bars)):
        ax.text(v + (0.02 if v >= 0 else -0.02), y_pos[i],
                f"{v:+.3f}", ha="left" if v >= 0 else "right",
                va="center", fontsize=9, color=colors_ds[i], weight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=9.5)
    ax.set_xlabel("Mean pairwise cosine", fontsize=9, color=MID)
    ax.set_xlim(-0.25, 0.95)
    ax.set_ylim(-0.5, n - 0.5)

    # Danger zone annotation
    ax.axvspan(0.5, 0.95, alpha=0.07, color=RED, zorder=0)
    ax.text(0.72, n - 1.1, "high\ncrowding", ha="center", va="top",
            fontsize=8, color=RED, style="italic")

    # ── Panel B: Baseline failure rate ────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Baseline Failure Rate\n(no attack)", fontsize=10,
                 color=GRAY, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(left=False, bottom=True, labelleft=False)

    for i, (v, c) in enumerate(zip(bl_fail, colors_ds)):
        ax.barh(y_pos[i], v, height=0.55, color=c, alpha=0.85,
                edgecolor="white", lw=1.5)
        ax.text(v + 0.5, y_pos[i], f"{v:.1f}%",
                ha="left", va="center", fontsize=9, color=c, weight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Baseline failure rate (%)", fontsize=9, color=MID)
    ax.axvline(x=0, color=MID, lw=0.5)

    # ── Panel C: Conditional attack success ───────────────────────────────────
    ax = axes[2]
    ax.set_title("Conditional Attack Success\n(given correct baseline)", fontsize=10,
                 color=GRAY, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MID)
    ax.tick_params(left=False, bottom=True, labelleft=False)

    for i, (v, c) in enumerate(zip(cond_atk, colors_ds)):
        ax.barh(y_pos[i], v, height=0.55, color=c, alpha=0.85,
                edgecolor="white", lw=1.5)
        lbl = f"{v:.2f}%" if v > 0 else "0.00%"
        ax.text(v + 0.1, y_pos[i], lbl,
                ha="left", va="center", fontsize=9, color=c, weight="bold")
        if v == 0 and bl_fail[i] > 50:
            ax.text(1.5, y_pos[i], "← no correct\nbaseline to attack",
                    ha="left", va="center", fontsize=7.5, color=MID,
                    style="italic")

    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("Conditional attack success (%)", fontsize=9, color=MID)
    ax.axvline(x=0, color=MID, lw=0.5)

    # ── Footnote / insight ────────────────────────────────────────────────────
    fig.text(0.5, -0.04,
             "Key insight: Centering drives pairwise cosine to ≈0, eliminating both baseline failure and attack vulnerability.\n"
             "Without centering, high pattern crowding (cos > 0.5) causes retrieval collapse — most probes fail even before attack.",
             ha="center", va="top", fontsize=9, color=MID, style="italic")

    fig.tight_layout()
    out = FIGURES / "methodology_fig5_cross_dataset_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Mermaid source files  (M2 and M3 — pure flowcharts)
# Open .mmd files in VS Code with "Markdown Preview Mermaid Support" extension.
# Export to SVG/PNG with:  mmdc -i file.mmd -o file.svg
# ═══════════════════════════════════════════════════════════════════════════

MMD_M2 = """\
---
title: "Figure M2: White-Box One-Pixel Attack Protocol"
---
flowchart LR
    A(["Query q\\n(d-dim)"])
    B["Enumerate candidates\\nd × 5 values\\n3,920 / 5,120 total"]
    C["Batch retrieve\\nnet.retrieve(candidates)"]
    D["Cosine to xn\\nfor each candidate"]
    E{{"Delta_cos < -1e-4?"}}
    F["argmin Delta_cos\\n-> pixel k*"]
    G["1st-order\\nsensitivity\\n-> pixel k*"]
    H["Set pixel k*\\n-> q*"]
    I{{"retrieved(q*)\\n!= xn?"}}
    J(["ATTACK\\nSUCCESS"])
    K(["ATTACK\\nFAILS"])

    A --> B --> C --> D --> E
    E -- Yes --> F --> H
    E -- No  --> G --> H
    H --> I
    I -- Yes --> J
    I -- No  --> K

    style A fill:#D6EAF8,stroke:#1A5276,color:#2C3E50,font-weight:bold
    style B fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style C fill:#D6EAF8,stroke:#1A5276,color:#2C3E50
    style D fill:#E8DAEF,stroke:#6C3483,color:#2C3E50
    style E fill:#FAE5D3,stroke:#A04000,color:#A04000,font-weight:bold
    style F fill:#D5F5E3,stroke:#1E6B3C,color:#1E6B3C
    style G fill:#FADBD8,stroke:#B03A2E,color:#B03A2E
    style H fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style I fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style J fill:#D5F5E3,stroke:#1E6B3C,color:#1E6B3C,font-weight:bold
    style K fill:#FADBD8,stroke:#B03A2E,color:#B03A2E,font-weight:bold
"""

MMD_M3 = """\
---
title: "Figure M3: Two-Stage Experimental Protocol -- Conditional Success"
---
flowchart LR
    A(["N probes\\n5 seeds"])
    B["Stage 1\\nBaseline retrieval\\n(no attack)"]
    C{{"Baseline\\ncorrect?"}}
    D["Excluded\\nMNIST: 59/500\\nFMNIST: 408/500"]
    E["Stage 2\\nApply one-pixel\\nattack -> q*"]
    F{{"Retrieval\\nchanged?"}}
    G(["CONDITIONAL SUCCESS\\nMNIST:  3.63%  (16/441)\\nFMNIST: 8.70%  (8/92)\\nCIFAR N=500: 2.01%  (4/199)"])
    H(["Attack\\nfails"])

    A --> B --> C
    C -- "No"  --> D
    C -- "Yes" --> E --> F
    F -- "Yes" --> G
    F -- "No"  --> H

    style A  fill:#D6EAF8,stroke:#1A5276,color:#2C3E50,font-weight:bold
    style B  fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style C  fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style D  fill:#FADBD8,stroke:#B03A2E,color:#B03A2E
    style E  fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style F  fill:#F2F3F4,stroke:#5D6D7E,color:#2C3E50
    style G  fill:#D5F5E3,stroke:#1E6B3C,color:#1E6B3C,font-weight:bold
    style H  fill:#FADBD8,stroke:#B03A2E,color:#B03A2E
"""


def write_mermaid_files():
    mmd2 = FIGURES.parent / "figures" / "methodology_fig2_onepixel_attack.mmd"
    mmd3 = FIGURES.parent / "figures" / "methodology_fig3_twostage_protocol.mmd"
    mmd2.write_text(MMD_M2, encoding="utf-8")
    mmd3.write_text(MMD_M3, encoding="utf-8")
    print(f"  saved: {mmd2.name}")
    print(f"  saved: {mmd3.name}")
    print()
    print("  To render Mermaid files:")
    print("    VS Code: install 'Markdown Preview Mermaid Support', open .mmd file,")
    print("             press Ctrl+Shift+V to preview.")
    print("    CLI:     npm install -g @mermaid-js/mermaid-cli")
    print("             mmdc -i methodology_fig2_onepixel_attack.mmd -o fig2.svg")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Generating methodology figures (matplotlib)...")
    fig_chnn_retrieval()
    fig_onepixel_attack()
    fig_twostage_protocol()
    fig_centering_pipeline()
    fig_cross_dataset_map()
    print()
    print("Generating Mermaid source files...")
    write_mermaid_files()
    print(f"\nAll outputs saved to:  {FIGURES}/")


if __name__ == "__main__":
    main()
