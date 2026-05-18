"""
Generate a single Slack-scannable chart summarizing the autoresearch journey:
   4 SFT runs (regressed)  vs.  prompt-only journey (kept) → 0.8750.

Usage:
   uv run --no-sync --with matplotlib python AUTORESEARCH_INFOGRAPHIC.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from pathlib import Path

# ── palette ────────────────────────────────────────────────────────────────────
PAPER         = "#f7f1e6"
INK           = "#252835"
INK_SOFT      = "#5a5d6b"
INK_FAINT     = "#8c8f99"
RULE          = "#bfb9ac"
RULE_SOFT     = "#dcd6c8"
KEPT          = "#c2592a"
KEPT_SOFT     = "#f1d8c8"
DISCARD       = "#6f7691"
DISCARD_DEEP  = "#4f566e"
BASELINE      = "#3f6973"

rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor":   PAPER,
    "axes.edgecolor":   INK,
    "axes.linewidth":   1.2,
    "axes.labelcolor":  INK,
    "xtick.color":      INK_FAINT,
    "ytick.color":      INK_FAINT,
    "text.color":       INK,
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size":        11,
})

# ── data ───────────────────────────────────────────────────────────────────────
# (x, accuracy, title, sub, warn)  — SFT runs (discarded)
sft = [
    (1.0,  0.708, "v3 baseline", "2 302 records",  None),
    (2.4,  0.482, "tiny SFT",    "100 records",    "catastrophic"),
    (3.8,  0.870, "tiny SFT",    "20 records",     "99% skip · cadence dead"),
]

# (x, accuracy, title, sub)  — prompt-only journey
prompt = [
    (6.0,  0.774,  "prompt-only",      "0020 · drop SFT"),
    (7.5,  0.863,  "max_tokens 24576", "0062 · raised eval cap"),
    (9.0,  0.868,  "CoT prefix",       "0076 · “step by step”"),
    (10.5, 0.876,  "cadence anchor",   "0080 · “three calls”"),
    (12.0, 0.8750, "final recipe",     "0105 · 26-sample mean"),
]

BASELINE_ACC = 0.880

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 7.8))

# title + subtitle
fig.text(0.06, 0.945, "Dropping SFT was the move.",
         fontsize=32, family="serif", color=INK, ha="left", va="top")
fig.text(0.06, 0.898,
         "Four supervised runs, four regressions. The recipe we kept uses pure prompt engineering — 0 training records.",
         fontsize=12.5, style="italic", family="serif", color=INK_SOFT, ha="left", va="top")

# top-right meta
fig.text(0.945, 0.945, "autoresearch · progress-update · DeepMath 500",
         fontsize=9, family="monospace", color=INK_FAINT, ha="right", va="top")
fig.text(0.945, 0.92, "159 experiments  ·  26 final-recipe samples",
         fontsize=9, family="monospace", color=INK_FAINT, ha="right", va="top")

# rule under masthead
fig.add_artist(plt.Line2D([0.06, 0.945], [0.86, 0.86], transform=fig.transFigure,
                          color=INK, linewidth=1.5))

# ── axes ───────────────────────────────────────────────────────────────────────
ax = fig.add_axes([0.06, 0.16, 0.885, 0.62])
ax.set_xlim(0.0, 13.0)
ax.set_ylim(0.44, 0.93)

yticks = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontfamily="monospace", fontsize=9.5)
ax.tick_params(axis="y", length=0, pad=6)
ax.set_xticks([])

for y in yticks:
    ax.axhline(y, color=RULE_SOFT, linewidth=1, zorder=0)
for y in [0.45, 0.70, 0.90]:
    ax.axhline(y, color=RULE, linewidth=1, zorder=0)

for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(INK)
ax.spines["left"].set_color(INK)

# divider between SFT (left) and prompt-only (right)
divider_x = 4.9
ax.axvline(divider_x, color=RULE, linewidth=1, linestyle=(0, (2, 4)), zorder=1)

# group label bars + text — above the plot
ax.plot([0.4, 4.4],  [0.925, 0.925], color=DISCARD_DEEP, linewidth=1.5, zorder=3)
ax.plot([5.4, 12.6], [0.925, 0.925], color=KEPT,         linewidth=1.5, zorder=3)
ax.text(2.4, 0.945, "SFT  ·  4 runs  ·  4 regressions",
        fontsize=10.5, family="monospace", color=DISCARD_DEEP,
        fontweight="bold", ha="center", va="bottom")
ax.text(9.0, 0.945, "prompt-only  ·  the journey we kept",
        fontsize=10.5, family="monospace", color=KEPT,
        fontweight="bold", ha="center", va="bottom")

# ── no-tool baseline (label centered on the divider — empty zone there) ──────
ax.axhline(BASELINE_ACC, color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
ax.text(divider_x, BASELINE_ACC + 0.006, "no-tool ceiling · 0.880",
        fontsize=10, family="monospace", color=BASELINE,
        ha="center", va="bottom",
        bbox=dict(boxstyle="square,pad=0.25", facecolor=PAPER,
                  edgecolor="none", linewidth=0))

# y axis name (anchored in the axis margin, not inside the plot)
fig.text(0.055, 0.78, "accuracy",
         fontsize=9, family="monospace", color=INK_FAINT,
         ha="right", va="top")

# ── PROMPT-ONLY journey ────────────────────────────────────────────────────────
px = [p[0] for p in prompt]
py = [p[1] for p in prompt]

ax.fill_between(px, py, 0.45, color=KEPT_SOFT, alpha=0.6, zorder=1)
ax.plot(px, py, color=KEPT, linewidth=2.6, zorder=5,
        solid_capstyle="round", solid_joinstyle="round")

for i, (x, y, title, sub) in enumerate(prompt):
    is_final = i == len(prompt) - 1
    if is_final:
        ax.scatter([x], [y], s=170, facecolor=KEPT, edgecolor=INK,
                   linewidth=1.6, zorder=7)
    else:
        ax.scatter([x], [y], s=72, facecolor=PAPER, edgecolor=KEPT,
                   linewidth=2.2, zorder=7)

    # value label above each prompt-only point (only the first two and the final)
    # the intermediate cluster gets de-cluttered
    if i in (0, len(prompt) - 1):
        val_str = f"{y:.4f}" if is_final else f"{y:.3f}"
        ax.text(x, y + 0.010, val_str,
                fontsize=11.5 if is_final else 10.5,
                family="monospace",
                color=KEPT if is_final else INK,
                fontweight="bold", ha="center", va="bottom")

    # below-axis label
    ax.text(x, 0.428, title, fontsize=10, family="serif",
            color=INK, fontweight="semibold", ha="center", va="top")
    ax.text(x, 0.412, sub, fontsize=8.5, family="monospace",
            color=INK_FAINT, ha="center", va="top")

# Single big callout to the final point — pulled down-right with arrow up-left
ax.annotate(
    "0 training records\n26-sample mean · 0.8750",
    xy=(12.0, 0.8750),
    xytext=(11.4, 0.62),
    fontsize=11, family="monospace", color=KEPT,
    fontweight="bold", ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color=KEPT, lw=1.2,
                    connectionstyle="arc3,rad=0.18",
                    shrinkA=0, shrinkB=10),
    bbox=dict(boxstyle="round,pad=0.5", facecolor=PAPER,
              edgecolor=KEPT, linewidth=1.2),
    zorder=10,
)

# Single big callout to the +9 pp lever (the biggest one) — placed below the line
ax.annotate(
    "+9 pp · biggest lever\n(eval cap was binding)",
    xy=(7.5, 0.863),
    xytext=(7.7, 0.62),
    fontsize=10.5, family="monospace", color=INK,
    fontweight="semibold", ha="center", va="center",
    arrowprops=dict(arrowstyle="->", color=INK_FAINT, lw=1.0,
                    connectionstyle="arc3,rad=-0.18",
                    shrinkA=0, shrinkB=10),
    bbox=dict(boxstyle="round,pad=0.5", facecolor=PAPER,
              edgecolor=RULE, linewidth=1),
    zorder=10,
)

# ── SFT regressions ───────────────────────────────────────────────────────────
for x, y, title, sub, warn in sft:
    color = DISCARD_DEEP if y < 0.70 else DISCARD
    ax.plot([x, x], [0.45, y], color=INK_FAINT, linewidth=0.8,
            linestyle=(0, (1, 3)), zorder=1)
    ax.scatter([x], [y], s=140, marker="X",
               facecolor=color, edgecolor=PAPER, linewidth=1.8, zorder=6)

    # tiny SFT 20rec sits right at the baseline — give its value+warn special placement
    if y > 0.85:
        # value to the RIGHT of marker; warning BELOW
        ax.text(x + 0.18, y, f"{y:.3f}",
                fontsize=10.5, family="monospace", color=color,
                fontweight="bold", ha="left", va="center")
        if warn:
            ax.text(x, y - 0.028, warn,
                    fontsize=9, family="monospace", color=DISCARD_DEEP,
                    fontweight="bold", ha="center", va="top")
    else:
        # value above; warning below
        ax.text(x, y + 0.012, f"{y:.3f}",
                fontsize=10.5, family="monospace", color=color,
                fontweight="bold", ha="center", va="bottom")
        if warn:
            ax.text(x, y - 0.024, warn,
                    fontsize=9, family="monospace", color=DISCARD_DEEP,
                    fontweight="bold", ha="center", va="top")

    # below-axis label
    ax.text(x, 0.428, title, fontsize=10, family="serif",
            color=INK, fontweight="semibold", ha="center", va="top")
    ax.text(x, 0.410, sub, fontsize=8.5, family="monospace",
            color=INK_FAINT, ha="center", va="top")

# ── legend (lower-left, well below the SFT 0.482 marker) ─────────────────────
lx, ly, lw, lh = 0.45, 0.555, 2.85, 0.075
ax.add_patch(patches.FancyBboxPatch(
    (lx, ly), lw, lh,
    boxstyle="square,pad=0",
    facecolor=PAPER, edgecolor=RULE, linewidth=1, zorder=8,
))
# row 1 — prompt-only kept
ax.plot([lx + 0.2, lx + 0.85], [ly + lh - 0.024, ly + lh - 0.024],
        color=KEPT, linewidth=2.5, zorder=9)
ax.scatter([lx + 0.525], [ly + lh - 0.024], s=55,
           facecolor=PAPER, edgecolor=KEPT, linewidth=2, zorder=10)
ax.text(lx + 1.0, ly + lh - 0.024, "prompt-only · kept",
        fontsize=10, family="monospace", color=INK, va="center", zorder=10)
# row 2 — SFT regressed
ax.scatter([lx + 0.525], [ly + 0.018], s=90, marker="X",
           facecolor=DISCARD, edgecolor=PAPER, linewidth=1.6, zorder=10)
ax.text(lx + 1.0, ly + 0.018, "SFT · regressed",
        fontsize=10, family="monospace", color=INK, va="center", zorder=10)

# ── footer ─────────────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.06, 0.945], [0.085, 0.085], transform=fig.transFigure,
                          color=RULE, linewidth=1))
fig.text(0.06, 0.05,
         "Prompt-only beats every fine-tuned variant we tried. "
         "Even 20 records was enough to collapse cadence.",
         fontsize=11.5, style="italic", family="serif", color=INK,
         ha="left", va="center")
fig.text(0.945, 0.05,
         "Qwen3-30B-A3B  ·  0 records  ·  95% CI [0.871, 0.880]",
         fontsize=10, family="monospace", color=INK_FAINT,
         ha="right", va="center")

# ── save ───────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
png = out_dir / "AUTORESEARCH_INFOGRAPHIC.png"
pdf = out_dir / "AUTORESEARCH_INFOGRAPHIC.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight", facecolor=PAPER, pad_inches=0.3)
fig.savefig(pdf,            bbox_inches="tight", facecolor=PAPER, pad_inches=0.3)
print(f"wrote {png}")
print(f"wrote {pdf}")
