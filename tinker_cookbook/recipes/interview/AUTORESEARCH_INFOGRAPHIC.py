"""Simple line chart: eval accuracy across iterations, marking the SFT → prompt-only pivot."""

import matplotlib.pyplot as plt
from pathlib import Path

# (label, accuracy)  — in iteration order, left to right
runs = [
    ("v3 baseline\n(2302 records SFT)", 0.708),
    ("tiny SFT\n(100 records)",         0.482),
    ("tiny SFT\n(20 records)",          0.870),
    ("0020\nprompt-only",               0.774),
    ("0062\n+ max_tokens 24576",        0.863),
    ("0076\n+ CoT prefix",              0.868),
    ("0080\n+ cadence\nanchor",         0.876),
    ("0105\nfinal\n(26-sample mean)",   0.8750),
]
NO_TOOL = 0.880
PIVOT = 2.5  # between idx 2 (last SFT) and idx 3 (first prompt-only)

x = list(range(len(runs)))
y = [r[1] for r in runs]
labels = [r[0] for r in runs]

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(x, y, "-", color="#888", linewidth=1.2, zorder=1)
ax.scatter(x[:3], y[:3], color="#888",    s=60, zorder=2, label="SFT")
ax.scatter(x[3:], y[3:], color="#c2592a", s=60, zorder=2, label="prompt-only")

# pivot marker: vertical line + annotation
ax.axvline(PIVOT, color="#c2592a", linestyle="--", linewidth=1.2, alpha=0.6)
ax.annotate(
    "threw SFT away →\npure prompting",
    xy=(PIVOT, 0.50), xytext=(PIVOT + 0.15, 0.50),
    color="#c2592a", fontsize=11, va="center", ha="left",
    fontweight="bold",
)

# no-tool baseline
ax.axhline(NO_TOOL, color="#3f6973", linestyle=":", linewidth=1.2)
ax.text(-0.4, NO_TOOL + 0.006, f"no-tool baseline {NO_TOOL}",
        color="#3f6973", ha="left", va="bottom", fontsize=9)

# value labels above each point
for xi, yi in zip(x, y):
    ax.text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

# annotate tiny-SFT(20): high score, but only because the model collapsed
# to skipping the tool ~99% of the time — passed eval, failed the task.
ax.annotate(
    "⚠ misleading: model collapsed to\n~99% tool-skip rate — passed eval\nby abandoning the tool entirely",
    xy=(2, 0.870), xytext=(1.55, 0.58),
    fontsize=10, color="#a83232", ha="left", va="center", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#a83232", lw=1.2,
                    connectionstyle="arc3,rad=-0.25"),
)

ax.set_xticks(x, labels, fontsize=8)
ax.set_ylim(0.42, 0.95)
ax.set_xlim(-0.5, len(runs) - 0.5)
ax.set_ylabel("DeepMath 500 accuracy")
ax.set_title("Eval accuracy across iterations — dropping SFT was the unlock",
             loc="left", fontsize=14, pad=14)

ax.legend(loc="lower right", frameon=False, fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=1)
ax.set_axisbelow(True)

fig.tight_layout()
out = Path(__file__).parent / "AUTORESEARCH_INFOGRAPHIC.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
