"""Single Slack-scannable chart: SFT runs vs prompt-only journey on DeepMath 500."""

import matplotlib.pyplot as plt
from pathlib import Path

# label, accuracy, kind ("sft" | "prompt")
runs = [
    ("v3 baseline (2302 records SFT)",   0.708, "sft"),
    ("tiny SFT (100 records)",           0.482, "sft"),
    ("tiny SFT (20 records)",            0.870, "sft"),
    ("0020  prompt-only (drop SFT)",     0.774, "prompt"),
    ("0062  + max_tokens 24576",         0.863, "prompt"),
    ("0076  + CoT prefix",               0.868, "prompt"),
    ("0080  + cadence anchor",           0.876, "prompt"),
    ("0105  final (26-sample mean)",     0.8750, "prompt"),
]
NO_TOOL = 0.880

labels = [r[0] for r in runs]
accs   = [r[1] for r in runs]
colors = ["#888" if r[2] == "sft" else "#c2592a" for r in runs]

fig, ax = plt.subplots(figsize=(10, 6))
y = list(range(len(runs)))
ax.barh(y, accs, color=colors, edgecolor="white")
ax.invert_yaxis()
ax.set_yticks(y, labels)
ax.set_xlim(0.40, 0.92)
ax.set_xlabel("DeepMath 500 accuracy")

# baseline
ax.axvline(NO_TOOL, color="#3f6973", linestyle="--", linewidth=1.2)
ax.text(NO_TOOL, -0.7, f"no-tool baseline {NO_TOOL}",
        color="#3f6973", ha="center", va="bottom", fontsize=9)

# value labels at the end of each bar
for yi, a in zip(y, accs):
    ax.text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=9)

# legend (top-left, above the bars where there's empty space)
from matplotlib.patches import Patch
ax.legend(
    handles=[
        Patch(facecolor="#c2592a", label="prompt-only · kept"),
        Patch(facecolor="#888",    label="SFT · regressed"),
    ],
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    frameon=False,
    fontsize=10,
)

fig.suptitle("Dropping SFT was the move.", x=0.02, ha="left", fontsize=18, y=0.99)
fig.text(0.02, 0.945,
         "4 SFT runs · 4 regressions.  The recipe we kept uses 0 training records.",
         fontsize=10, color="#555")

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eee", linewidth=1)
ax.set_axisbelow(True)

fig.tight_layout()
out = Path(__file__).parent / "AUTORESEARCH_INFOGRAPHIC.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
