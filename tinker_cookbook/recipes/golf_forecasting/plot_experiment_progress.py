"""Generate a Karpathy-style experiment progress chart for the golf forecasting autoresearch."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


RESULTS_TSV = os.path.join(os.path.dirname(__file__), "results.tsv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def load_experiments():
    experiments = []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                ll = float(row["anchor_log_loss"])
            except (ValueError, TypeError):
                ll = None
            experiments.append(
                {
                    "name": row["commit"],
                    "anchor_ll": ll,
                    "status": row["status"],
                    "change_type": row["change_type"],
                    "description": row["description"],
                }
            )
    return experiments


def plot_progress_chart(experiments, output_path):
    """Main Karpathy-style chart: all experiments with running best frontier."""
    fig, ax = plt.subplots(figsize=(16, 7))

    xs, ys, colors, markers = [], [], [], []
    best_xs, best_ys = [], []
    best_so_far = float("inf")

    # Track the actual best experiment for the improvement arrow
    best_exp_x, best_exp_y = None, None

    for i, e in enumerate(experiments):
        if e["anchor_ll"] is None:
            continue
        ll = e["anchor_ll"]
        display_ll = min(ll, 4.5)
        xs.append(i + 1)
        ys.append(display_ll)

        if e["status"] in ("reverted", "not-kept"):
            colors.append("#d44")
            markers.append("x")
        elif e["status"] == "superseded":
            colors.append("#fb8")
            markers.append("o")
        elif e["status"] == "baseline":
            colors.append("#888")
            markers.append("s")
        else:
            colors.append("#2a9d8f")
            markers.append("o")

        if ll < best_so_far:
            best_so_far = ll
            best_xs.append(i + 1)
            best_ys.append(ll)

        if e["name"] == "exp79-binary-margin":
            best_exp_x, best_exp_y = i + 1, ll

    # Plot all experiments as scatter
    for x, y, c, m in zip(xs, ys, colors, markers):
        ax.scatter(x, y, c=c, marker=m if m != "x" else "X", s=40 if m != "x" else 50, zorder=3, edgecolors="none")

    # Plot best frontier as step line
    if best_xs:
        step_xs = []
        step_ys = []
        for j in range(len(best_xs)):
            if j > 0:
                step_xs.append(best_xs[j])
                step_ys.append(best_ys[j - 1])
            step_xs.append(best_xs[j])
            step_ys.append(best_ys[j])
        step_xs.append(xs[-1])
        step_ys.append(best_ys[-1])
        ax.plot(step_xs, step_ys, color="#264653", linewidth=2, alpha=0.8, label="Best so far")

    # Manually place milestone annotations
    milestones = [
        (1, 2.81, "Heuristic baseline", 8, 3.9),
        (9, 1.90, "Bug fix", 16, 3.1),
        (18, 1.27, "Top-3 candidates", 26, 2.4),
        (36, 1.23, "SFT distillation", 44, 1.95),
        (64, 1.23, "1B matches 8B", 60, 1.7),
    ]

    for mx, my, label, tx, ty in milestones:
        ax.annotate(
            label,
            xy=(mx, my),
            xytext=(tx, ty),
            fontsize=9.5,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
        )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2a9d8f", markersize=8, label="Kept"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#d44", markersize=8, label="Reverted / not kept"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fb8", markersize=8, label="Superseded"),
        Line2D([0], [0], color="#264653", linewidth=2, label="Best so far"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel("Log-Loss (lower = better)", fontsize=13)
    ax.set_title(
        "108 Experiments in 49 Hours, Zero Human Intervention",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0, len(experiments) + 2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(True, alpha=0.3)

    # Point the improvement annotation at the actual best experiment
    if best_exp_x and best_exp_y:
        ax.annotate(
            "2.81 \u2192 0.54",
            xy=(best_exp_x, best_exp_y),
            xytext=(78, 1.6),
            fontsize=13,
            fontweight="bold",
            color="#264653",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#264653", lw=1.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_change_type_breakdown(experiments, output_path):
    """Pie chart of what types of changes were tried."""
    from collections import Counter

    types = Counter(e["change_type"] for e in experiments)
    labels = []
    sizes = []
    for k, v in types.most_common():
        labels.append(k)
        sizes.append(v)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_pie = ["#2a9d8f", "#264653", "#e9c46a", "#f4a261", "#e76f51", "#a8dadc", "#457b9d", "#1d3557", "#606c38"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.0f%%", colors=colors_pie[: len(sizes)], startangle=90
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("What the Agent Explored\n(108 experiments by change type)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_kept_vs_reverted(experiments, output_path):
    """Bar chart: kept vs reverted over time (binned)."""
    bin_size = 20
    bins = []
    for start in range(0, len(experiments), bin_size):
        chunk = experiments[start : start + bin_size]
        kept = sum(1 for e in chunk if e["status"] in ("kept", "superseded", "baseline"))
        reverted = sum(1 for e in chunk if e["status"] in ("reverted", "not-kept"))
        bins.append((f"{start + 1}-{start + len(chunk)}", kept, reverted))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(bins))
    labels = [b[0] for b in bins]
    kept_vals = [b[1] for b in bins]
    rev_vals = [b[2] for b in bins]

    ax.bar(x, kept_vals, color="#2a9d8f", label="Kept / superseded")
    ax.bar(x, rev_vals, bottom=kept_vals, color="#d44", label="Reverted / not kept")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Experiment Range", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Experiment Outcomes Over Time", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiments = load_experiments()
    plot_progress_chart(experiments, os.path.join(OUTPUT_DIR, "experiment_progress.png"))
    plot_change_type_breakdown(experiments, os.path.join(OUTPUT_DIR, "change_type_breakdown.png"))
    plot_kept_vs_reverted(experiments, os.path.join(OUTPUT_DIR, "kept_vs_reverted.png"))
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
