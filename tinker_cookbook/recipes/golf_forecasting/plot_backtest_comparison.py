"""Generate a comparison chart of model predictions vs Kalshi/sportsbook odds for the 2026 Masters."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def plot_backtest():
    rounds = ["After R1\n(co-leading -5)", "After R2\n(6-shot lead -12)", "After R3\n(tied -11)"]
    x = np.arange(len(rounds))
    width = 0.25

    model_probs = [12.1, 33.0, 55.0]  # Binary for R3, Kimi for R1-R2
    kalshi_probs = [25.0, 65.0, 36.0]  # Kalshi/market estimates
    sportsbook_probs = [25.0, 73.7, 42.6]  # FanDuel implied

    fig, ax = plt.subplots(figsize=(12, 6.5))

    bars1 = ax.bar(x - width, model_probs, width, label="Our Model", color="#2a9d8f", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x, kalshi_probs, width, label="Kalshi", color="#e76f51", edgecolor="white", linewidth=0.8)
    bars3 = ax.bar(x + width, sportsbook_probs, width, label="Sportsbook (implied)", color="#e9c46a", edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Add annotation for the big disagreement
    ax.annotate(
        "41 pp gap!\nModel far more\nskeptical of lead",
        xy=(1 - width, 33),
        xytext=(1.7, 60),
        fontsize=9,
        ha="center",
        color="#264653",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#264653", lw=1.5),
    )

    # Add "what happened next" annotation
    ax.annotate(
        "McIlroy shot 73 in R3\nLead collapsed from 6 to 0",
        xy=(1.5, 82),
        fontsize=8.5,
        ha="center",
        va="bottom",
        color="#d44",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5", edgecolor="#d44", alpha=0.9),
    )

    ax.set_ylabel("McIlroy Win Probability (%)", fontsize=12)
    ax.set_title(
        "2026 Masters Backtest: Model vs. Prediction Markets\n"
        "McIlroy win probability at each round (he won at -12)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rounds, fontsize=11)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # Add outcome banner at bottom
    ax.text(
        0.5,
        -0.15,
        "Outcome: McIlroy won (-12) by 1 shot over Scheffler. Back-to-back Masters champion.",
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        style="italic",
        color="#264653",
    )

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "masters_2026_backtest.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_backtest()
