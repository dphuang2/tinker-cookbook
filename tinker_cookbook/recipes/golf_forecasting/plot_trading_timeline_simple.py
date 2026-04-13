"""Simplified trading timeline: model vs market with the gap shaded as the edge."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def plot_simple_timeline():
    fig, ax = plt.subplots(figsize=(13, 7))

    x = [0, 1, 2, 3]
    xlabels = ["Pre-tournament", "After R1\n(Thursday)", "After R2\n(Friday)", "After R3\n(Saturday)"]

    # Use Kalshi prices and exp79/80 fine-tuned 1B predictions
    market = [18, 25, 65, 36]
    model = [None, 18, 85, 35]

    # Plot lines
    ax.plot(x, market, "o-", color="#e76f51", linewidth=2.5, markersize=10, label="Prediction market (Kalshi)", zorder=4)
    ax.plot([1, 2, 3], model[1:], "o-", color="#2a9d8f", linewidth=2.5, markersize=10, label="Our model", zorder=5)

    # Shade the gap
    # R1 to R2: model below market, then above
    # Lines cross between R1 and R2: model goes 18->85, market goes 25->65
    t_cross_12 = (25 - 18) / ((85 - 18) - (65 - 25))
    x_cross_12 = 1 + t_cross_12
    # Before crossing: model < market
    xs_pre1 = np.linspace(1, min(x_cross_12, 2), 20)
    model_pre1 = np.interp(xs_pre1, [1, 2], [18, 85])
    market_pre1 = np.interp(xs_pre1, [1, 2], [25, 65])
    ax.fill_between(xs_pre1, model_pre1, market_pre1, color="#e76f51", alpha=0.12, zorder=2)
    # After crossing: model > market (buy YES zone)
    if x_cross_12 < 2:
        xs_post1 = np.linspace(x_cross_12, 2, 20)
        model_post1 = np.interp(xs_post1, [1, 2], [18, 85])
        market_post1 = np.interp(xs_post1, [1, 2], [25, 65])
        ax.fill_between(xs_post1, model_post1, market_post1, color="#2a9d8f", alpha=0.12, zorder=2)
    # R2 to R3: model goes 85->35, market goes 65->36. Model starts above, crosses down.
    t_cross_23 = (85 - 65) / ((85 - 35) - (65 - 36))
    x_cross_23 = 2 + t_cross_23
    xs_pre2 = np.linspace(2, min(x_cross_23, 3), 20)
    model_pre2 = np.interp(xs_pre2, [2, 3], [85, 35])
    market_pre2 = np.interp(xs_pre2, [2, 3], [65, 36])
    ax.fill_between(xs_pre2, model_pre2, market_pre2, color="#2a9d8f", alpha=0.12, zorder=2)
    if x_cross_23 < 3:
        xs_post2 = np.linspace(x_cross_23, 3, 20)
        model_post2 = np.interp(xs_post2, [2, 3], [85, 35])
        market_post2 = np.interp(xs_post2, [2, 3], [65, 36])
        ax.fill_between(xs_post2, model_post2, market_post2, color="#e76f51", alpha=0.12, zorder=2)

    # Gap labels
    ax.text(1.65, 75, "Model\nmore\nbullish", fontsize=11, fontweight="bold", color="#1a7a6d", ha="center")

    # Value labels on points
    for xi, mi, ki in [(1, 18, 25), (2, 85, 65), (3, 35, 36)]:
        m_offset = -4.5 if mi < ki else 4
        k_offset = 4.5 if ki > mi else -5.5
        ax.text(xi - 0.08, mi + m_offset, f"{mi}%", ha="center", fontsize=11, color="#2a9d8f", fontweight="bold")
        ax.text(xi + 0.08, ki + k_offset, f"{ki}%", ha="center", fontsize=11, color="#e76f51", fontweight="bold")
    ax.text(0, market[0] + 4, f"{market[0]}%", ha="center", fontsize=11, color="#e76f51", fontweight="bold")

    # Event annotations -- one per key moment, placed cleanly
    ax.annotate("6-shot lead", xy=(2, 65), xytext=(0.8, 78),
                fontsize=10, color="#777", ha="center",
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1))
    ax.annotate("Shoots 73,\nlead collapses", xy=(2.5, 50), xytext=(2.5, 85),
                fontsize=10, color="#777", ha="center", style="italic",
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1))
    ax.annotate("Wins by 1", xy=(3, 55), xytext=(3.35, 72),
                fontsize=10, color="#777", ha="center", style="italic",
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="#2a9d8f", linewidth=2.5, marker="o", markersize=8, label="Our model"),
        plt.Line2D([0], [0], color="#e76f51", linewidth=2.5, marker="o", markersize=8, label="Kalshi"),
        mpatches.Patch(facecolor="#e76f51", alpha=0.15, edgecolor="none", label="Model says overpriced"),
        mpatches.Patch(facecolor="#2a9d8f", alpha=0.15, edgecolor="none", label="Model says underpriced"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left", framealpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel("McIlroy Win Probability (%)", fontsize=13)
    ax.set_ylim(0, 95)
    ax.set_xlim(-0.3, 3.6)
    ax.grid(True, alpha=0.2)

    ax.set_title("2026 Masters: Model vs. Kalshi", fontsize=15, fontweight="bold")

    # Outcome line at bottom -- short
    ax.text(0.5, -0.10, "McIlroy won (-12), 1 shot over Scheffler.",
            transform=ax.transAxes, fontsize=11, ha="center", color="#555", style="italic")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "masters_2026_trading_timeline.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_simple_timeline()
