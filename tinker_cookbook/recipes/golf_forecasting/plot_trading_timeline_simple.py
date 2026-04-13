"""Simplified trading timeline: model vs market with the gap shaded as the edge."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")


def plot_simple_timeline():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    # X positions and labels
    x = [0, 1, 2, 3]
    xlabels = ["Pre-tournament", "After R1\n(Thursday)", "After R2\n(Friday)", "After R3\n(Saturday)"]

    # Market line (sportsbook implied, most dramatic contrast)
    market = [18, 25, 74, 36]
    # Model line
    model = [None, 12, 34, 55]

    # Plot market line
    ax.plot(x, market, "o-", color="#e76f51", linewidth=2.5, markersize=9, label="Prediction market", zorder=4)
    # Plot model line (starts at R1)
    ax.plot([1, 2, 3], model[1:], "o-", color="#2a9d8f", linewidth=2.5, markersize=9, label="Our model", zorder=5)

    # Shade the gap: RED where model < market (buy NO), GREEN where model > market (buy YES)
    # R1 to R2: model below market
    ax.fill_between([1, 2], [model[1], model[2]], [market[1], market[2]],
                    color="#e76f51", alpha=0.15, zorder=2)
    # R2 to R3: lines cross, need to find intersection
    # model goes 34->55, market goes 74->36. Cross at: 34+t*21 = 74-t*38 => t=40/59 ≈ 0.678
    t_cross = (74 - 34) / (21 + 38)
    x_cross = 2 + t_cross
    y_cross = 34 + t_cross * 21
    # R2 to crossing: model < market (buy NO zone)
    xs_pre = np.linspace(2, x_cross, 20)
    model_interp_pre = np.interp(xs_pre, [2, 3], [34, 55])
    market_interp_pre = np.interp(xs_pre, [2, 3], [74, 36])
    ax.fill_between(xs_pre, model_interp_pre, market_interp_pre, color="#e76f51", alpha=0.15, zorder=2)
    # Crossing to R3: model > market (buy YES zone)
    xs_post = np.linspace(x_cross, 3, 20)
    model_interp_post = np.interp(xs_post, [2, 3], [34, 55])
    market_interp_post = np.interp(xs_post, [2, 3], [74, 36])
    ax.fill_between(xs_post, model_interp_post, market_interp_post, color="#2a9d8f", alpha=0.15, zorder=2)

    # Label the gaps
    ax.annotate(
        "BUY NO\n40pp edge",
        xy=(2, 54), fontsize=11, fontweight="bold", color="#c0392b",
        ha="center", va="center",
    )
    ax.annotate(
        "BUY YES\n19pp edge",
        xy=(2.85, 48), fontsize=11, fontweight="bold", color="#1a7a6d",
        ha="center", va="center",
    )

    # Add what happened callouts
    ax.annotate(
        "McIlroy has historic\n6-shot lead",
        xy=(2, 74), xytext=(1.25, 85),
        fontsize=9, color="#888", ha="center",
        arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.2),
    )
    ax.annotate(
        "Shoots 73, lead\ncollapses to 0",
        xy=(2.5, 55), xytext=(2.5, 90),
        fontsize=9, color="#888", ha="center", style="italic",
        arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.2),
    )
    ax.annotate(
        "Holds on, wins\nby 1 stroke",
        xy=(3, 55), xytext=(3.3, 75),
        fontsize=9, color="#888", ha="center", style="italic",
        arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.2),
    )

    # Add value labels on each point
    for xi, mi, ki in [(1, 12, 25), (2, 34, 74), (3, 55, 36)]:
        ax.text(xi, mi - 4, f"{mi}%", ha="center", fontsize=10, color="#2a9d8f", fontweight="bold")
        offset = 4 if ki > mi else -5
        ax.text(xi, ki + offset, f"{ki}%", ha="center", fontsize=10, color="#e76f51", fontweight="bold")
    ax.text(0, market[0] + 3, f"{market[0]}%", ha="center", fontsize=10, color="#e76f51", fontweight="bold")

    # Legend with custom patches
    legend_elements = [
        plt.Line2D([0], [0], color="#2a9d8f", linewidth=2.5, marker="o", markersize=8, label="Our model"),
        plt.Line2D([0], [0], color="#e76f51", linewidth=2.5, marker="o", markersize=8, label="Prediction market"),
        mpatches.Patch(facecolor="#e76f51", alpha=0.15, edgecolor="none", label="Model says BUY NO (overpriced)"),
        mpatches.Patch(facecolor="#2a9d8f", alpha=0.15, edgecolor="none", label="Model says BUY YES (underpriced)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left", framealpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("McIlroy Win Probability (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.3, 3.5)
    ax.grid(True, alpha=0.2)

    ax.set_title(
        "2026 Masters: Our Model vs. Prediction Markets\n"
        "The model faded the crowd in both directions -- and was right both times",
        fontsize=13, fontweight="bold",
    )

    # Bottom text with the trade summary
    ax.text(
        0.5, -0.13,
        "Trades: BUY NO at 26c (Fri) -> SELL NO at 64c (Sat) = +38c  |  "
        "BUY YES at 36c (Sat) -> McIlroy wins = +64c  |  "
        "Net: +$1.02 on 26c",
        transform=ax.transAxes, fontsize=10, ha="center", color="#264653", fontweight="bold",
    )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "masters_2026_trading_timeline.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_simple_timeline()
