"""Render audit plots from experiments/opd_rl/data/ → experiments/opd_rl/figures/.

Run: `uv run python -m experiments.opd_rl.make_figures`
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).parent / "data"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)


def load_metrics(iter_id: str) -> list[dict]:
    p = DATA / f"iter{iter_id}" / "metrics.jsonl"
    return [json.loads(l) for l in p.open()]


def smooth(xs, w=3):
    xs = np.asarray(xs, float)
    if len(xs) < w:
        return xs
    pad = w // 2
    kernel = np.ones(w) / w
    s = np.convolve(np.concatenate([np.full(pad, xs[0]), xs, np.full(pad, xs[-1])]), kernel, mode="valid")
    return s[: len(xs)]


def get_curve(iter_id: str, key: str = "env/all/correct"):
    rows = load_metrics(iter_id)
    steps = [r.get("progress/batch", i) for i, r in enumerate(rows)]
    vals = [r.get(key, np.nan) for r in rows]
    return np.array(steps), np.array(vals, float)


# Teacher baseline (constant)
teacher_ref = json.loads((DATA / "teacher_ref.json").read_text())
teacher_correct = teacher_ref["correct_mean"]


# ---- Figure 1: training curves ----
fig, ax = plt.subplots(figsize=(10, 6))
runs = [
    ("04", "OPD only (30 steps)", "tab:blue", "-"),
    ("05", "RL matched-hparams (collapse)", "tab:red", "-"),
    ("06", "RL tuned (30 steps)", "tab:orange", "-"),
    ("12", "RL tuned (60 steps)", "tab:orange", "--"),
    ("07", "OPD-then-RL (30 steps)", "tab:green", "-"),
    ("10", "OPD-then-RL (60 steps)", "tab:green", "--"),
    ("08", "OPD-then-RL seed=2", "tab:cyan", ":"),
    ("09", "RL tuned seed=2", "tab:olive", ":"),
]
for iter_id, label, color, ls in runs:
    s, c = get_curve(iter_id)
    ax.plot(s, c, color=color, linestyle=ls, alpha=0.35)
    ax.plot(s, smooth(c, 5), color=color, linestyle=ls, label=label, linewidth=2)

ax.axhline(teacher_correct, color="black", linestyle="--", linewidth=1.2, label=f"teacher zero-shot ({teacher_correct:.1%})")
ax.set_xlabel("training step")
ax.set_ylabel("env correct rate (per-batch mean)")
ax.set_title("Countdown: per-step accuracy across training variants")
ax.set_ylim(-0.02, 1.0)
ax.set_xlim(-1, 65)
ax.legend(loc="upper left", fontsize=9, ncol=1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "training_curves.png", dpi=140)
print("wrote", FIG / "training_curves.png")
plt.close(fig)


# ---- Figure 2: KL trajectory for OPD run ----
fig, ax = plt.subplots(figsize=(8, 4.5))
s, kl = get_curve("04", "teacher_kl")
ax.plot(s, kl, color="tab:blue", label="OPD: teacher_kl")
ax.set_xlabel("training step")
ax.set_ylabel("teacher KL")
ax.set_title("OPD (iter04): teacher-KL decreases as student approaches teacher")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "opd_kl.png", dpi=140)
print("wrote", FIG / "opd_kl.png")
plt.close(fig)


# ---- Figure 3: final-asymptote bar ----
def last10_mean(iter_id, key="env/all/correct"):
    _, c = get_curve(iter_id, key)
    return float(np.mean(c[-10:]))


variants = [
    ("base zero-shot", "—", 0.28, "tab:gray"),  # from iter04 step 0 avg
    ("teacher zero-shot", "—", teacher_correct, "black"),
    ("OPD only (30)", "04", last10_mean("04"), "tab:blue"),
    ("RL matched (30)", "05", last10_mean("05"), "tab:red"),
    ("RL tuned (30)", "06", last10_mean("06"), "tab:orange"),
    ("RL tuned (60)", "12", last10_mean("12"), "tab:brown"),
    ("OPD-then-RL (30)", "07", last10_mean("07"), "tab:green"),
    ("OPD-then-RL seed=2 (30)", "08", last10_mean("08"), "tab:cyan"),
    ("RL tuned seed=2 (30)", "09", last10_mean("09"), "tab:olive"),
    ("OPD-then-RL (60)", "10", last10_mean("10"), "tab:purple"),
]
fig, ax = plt.subplots(figsize=(10, 5.5))
ys = np.arange(len(variants))
xs = [v[2] for v in variants]
colors = [v[3] for v in variants]
labels = [v[0] for v in variants]
ax.barh(ys, xs, color=colors, alpha=0.85)
for y, x in zip(ys, xs):
    ax.text(x + 0.005, y, f"{x:.3f}", va="center", fontsize=9)
ax.axvline(teacher_correct, color="black", linestyle="--", linewidth=1, alpha=0.6)
ax.set_yticks(ys)
ax.set_yticklabels(labels)
ax.set_xlabel("mean correct rate (last 10 steps)")
ax.set_xlim(0, 0.85)
ax.set_title("Asymptote across variants (mean correct over last 10 training steps)")
ax.invert_yaxis()
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "asymptote_bar.png", dpi=140)
print("wrote", FIG / "asymptote_bar.png")
plt.close(fig)


# ---- Figure 4: forgetting eval ----
labels16 = ["base", "OPD-30", "RL matched (collapse)", "RL tuned", "OPD-then-RL"]
scores16 = [
    json.loads((DATA / "forgetting-base.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting-opd30.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting-rl-matched.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting-rl-tuned.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting-opd-then-rl.json").read_text())["forgetting_score"],
]
labels26 = labels16 + ["OPD-then-RL-60", "RL-tuned-60"]
scores26 = [
    json.loads((DATA / "forgetting2-base.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-opd30.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-rl-matched.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-rl-tuned.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-opd-then-rl.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-opd-then-rl-60.json").read_text())["forgetting_score"],
    json.loads((DATA / "forgetting2-rl-tuned-60.json").read_text())["forgetting_score"],
]

fig, ax = plt.subplots(figsize=(10, 5))
x16 = np.arange(len(labels16))
x26 = np.arange(len(labels26)) + 0.0
width = 0.4
ax.bar(np.arange(len(labels16)) - width / 2, scores16, width, label="16-prompt rubric (iter08)", color="tab:blue")
ax.bar(np.arange(len(labels26)) + width / 2, scores26, width, label="26-prompt rubric (iter11)", color="tab:orange")
for i, s in enumerate(scores16):
    ax.text(i - width / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
for i, s in enumerate(scores26):
    ax.text(i + width / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=8)
ax.set_xticks(np.arange(len(labels26)))
ax.set_xticklabels(labels26, rotation=15, ha="right")
ax.set_ylabel("forgetting score (higher = less forgetting)")
ax.set_title("Instruction-following preservation across checkpoints")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "forgetting_bar.png", dpi=140)
print("wrote", FIG / "forgetting_bar.png")
plt.close(fig)


# ---- Figure 5c: cold-start regime (countdown-v2) ----
cold_runs = [
    ("15", "RL-only", "tab:red"),
    ("16", "OPD-only", "tab:blue"),
    ("17", "OPD-then-RL", "tab:purple"),
    ("19", "SFT-then-RL", "tab:green"),
]
fig, ax = plt.subplots(figsize=(10, 5.5))
for iter_id, label, color in cold_runs:
    try:
        s, c = get_curve(iter_id)
        ax.plot(s, c, color=color, alpha=0.3)
        ax.plot(s, smooth(c, 5), color=color, label=label, linewidth=2)
    except FileNotFoundError:
        continue
# Add SFT-only horizontal line from sft_eval_v2.json
try:
    import json as _json
    sft_eval = _json.loads((DATA / "sft_eval_v2.json").read_text())
    ax.axhline(sft_eval["correct_mean"], color="tab:olive", linestyle=":", linewidth=2, label=f"SFT-only pass@1 ({sft_eval['correct_mean']:.1%})")
except Exception:
    pass
ax.set_xlabel("training step")
ax.set_ylabel("env correct rate (per-batch mean)")
ax.set_title("Cold-start regime (countdown-v2): OPD fails, SFT wins")
ax.set_ylim(0, 0.75)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "cold_start_v2.png", dpi=140)
print("wrote", FIG / "cold_start_v2.png")
plt.close(fig)


# ---- Figure 5b: fair 60-step head-to-head ----
fig, ax = plt.subplots(figsize=(10, 5.5))
for iter_id, label, color in [("10", "OPD-then-RL (60 steps)", "tab:purple"), ("12", "RL tuned (60 steps)", "tab:brown")]:
    s, c = get_curve(iter_id)
    ax.plot(s, c, color=color, alpha=0.3)
    ax.plot(s, smooth(c, 5), color=color, label=label, linewidth=2.2)
ax.axhline(teacher_correct, color="black", linestyle="--", linewidth=1.2, label=f"teacher zero-shot ({teacher_correct:.1%})")
ax.set_xlabel("training step")
ax.set_ylabel("env correct rate (per-batch mean)")
ax.set_title("Apples-to-apples at 60 steps: OPD-then-RL vs RL-tuned (both with same hparams)")
ax.set_ylim(0, 0.9)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fair_60step.png", dpi=140)
print("wrote", FIG / "fair_60step.png")
plt.close(fig)


# ---- Figure 6: per-decade trajectory iter10 ----
s, c = get_curve("10")
decades = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
means = [float(np.mean(c[a:b])) for a, b in decades]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar([f"{a}-{b - 1}" for a, b in decades], means, color="tab:purple", alpha=0.85)
for i, m in enumerate(means):
    ax.text(i, m + 0.01, f"{m:.2f}", ha="center", fontsize=9)
ax.axhline(teacher_correct, color="black", linestyle="--", linewidth=1, label=f"teacher ({teacher_correct:.2f})")
ax.set_ylabel("mean correct rate")
ax.set_xlabel("training step range")
ax.set_title("60-step OPD-then-RL (iter10): per-decade mean correct rate")
ax.set_ylim(0, 0.85)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "iter10_decades.png", dpi=140)
print("wrote", FIG / "iter10_decades.png")
plt.close(fig)
