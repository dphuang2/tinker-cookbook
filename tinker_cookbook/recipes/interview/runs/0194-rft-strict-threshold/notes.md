# 0194 phaseB-rft-strict-threshold — discard

**Hypothesis**: filtering at score ≥ 0.4 (vs 0193's 0.2) removes the
worst kept positives (which the per-positive analysis showed are
inefficient 11k-token rollouts) and produces a cleaner SFT signal.

**Method**: re-filter the 0193 sampling output at threshold 0.4
(no new sampling). 266 → 171 positives. Train 44 steps (2 epochs ×
batch 8) from-scratch LoRA.

**Eval result vs 0193**:
| metric              | 0193 (266 pos) | 0194 (171 pos) | Δ |
|---------------------|---------------:|---------------:|---:|
| accuracy            | 0.854          | 0.842          | −1.2 pp |
| turn_split_rate     | 0.938          | 0.910          | −2.8 pp |
| mean_split_balance  | 0.478          | 0.468          | flat |
| mean_total_tokens   | 10411          | 10336          | flat |
| efficiency          | 0.528          | 0.532          | flat |
| **primary_score**   | **0.2025**     | **0.1908**     | **−0.012** |

**Status**: `discard`. Tighter threshold hurt every metric.

**Meta-lesson**: at this dataset size (~150-250 positives), **dataset
diversity beats per-positive quality**. The bottom 95 positives at
threshold 0.2 brought down individual scores but added problem
coverage and SFT signal. Removing them shrank the data without
improving the average more than the size loss cost.

**Implication for next experiment**: scale UP the sample count to
allow BOTH more data AND stricter filtering. e.g. 800 problems × 4
samples = 3200 rollouts, threshold 0.4 → likely ~300-400 high-q
positives. Best of both worlds.

**Locked-in v3 best stays at 0193: 0.2025.**
