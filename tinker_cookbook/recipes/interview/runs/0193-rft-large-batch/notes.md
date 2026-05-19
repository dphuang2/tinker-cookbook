# 0193 phaseB-rft-large-batch — NEW V3 BEST

**Hypothesis**: doubling the sample budget (400 problems × 4 vs 200
× 4) gives wider problem coverage and more high-quality positives,
producing a more robust SFT signal than 0191.

**Pipeline**:
- Sample: 1600 rollouts on OPSD 0181 step_2 across 400 train problems
- Filter: v2.2 score > 0.2, top-1 per problem → 266 positives
- Train: 68 steps (2 epochs × batch 8), from-scratch LoRA rank=32

**Sample stats**:
- score_mean 0.253, score_max 0.935
- frac_correct 0.826, frac_interleaved 0.971

**Eval result vs 0191 (prior best)**:
| metric              | 0191 (141 pos) | **0193 (266 pos)** | Δ |
|---------------------|---------------:|-------------------:|---:|
| accuracy            | 0.834          | **0.854**          | +2.0 pp |
| in_think_rate       | 0.000          | 0.000              | 0 |
| turn_split_rate     | 0.910          | 0.938              | +2.8 pp |
| mean_split_balance  | 0.494          | 0.478              | −0.016 |
| mean_total_tokens   | 10349          | 10411              | flat |
| efficiency_factor   | 0.531          | 0.528              | flat |
| **primary_score**   | **0.1994**     | **0.2025**         | **+0.003** |

Cadence histogram: 1 skip, 30/108/288/72/1 at 1/2/3/4/5 calls. Mode
at 3 calls (288/500 = 58%), very few skips, no runaway tail.

**Status**: `keep` — new v3 best by thin margin.

**Where the improvement came from**: the 2-pp accuracy gain
(0.834 → 0.854). With twice the training data, the model preserved
accuracy better while learning the interleaving pattern. The
placement quality metrics (turn_split_rate, split_balance, tokens)
were already saturated at 0191 levels — the lift is purely from
accuracy.

**Next idea**: try **stricter filtering** at the same sample
budget (1600 rollouts), threshold 0.4 → likely ~80-100 positives
but every one is a high-quality rollout. The 0193 analysis showed
the worst-kept positives at threshold 0.2 are inefficient
(11k tokens, eff=0.47). Removing them may help the model learn a
cleaner shape without the token-bloat noise.

**Locked-in v3 best**: **0193 at primary_score 0.2025**.
