# 0192 phaseB-rft-iter-on-0191 — discard (regression)

**Hypothesis**: expert iteration — sample from 0191 step_36_final,
filter top-1 per problem at threshold 0.2 (v2.2 score), RFT-from-
scratch — would tighten the policy further and improve primary_score.

**Pipeline**: identical to 0191 but sourced from 0191 sampler instead
of OPSD step_2. 158 positives, all interleaved by definition.

**Eval result vs 0191 (prior best)**:
| metric              | 0191   | **0192**   | Δ |
|---------------------|-------:|-----------:|---:|
| accuracy            | 0.834  | 0.832      | flat |
| turn_split_rate     | 0.910  | **0.128**  | **−78 pp** |
| mean_split_balance  | 0.494  | 0.464      | flat |
| mean_total_tokens   | 10349  | **5646**   | **−45%** |
| efficiency_factor   | 0.531  | **0.974**  | +0.44 |
| **primary_score (v2.2)** | **0.1994** | **0.0481** | **−0.151** |

Cadence: 240 skip, 195 at 1 call, 53 at 2, 12 at 3.

**Status**: `discard` — model drifted toward sparse tool use.

**Why this happened (likely)**:
1. **RFT-from-scratch on filtered-on-policy positives is unstable**.
   When the source sampler (0191) is already trained, its positives
   are a narrow distribution. Refitting from scratch on this narrow
   distribution amplifies whatever non-tool patterns leaked into the
   filtered positives.
2. **The score formula's `is_interleaved` is binary (0/1)**.
   So a positive with 2 calls and bal=0.45 scores the same way (in
   the interleaved factor) as one with 6 calls and bal=0.45. The
   filter doesn't reward higher cadence — only that it cross-turn.
   The model may have learned that 1 turn-split is "good enough"
   even though such rollouts wouldn't pass our filter.
3. **No KL regularization to the prior sampler**. From-scratch SFT
   freely abandons whatever doesn't appear in the filtered data.

**Implication**: 0191 was the right number of RFT iterations. The
gradient direction "tighter selection → more iteration" doesn't
hold; you over-narrow.

**Better next directions**:
1. **RFT *from 0191's weights*** (not from scratch) — needs a state
   path. We don't have one for 0191; would require re-running RFT
   with state-saving on 0191's positives.
2. **Larger positive dataset** — 158 is small. 500+ problems × 8
   samples → potentially 400+ positives, less overfitting risk.
3. **OPSD-warm-start RFT** — chain training (OPSD state → RFT
   continue) rather than from-scratch.

**Locked-in v3 best: 0191 at primary_score 0.1994.**
