# 0191 phaseB-rft-fixed-datum — NEW V3 BEST

**Hypothesis**: 0190's tool-abandonment was a *datum-construction
bug*, not a fundamental issue with SFT-on-positives. Reconstructing
tool_calls as proper `ToolCall` objects (not stuffing JSON into
content text) lets `build_supervised_example` emit the correct chat
template, so SFT actually learns to produce tool_calls.

**Diff**:
- `rft_sample.py`: serialize tool_calls as JSON-safe dicts (id, type,
  function.name, function.arguments) rather than via `default=str`.
- `rft_train._build_datum_from_record`: rebuild `tinker_cookbook.
  renderers.ToolCall` objects from those dicts into the assistant
  message's structured `tool_calls` field. No more "append
  `<tool_call>...</tool_call>` to content text" hack.

**Sampling** (OPSD step_2, v2.2 scoring, threshold 0.2):
- 800 rollouts → 141 positives
- score_mean 0.258, score_max 0.926
- frac_correct 0.868, frac_interleaved **0.988**

**Training**: 36 steps (2 epochs × batch 8), mean_lp converged
from -0.6 → -1.0 to -1.2. The higher mean_lp than 0186/0187
suggests the tool_call tokens are now real SFT targets (harder to
predict from base prior).

**Eval result vs 0187 (prior best)**:
| metric              | 0187 (broken serialization) | **0191 (fixed)** |
|---------------------|----------------------------:|-----------------:|
| accuracy            | 0.834                        | **0.834**       |
| in_think_rate       | 0.004                        | 0.000           |
| **turn_split_rate** | 0.638                        | **0.910**       |
| mean_split_balance  | 0.515                        | 0.494           |
| mean_total_tokens   | 8187                         | 10349           |
| efficiency          | 0.672                        | 0.531           |
| **primary_score**   | **0.184**                    | **0.1994**      |

Cadence histogram: 9 skip, 36 at 1 call, 123 at 2, **273 at 3**,
58 at 4, 1 at 5. Mode shifted to 3 calls (was 2 in 0187), with
much sharper concentration.

**Status**: `keep` — new v3 best at primary_score 0.1994.

**Why this is a real improvement over 0187**:
- 91% of rollouts cross-turn interleave (vs 64% in 0187) — the
  model genuinely emits tool calls almost always
- mean_split_balance roughly equal (placement quality preserved)
- Trade-off: 26% more tokens (efficiency 0.53 vs 0.67), but the
  interleaving lift more than compensates in the v2.2 metric

**v3 leaderboard (under v2.2)**:
| Run                                | v2.2 score |
|------------------------------------|-----------:|
| **0191 RFT fixed datum**           | **0.1994** |
| 0187 RFT broken-serial (1st iter)  | 0.1841     |
| 0181 OPSD step_2                   | 0.1656     |
| 0190 RFT v2.2-filtered (buggy)     | 0.0088     |
| 0188 RFT expert iter (Goodhart)    | 0.0051     |
| v2.1 base 0172                     | 0.0056     |

**Next ideas**:
1. **0192 RFT iteration on 0191**: sample from 0191 step_36_final,
   filter, retrain. Should produce a model that's even more
   confidently-interleaved (the dataset will all come from a
   91%-interleaved sampler).
2. **0193 stricter threshold**: re-sample from OPSD step_2 with
   threshold 0.3 (was 0.2). Tighter filter, fewer positives but
   higher quality.
3. **0194 OPSD-warm-start RFT**: requires state path from OPSD —
   needs re-running OPSD with the state-saving from 42cce7b.
