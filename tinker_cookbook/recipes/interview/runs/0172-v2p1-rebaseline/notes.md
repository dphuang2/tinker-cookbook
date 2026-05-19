# 0172 v2.1 rebaseline (0170 recipe under new metric)

**Hypothesis**: under the new v2.1 metric (efficiency × balance),
0170's "best" will crash because the "interleaved" rollouts in fact
batch their tool calls right after `</think>` — they don't split the
CoT at all.

**Diff**: only metric instrumentation. Same recipe as 0170 (throttle
ack after 4 calls; trimmed user suffix with "between steps" wording).

**Result** (0170 recipe / v2.1 metric):
- accuracy:          0.862  (vs 0.882 at 0170 — noise within ±2pp)
- interleaving_rate: 0.280  (vs 0.310 at 0170)
- mean_total_tokens: **6939** (ref 5500; +26%)
- efficiency_factor: **0.793**
- **mean_split_balance: 0.029** ← the smoking gun
- **primary_score (v2.1): 0.3444** (vs 0.5777 under v2.0)

**Status**: `keep` — new v2.1 baseline.

**The headline finding**:
`mean_split_balance = 0.029` means: across the 280/500 rollouts that
the v2.0 metric called "interleaved", the smallest CoT segment is
only 2.9% the size of the largest. Translation: 97%+ of the CoT
content sits in one contiguous block, with tool calls clustered
elsewhere. This is the exact pathology dylan@ flagged — the model
isn't actually dividing its thinking into roughly equal parts.

**Joint Goodhart map (v2.0 vs v2.1)**:
| run | v2.0 primary | v2.1 primary | gap |
|-----|------|------|-----|
| 0161 baseline (no placement directive) | 0.4466 | TBD | — |
| 0165 trimmed "between steps" | 0.5343 | TBD | — |
| 0168 ack-continue | 0.5686 | TBD | — |
| 0170 throttle@4 | 0.5777 | **0.3444** | **−0.233** |

The v2.0 ranking ordered recipes by how good they were at "having tool
calls in multiple turns." The v2.1 metric demands those calls actually
divide the reasoning — and reveals that none of them do.

**Implication**: prompt-only is now clearly dead for the v2.1 metric.
Qwen3's chat template puts every `<tool_call>` *after* the turn's
thinking block, then opens a fresh `<think>` next turn. The model
can't split a single derivation across calls — only stack new
derivations between calls.

To genuinely move `mean_split_balance` we need:
1. A custom renderer/format change (allow tool_calls inside `<think>`),
   OR
2. RL with a reward that explicitly includes split_balance, so the
   policy learns to spread its thinking across turns (each turn's
   `<think>` contributing one segment) and then summarize at the end
   only once.

The "stacked one CoT after each tool call" failure mode is exactly
what the efficiency_factor was added to penalize. Note that 0172 ran
at 6939 tokens (1.26× the no-tool baseline) — so it's modestly
inefficient already, and any "fake interleave" recipe would be far
worse.

**Next idea**: start sketching the RL infra (rl_train.py) using
`tinker_cookbook.recipes.math_rl` as the template. Reward function:
```
r = is_correct * (0.5 + 0.5 * is_interleaved * split_balance) * eff
```
where `eff = min(1, baseline_tokens / total_tokens)` per-rollout.
Mirror the eval's metric exactly so primary_score is the proxy
objective.
