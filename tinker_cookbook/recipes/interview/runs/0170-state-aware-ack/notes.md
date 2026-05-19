# 0170 state-aware-ack (NEW BEST)

**Hypothesis**: change the tool ack as a function of cadence. While
`num_progress_updates < 4`, return "noted; continue your reasoning"
(0168 win). Once it hits 4, switch to "you've checkpointed enough;
finalize your answer now." Goal: keep 0168's placement lift while
killing the runaway tail.

**Diff**: `eval_deepmath_agent.py` agent loop — branching ack based
on `len(progress_updates) >= 4`.

**Result vs 0168 (prior best)**:
- accuracy:          **0.882**  (was 0.810, **+7.2 pp — best v2 acc**)
- in_think_rate:     0.008   (was 0.022, −1.4 pp)
- turn_split_rate:   0.310   (was 0.400, −9.0 pp)
- interleaving_rate: 0.310   (was 0.404, −9.4 pp)
- **primary_score:   0.5777** (was 0.5686, **+0.009 → new best**)
- num_errored:       0/500   (was 1/500, throttle prevents overflows)

**Status**: `keep` — marginal new best.

**Cadence cleanup is the headline**:
- 0168 tail: 24:9, 27:1, 28:1, 30:1, 32:1, 33:1, 40:1
- 0170 tail: 9:4 (max!)

The throttle works exactly as intended: when the model hits 4 calls,
the new ack tells it to finalize, and it does. Mode is now sharply
bimodal at 3 and 4 (335/500 = 67% in that band), with a small skip
tail (31) and small extension (123 between 5–9).

**Why accuracy improved 7.2pp**:
- Previous runs had ~10% of rollouts in pathological "spam tool
  calls until max_turns=8" mode, all scoring 0.
- The throttle eliminates that failure mode entirely — every rollout
  now either skips or hits a reasonable count and exits.
- Accuracy of 0.882 nearly matches the no-tool baseline (0.880),
  meaning the tool calls no longer cost any net reasoning quality.

**Trade-off**: interleaving_rate dropped 9pp because the throttle
discourages cross-turn cadence in those edge cases that were
previously responsible for the long tail. But those were the LOW-
accuracy rollouts anyway, so primary_score still goes up.

**Implication for RL design**: a hard cap on cadence (or a reward
penalty for >N calls) is essential to prevent runaway. RL would
learn this naturally with a correctness-dominant reward, but it's
useful that we now have a prompt-only/agent-loop baseline at
0.882 accuracy showing the cadence shape isn't load-bearing for
correctness — it's purely the placement signal we're shaping.

**Next idea**: lower the throttle threshold to 3 (since the natural
mode is 3 calls) — this should force the model to either finalize
right at the typical anchor or skip. Tests whether 4 is the right
threshold or if we can push more rollouts toward earlier completion.
