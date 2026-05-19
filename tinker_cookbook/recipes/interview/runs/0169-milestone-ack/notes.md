# 0169 milestone-ack (discard)

**Hypothesis**: a more directive ack — "noted; keep thinking, and
call me again at your next milestone" — would push interleaving even
higher than 0168's "noted; continue your reasoning" while keeping
accuracy.

**Diff**: only the tool ack content.

**Result vs 0168 (best)**:
- accuracy:          **0.630**  (was 0.810, **−18 pp catastrophic**)
- in_think_rate:     0.020   (was 0.022, flat)
- turn_split_rate:   **0.710**  (was 0.400, **+31 pp**)
- interleaving_rate: 0.710   (was 0.404, +30.6 pp)
- primary_score:     0.5386  (was 0.5686, **−0.030 → discard**)

**Status**: `discard`. Reverted ack to 0168's wording.

**Why this failed**:
1. "call me again at your next milestone" is a direct instruction
   to keep calling. The model obliged: 51 rollouts skipped tool
   entirely (down from 174 baseline / 43 in 0168), but the rest
   spread cadence across the full range with heavy mass at 4–10
   calls (and a 14-rollout cluster at 32 calls).
2. The dispersion of cadence drained accuracy. Lots of rollouts
   chasing checkpoints instead of finishing the problem.
3. `in_think_rate` flat at 0.020 — the placement directive in
   ack does NOT move within-think calls. It only redistributes
   across turns. The chat-template wall on `<tool_call>` inside
   `<think>` is unmoved by any ack variant.

**Insight**: ack content has a Goldilocks zone. 0168's "continue
your reasoning" is closer to optimal than either "ok" (no nudge)
or "call me at next milestone" (over-encouraging). The right ack
implies continuation without prescribing another tool call.

**Next idea**: stop iterating on ack wording (saturated). Try a
*per-rollout cap* in the agent loop — if `num_tool_calls >= 4`,
replace the ack with "you've checkpointed enough; finalize your
answer now." This is a recipe change (the agent loop sees state
and adapts the ack), keeping the win from 0168 while killing the
runaway tail.
