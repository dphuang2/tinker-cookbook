# 0164 exactly-three (FAILED — discard)

**Hypothesis**: stricter cap with structural anchors ("exactly three
times: once after problem setup, once after derivation, once before
answer") would pin cadence to 3 and recover the accuracy lost in 0163.

**Diff**: `USER_INSTRUCTION_SUFFIX` rewritten with "exactly three" +
explicit positional anchors.

**Result vs 0163**:
- accuracy:          **0.236**  (was 0.736, **−50 pp catastrophe**)
- in_think_rate:     0.020   (was 0.004, +1.6 pp)
- turn_split_rate:   **0.932**  (was 0.396, +53.6 pp)
- interleaving_rate: **0.932**  (was 0.398, +53.4 pp)
- primary_score:     **0.2280** (was 0.5145, **−0.287**)

**Status**: `discard` — primary_score crashed below baseline.

**Why this failed catastrophically**:
1. Cadence pinned not to 3 but to **24** (245/500 rollouts!) plus
   long tail including 36 and 64.
2. 24 = 3 × MAX_TURNS (8). The model parsed "exactly three times" as
   "exactly three per turn" → emits 3 batched tool_calls per
   assistant turn → consumes all 8 turns without writing an answer.
3. Final visible answer is empty/truncated for most rollouts → 0.236
   accuracy is mostly extract_ok=False.
4. turn_split_rate=0.932 is a Goodhart artifact: tool calls are
   "split across turns" but the structure is "tool/tool/tool" each
   turn until max_turns, never breaking out to write the answer.
5. Eval cycle took 35+ min vs typical 8–14 min — every problem
   blowing through all 8 turns.

**Lesson**: explicit "exactly N" prompts are dangerous with this
agent loop — they couple to MAX_TURNS in unexpected ways. The
"about three, four max" hedging in 0163 was load-bearing for the
same reason verbose hedging was load-bearing in v1.

**Reverting** `USER_INSTRUCTION_SUFFIX` to the 0163 state. Next
experiment should diverge from the prompt-only path — the in_think
wall isn't moving with prompt edits.
