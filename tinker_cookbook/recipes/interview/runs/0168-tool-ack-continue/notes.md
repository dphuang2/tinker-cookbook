# 0168 tool-ack-continue (NEW BEST)

**Hypothesis**: tool ack content matters. Changing the ack from "ok"
to "noted; continue your reasoning" should explicitly nudge the model
to keep thinking rather than emit another checkpoint, lifting the
interleaved shape without an accuracy cost. Now runnable under
0167's robust eval.

**Diff**: `eval_deepmath_agent.py:125` — `"content": "ok"` →
`"content": "noted; continue your reasoning"`. No other recipe change
(USER_INSTRUCTION_SUFFIX still 0165's trimmed-between-steps).

**Result vs 0165 (prior best)**:
- accuracy:          0.810   (was 0.812, −0.2 pp — flat)
- in_think_rate:     **0.022**  (was 0.004, **+1.8 pp / 5× lift**)
- turn_split_rate:   **0.400**  (was 0.316, **+8.4 pp**)
- interleaving_rate: **0.404**  (was 0.316, **+8.8 pp**)
- **primary_score:   0.5686** (was 0.5343, **+0.034 → new v2 best**)
- num_errored: 1/500 (single context-overflow handled gracefully)

**Status**: `keep` — new v2 best.

**Why this worked**:
1. The original "ok" ack is a terminal signal — the model interprets
   it as "I'm done with that turn, time to write the next message."
   Combined with the chat-template prior, "next message" defaults
   to either another tool_call (batched mode) or the final answer
   (skip mode).
2. "noted; continue your reasoning" gives the model a *direction* —
   it's expected to keep working. This creates space for further
   `<think>` content in the next turn, including (rarely)
   `<tool_call>` inside that thinking.
3. `in_think_rate` quintupled (0.4% → 2.2%) while accuracy held
   flat. That's the cleanest signal yet that something is moving
   the *placement* metric, not just the count.
4. Cadence tail is slightly heavier than 0165 (more 24-40 outliers)
   but mode is still at 3.

**Where we are now**:
- 0161 baseline:           0.4466
- 0162 explicit-between:   0.5058
- 0163 cap-at-four:        0.5145
- 0164 exactly-three:      0.2280 (discard)
- 0165 trimmed-between:    0.5343
- 0166 sys-prompt-place:   0.4740 (discard)
- 0167 ack-continue v1:    crash (context overflow, eval-killing)
- **0168 ack-continue v2: 0.5686** (new best, robust eval)

**Next idea**: stack — combine the ack-continue signal with another
small lever. Try a one-sentence anti-batch hint added back to
USER_INSTRUCTION_SUFFIX: "After each checkpoint, the tool will
respond and you continue thinking." This explicitly primes the model
for the new ack format, telling it the response indicates a
continuation rather than termination.
