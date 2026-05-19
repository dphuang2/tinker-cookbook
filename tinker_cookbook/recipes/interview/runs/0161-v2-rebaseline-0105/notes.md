# 0161 v2 rebaseline (0105 recipe under interleaving metric)

**Hypothesis**: 0105 prompt-only recipe will reproduce v1 accuracy
(~0.875) but score poorly on the new interleaving metric — confirming
the post-hoc audit that placement was never being optimized.

**Diff**: none to the recipe itself. Only change is the v2 eval
instrumentation committed in 0160. This is the v2 ceiling for
prompt-only.

**Result**:
- accuracy: 0.874 (consistent with v1 26-sample mean 0.8750 ± 0.010)
- in_think_rate: **0.000** — zero mid-thinking tool calls
- turn_split_rate: 0.022 (11/500 split tool calls across ≥2 turns)
- interleaving_rate: 0.022
- **primary_score: 0.4466**

**Status**: `keep` — this is the v2 reference baseline. Everything
else compares against `primary_score=0.4466`.

**Takeaways**:
1. `in_think_rate=0.000` is a *categorical* failure of the v1 goal.
   The chat-template prior fully suppresses `<tool_call>` inside
   `<think>...</think>`. SFT will not fix this without a renderer
   change or a custom format.
2. Only ~2.2% of rollouts even split tool calls across turns —
   matching the post-hoc audit (10/500 in run 0152).
3. To improve, we need either:
   - A **format change** that lets the model emit checkpoints
     inside `<think>` (custom renderer, or special tokens)
   - **RL** with a placement-aware reward starting from these rare
     positives
   - A **stronger prompt** with an in-context demonstration of
     mid-think calls (cheap to try, may unlock the shape)

**Next idea**: cheap prompt-only swing first — add an explicit
one-shot demonstration showing think → call → think → call → answer
in the user message, before resorting to RL infra. If a demo unlocks
in_think_calls without RL, that's the simplest win.
