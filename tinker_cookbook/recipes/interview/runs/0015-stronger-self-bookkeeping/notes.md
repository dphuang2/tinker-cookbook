# 0015 stronger-self-bookkeeping

**Hypothesis**: 0011's "for your own bookkeeping" framing gave +0.4pp over 0002. Pushing the framing even harder ("your private scratchpad", "they're for you", "call it every time you finish a sub-step") might amplify the effect.

**Diff**: sft_train.py PROGRESS_TOOL_SPEC description rewritten with more aggressive self-bookkeeping language. Reverted MIN_TOTAL_THINKING_CHARS to 0.

**Training**: 142 steps, final NLL 0.239.

**Result**: accuracy **0.722**, cadence `0:180, 1:218, 2:101, 3:1`. Worse than 0011 (0.740) by 1.8pp.

| Run | accuracy | description style |
|-----|----------|-------------------|
| 0002 | 0.736 | "user can follow along" |
| **0011** | **0.740** | "for your own bookkeeping (user reads to follow along)" |
| 0015 | 0.722 | "your private scratchpad, they're for you" |

**Status**: `discard`. Pushing the framing harder backfired. 0011 was at the sweet spot.

**Theory of why**: "private scratchpad" + "call every time you finish a sub-step" may have pushed the model toward over-calling, breaking reasoning continuity even on simple problems. Or the "they're for you" line creates dissonance — the model can't quite reconcile "the user sees these but they're for you".

**Take-away**: tool description framing has a *narrow* sweet spot. 0011's mid-level framing (acknowledges both audiences) is better than either pure "user-facing" (0002) or pure "internal" (0015).

**Next ideas (still untried)**:
1. **System-prompt addition** — add an instruction line above the tool spec (e.g. "Whenever you make a substantive reasoning step, call `checkpoint(...)`. Use it freely.")
2. **Larger dataset (3500+ records)** — sample more raw traces. Tests if 2300 is the data-bound ceiling.
3. **Replace `_assistant_turn_with_update`** with one that has the tool call BEFORE the thinking content rather than after, to test ordering effects.

Picking #1: cheapest. The system prompt currently only contains the renderer-generated tool spec. Adding a directive line might bias cadence positively.
