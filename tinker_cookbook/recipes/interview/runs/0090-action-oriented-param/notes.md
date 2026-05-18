# 0090 action-oriented-param

**Hypothesis**: replace the u-substitution example in PROGRESS_TOOL_SPEC `message` param description with action-oriented "what you just established and what you're attempting next" — state-transition framing.

**Result**: accuracy **0.856**, cadence `0:269, 1:34, 2:18, 3:123, 4:38, 5:6, 6:5, 7:5, 9:1, 10:1`. 53.8% 0-call.

**Status**: `discard`. -1.85pp vs 0080 4-sample mean 0.8745 (2.6σ regression — outside noise).

**Interpretation**: same lesson as 0066 (shorter tool-spec description hurt). The u-substitution example in the param description was acting as a *concrete reasoning template* — showing the model what good checkpointing looks like. The abstract action-oriented description removes this concrete example.

**Take-away**: concrete examples in tool spec are load-bearing reasoning scaffolding. Don't abstract them away.

**Action**: revert param description.

**Best remains 0080-recipe at 4-sample mean 0.8745.**

**Next ideas**:
1. **Try DIFFERENT concrete example** — maybe a non-math one for generality.
2. **Try adding a 2nd concrete example** — e.g., u-substitution + completing-the-square.
3. **Try a tiny SFT with low LR (5 records)** — minimal nudge.

Picking #2: adds info without removing the working example.
