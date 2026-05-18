# 0117 directives-prepended

**Hypothesis**: directives currently appended after question. Try prepending — does question-last format improve compliance attention?

**Diff**: eval_deepmath_agent.py user message construction: `USER_INSTRUCTION_SUFFIX + question` (prepended) instead of `question + USER_INSTRUCTION_SUFFIX` (appended).

**Result**: accuracy **0.876**, cadence `0:359, 1:4, 3:126, 4:6, 5:1, 6:3, 9:1`. **71.8% 0-call (28% tool use)**.

**Status**: `discard`. Accuracy at parity (within noise of 0105 mean 0.8744), but cadence weakened significantly. Tool use dropped from ~50% (0105) to 28%.

**Take-away**: directive position matters for cadence. Appended-after-question directives keep the tool prominent in attention (close to where the answer is generated); prepended directives get attention-decayed by the question that follows.

This is consistent with the 0097 finding (directives in system prompt also suppress cadence).

**Action**: revert eval_deepmath_agent.py to append directives after question.

**Best remains 0105-recipe at 5-sample mean 0.8744.**
