# 0071 directive-reorder

**Hypothesis**: in 0062's USER_INSTRUCTION_SUFFIX, "Write your answer in \boxed{}" appears FIRST. Recency bias suggests the user message's last sentence has the most weight on the model's output. Reordering with boxed-format LAST might tighten compliance + free attention budget for tool/reasoning guidance up front.

**Diff**: reordered USER_INSTRUCTION_SUFFIX: tool guidance → anti-rumination → boxed-format (last).

**Result**: accuracy **0.854**, cadence `0:396, 1:56, 2:20, 3:9, 4:6, 5:3, 6:3, 7:1, 8:6`. 79% 0-call.

**Status**: `discard`. -1.3pp vs 0062 2-sample mean (0.867). Outside 3σ noise band — genuine regression.

**Interpretation**: contrary to recency hypothesis, putting boxed-format last hurt. Possibly because: (a) the boxed-format instruction is the *most concrete* compliance signal and should anchor the start, (b) the tool-guidance text is naturally less weight-sensitive to position, (c) the 0062 ordering happens to fit the model's expected input shape.

**Take-away**: position matters; 0062's ordering is non-arbitrary. Recency bias intuition fails here.

**Action**: revert USER_INSTRUCTION_SUFFIX to 0062 ordering.

**Best remains 0062 at 0.867 (2-sample mean).**

**Pattern after 10 post-0062 experiments**: extremely sharp local optimum. 0064 (degenerate cadence) is the only variant that breaks it on accuracy, and that's disqualified.

**Next ideas**:
1. **Punctuation/formatting tweaks** — replace " -- " with em-dashes; replace dashes with commas; etc. Likely no-op; cheap to confirm.
2. **Try the agent loop ack message variations** — "ok" vs "" vs "continue". Already explored some; could revisit.
3. **Investigate whether single-shot temperature variation helps** — but temp is FIXED.
4. **System prompt with terse rubric** — separate channel from user message.

Picking #4 (system prompt rubric): genuinely orthogonal to USER_INSTRUCTION_SUFFIX wording. Tests whether moving reasoning emphasis to a different message channel helps.
