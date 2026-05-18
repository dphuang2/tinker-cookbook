# 0100 shorter-system-prompt

**Hypothesis**: 0095 system prompt has "for tracking your reasoning progress on hard multi-step problems" — possibly redundant given the user message anchors cadence. Trim to "for tracking progress."

**Result**: accuracy **0.882**, cadence `0:292, 1:16, 2:11, 3:122, 4:38, 5:10, 6:7, 7:1, 8:1, 9:1, 16:1`. 58.4% 0-call.

**Status**: `keep` (parity with 0095; cleaner wording wins at tie).

| Run | accuracy | description |
|-----|----------|-------------|
| 0095 mean (n=4) | 0.878 | longer system prompt |
| 0100 (n=1) | 0.882 | trimmed system prompt |
| no-tool baseline | 0.880 | — |

Single-sample 0.882 at the top end of 0095's noise band. Cadence identical shape.

**Take-away**: the "for hard multi-step problems" qualifier in system prompt is decorative. The user-message directive carries the actual cadence anchor.

**Action**: keep shorter system prompt. Current recipe is genuinely minimal.

**Best**: 0100-recipe (current).

**Next**: variance corroboration of 0100.
