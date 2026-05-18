# 0078 0076-minus-anti-rumination

**Hypothesis**: 0068 showed dropping "Don't think for too long" hurt under the 0062 recipe (anti-rumination was load-bearing). Test the same removal under the 0076 recipe — maybe CoT prefix obviates anti-rumination.

**Diff**: removed the anti-rumination sentence from USER_INSTRUCTION_SUFFIX.

**Result**: accuracy **0.860**, cadence `0:292, 1:36, 2:120, 3:36, 4:11, 5:3, 7:2`. 58% 0-call.

**Status**: `discard` (conservative; within noise but no improvement).

Within 0076-recipe mean (0.865) noise. Cadence shifted slightly toward more tool use (cadence peak at "2 calls" rather than "1 call"). The anti-rumination sentence appears neutral under the 0076 recipe (vs load-bearing under 0062 recipe).

**Take-away**: under CoT prefix, the anti-rumination directive's effect is diluted. Not load-bearing here but also not actively hurting.

**Action**: revert to 0076 wording. Be conservative — don't bake in changes without variance corroboration.

**Best remains 0076-recipe at 2-sample mean 0.865, healthy cadence.**

**Next ideas**:
1. **Try "two calls is plenty" → "two or three calls"** — slight cadence-encouragement adjustment.
2. **Add system prompt with brief role identity** — already tried in 0072 (weakened cadence). Could revisit under 0076 recipe.
3. **Try removing "one or two calls is plenty"** — see if explicit cadence-numerals matter.

Picking #3: test cadence-numeral specificity.
