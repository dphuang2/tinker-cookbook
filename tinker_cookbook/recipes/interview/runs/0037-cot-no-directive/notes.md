# 0037 cot-no-directive

**Hypothesis**: 0036's "Think step by step" might subsume the sparing-tool directive. Drop the directive, keep CoT.

**Diff**: removed the sparing-use directive from USER_INSTRUCTION_SUFFIX. Kept "Think step by step" + boxed-answer + don't-think-too-long.

**Result**: accuracy **0.784**, cadence `0:307, 1:159, 2:21, 3:11, 5:1, 6:1`. 61% 0-call (up from 0036's 97%, but tool use returned somewhat).

**Status**: `discard`. -2.6pp vs 0036. The sparing-use directive IS still load-bearing on top of CoT — they're complementary, not redundant.

**Trade-off observation**:
| Recipe | acc | 0-call % |
|--------|-----|----------|
| 0024 (no CoT, w/ directive) | 0.798 | 82% |
| **0036 (CoT + directive)** | **0.810** | 97% |
| 0037 (CoT, no directive) | 0.784 | 61% |

Adding CoT to the directive: +1.2pp, but cadence collapses to 2.6%.
Removing the directive from CoT: more tool use (39% calls), but accuracy drops 2.6pp.

So the two pieces interact: CoT improves accuracy but pushes the model to skip the tool; the directive further suppresses tool use AND adds accuracy. Together they push the model into the "use tool only when essential" regime where both effects compound.

**Best remains 0036 at 0.810** (accuracy) but with the cadence caveat. If non-degenerate cadence is required, 0024 at 0.798 with 82% 0-call (= 18% tool use) is preferable.

**Next ideas**:
1. **Find middle ground**: CoT + a milder directive — keep some tool use (~10-20%) while preserving most of the CoT gain.
2. Investigate whether the cadence collapse in 0036 is from the directive or the CoT itself — was already partly answered here (CoT alone gives 39% tool use, which is between 0024's 18% and 0036's 2.6%).

Picking #1: try CoT + a *positive* tool framing ("use the checkpoint tool when you find it useful") instead of suppressive ("use sparingly"). Maybe accuracy holds.
