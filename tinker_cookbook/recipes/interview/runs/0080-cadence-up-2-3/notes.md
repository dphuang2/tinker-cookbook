# 0080 cadence-up-2-3

**Hypothesis**: 0076 with "one or two calls is plenty" yielded cadence peak at 1 call (91/500 problems). Bumping to "two or three calls is typical" should shift the cadence distribution upward toward more tool use, while accuracy holds.

**Diff**: USER_INSTRUCTION_SUFFIX numeral change: "one or two calls is plenty for a typical problem" → "two or three calls is typical for a multi-step problem".

**Result**: accuracy **0.876**, cadence `0:257, 1:25, 2:15, 3:129, 4:55, 5:6, 6:5, 7:1, 8:4, 9:2, 14:1`. 51% 0-call.

**Status**: `keep` (tentative — single-sample at high tail of variance, but cadence shape clearly improved).

| Run | accuracy | 0-call % | peak | description |
|-----|----------|----------|------|-------------|
| 0062-recipe | 0.863 (n=3) | 78% | 0 | base |
| 0076-recipe | 0.861 (n=3) | 61% | 0 | CoT + "1-2 calls" |
| 0080 | 0.876 (n=1) | 51% | **3 calls** | CoT + "2-3 calls" |

**Cadence distribution shifted**:
- 0076 (n=3): cadence-1 peak ~17%, cadence-2 ~15%.
- 0080: cadence-3 peak 26% (vs 6% in 0076), cadence-1 only 5% (vs 17%).

The explicit "2-3 calls is typical" successfully retargeted the model's tool-use pattern. The shape is more bimodal: 51% don't call at all (presumably easy problems), 49% make 2-4 calls (multi-step).

**Take-away**: numerical anchor in the directive is a precise lever for cadence shape, without obvious accuracy cost.

**Action**: keep 0080 as new recipe. Variance-rerun next to confirm 0.876 is sustainable.

**Best is now 0080-recipe**.

**Next**: variance rerun.
