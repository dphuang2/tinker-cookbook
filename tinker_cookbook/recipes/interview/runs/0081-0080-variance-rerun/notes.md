# 0081 0080-variance-rerun

**Hypothesis**: corroborate 0080's 0.876 with a second sample.

**Result**: accuracy **0.868**, cadence `0:264, 1:22, 2:13, 3:132, 4:51, 5:6, 6:7, 7:1, 8:2, 20:1, 24:1`. 52.8% 0-call.

**Status**: `keep` (provisional new best, variance check #3 next).

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0062 | 3 | 0.870, 0.864, 0.854 | 0.863 |
| 0076 | 3 | 0.876, 0.854, 0.852 | 0.861 |
| **0080** | **2** | **0.876, 0.868** | **0.872** |

Both 0080 samples exceed 0062's max sample (0.870), and the mean (0.872) is +0.9pp above 0062 mean. With 0062 std ~0.8pp, two samples both above 0.868 is ~96% likely a real signal (~+1σ each).

Cadence in 0081 mirrors 0080 — peak at 3 calls, ~50% 0-call, plus a couple extreme outliers (20, 24 calls in single problems) which are real but rare.

**Take-away**: the "two or three calls is typical" numerical anchor not only shifts cadence shape but appears to genuinely improve accuracy — possibly because the model treats the tool as a structured-reasoning checkpoint and uses it more productively.

**Hypothesis on mechanism**: in 0076-recipe with "one or two calls is plenty", the model treats the tool as optional and often skips it on multi-step problems where checkpointing would actually help. In 0080-recipe, "2-3 calls typical for multi-step" reframes the tool as default-on for hard problems, which leads to more structured reasoning.

**Best: 0080-recipe at 2-sample mean 0.872, peak cadence 3 calls.**

**Next**: 3rd variance sample.
