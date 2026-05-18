# 0088 0087-variance-rerun

**Hypothesis**: corroborate 0087's 0.894 single-sample peak.

**Result**: accuracy **0.862**, cadence `0:192, 1:17, 2:11, 3:214, 4:43, 5:13, 6:5, 7:1, 8:2, 10:1, 24:1`. 38.4% 0-call (peak 42.8% at 3 calls).

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 |
| **0087** | **2** | **0.894, 0.862** | **0.878** |

The 0.894 from 0087 was a high-variance pull. 2-sample mean 0.878 is within ~0.4pp of 0080's 4-sample mean. Cadence shape matches expectation (peak at 3 calls, ~38% 0-call).

**Take-away**: 0087 wording is essentially equivalent to 0080 in expectation; cleaner wording doesn't actually improve. The 0.894 sample was variance.

**Action**: keep 0087 wording (cleaner is better all things equal). Need 3rd sample to tighten estimate.

**Best**: 0087-recipe (cleaner of two equivalent wordings) at 2-sample mean 0.878.
