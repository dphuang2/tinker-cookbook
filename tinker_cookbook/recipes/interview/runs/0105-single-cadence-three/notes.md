# 0105 single-cadence-three

**Hypothesis**: "two or three calls" gives the model a range; "three calls" forces a single anchor. May tighten the cadence shape further.

**Result**: accuracy **0.890**, cadence `0:281, 1:2, 2:1, 3:187, 4:20, 5:4, 6:4, 8:1`. 56.2% 0-call. Cadence is now cleanly bimodal: skip or exactly 3.

**Status**: `keep` (single-sample +1pp above no-tool baseline; needs corroboration).

| Run | accuracy | cadence shape |
|-----|----------|---------------|
| 0100 mean (n=4) | 0.877 | peak at 3 calls (~27%), some spread |
| 0105 (n=1) | **0.890** | peak at 3 calls (37.4%), very tight |

Cadence peak more concentrated (187 vs 122-159 in prior 0100 samples). 0-call rate similar.

**Hypothesis on mechanism**: a single number ("three") gives the model a precise target. The range ("two or three") let the model interpret either; the single anchor forces commitment. Tighter cadence shape may correlate with cleaner reasoning, hence higher accuracy.

**Action**: keep, variance rerun.

**Best**: 0105-recipe (tentative).
