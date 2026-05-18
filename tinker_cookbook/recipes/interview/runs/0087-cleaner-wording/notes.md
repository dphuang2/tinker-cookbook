# 0087 cleaner-wording

**Hypothesis**: 0080 wording had redundancy ("multi-step problem" appears twice). Tighten: "Use the checkpoint tool to organize hard multi-step problems -- two or three calls is typical." Drops "when it helps you" hedge.

**Diff**:
- before: "Use the checkpoint tool when it helps you organize hard multi-step problems -- two or three calls is typical for a multi-step problem."
- after: "Use the checkpoint tool to organize hard multi-step problems -- two or three calls is typical."

**Result**: accuracy **0.894**, cadence `0:189, 1:17, 2:13, 3:204, 4:46, 5:9, 6:12, 7:2, 8:2, 9:3, 10:1, 16:1, 21:1`. 37.8% 0-call.

**Status**: `keep` (tentative new best; needs variance corroboration).

| Recipe | n | mean | this sample |
|--------|---|------|-------------|
| 0062 | 3 | 0.863 | — |
| 0080 | 4 | 0.8745 | — |
| **0087** | **1** | — | **0.894** |
| no-tool baseline | - | 0.880 | — |

0.894 is +1.4pp above no-tool baseline and +2σ above 0080 mean (0.8745 ± 0.7pp). Could be variance, but worth pursuing.

**Cadence change**: 37.8% 0-call vs 0080's ~52% — meaningful shift. Peak still at 3 calls but now 40.8% vs 0080's 26.4%. The model is using the tool on ~62% of problems vs 0080's ~48%.

**Hypothesis on mechanism**: dropping "when it helps you" (a hedge) makes the directive less ambiguous about when to use the tool. The model treats the tool as default-on more often, but unlike 0085's imperative which over-saturated, the "to organize hard multi-step" qualifier still scopes it correctly. Sweet spot may be slightly different from 0080.

**Action**: keep, variance rerun next.
