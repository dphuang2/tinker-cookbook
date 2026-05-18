# 0089 0087-variance-rerun-3

**Result**: accuracy **0.860**, cadence `0:198, 1:14, 2:17, 3:189, 4:55, 5:13, 6:10, 7:3, 9:1`. 39.6% 0-call.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 | 0.007 |
| **0087** | **3** | **0.894, 0.862, 0.860** | **0.872** | **0.019** |

**Status**: `variance`. 0087 has higher single-sample peak (0.894) but wider variance (0.019 vs 0080's 0.007). Means within 0.25pp.

**Take-away**: cleaner wording (0087) is statistically equivalent to 0080. Higher variance suggests the model is more "uncertain" about when to call the tool without the explicit "when it helps you" hedge. The hedge in 0080 may stabilize behavior.

**Trade-off**:
- 0087 wording: shorter prompt, equivalent mean, peak 0.894, wider variance.
- 0080 wording: more verbose, equivalent mean, tighter variance.

**Action**: keep 0087 (shorter is better default). For maximally robust deployment, revert to 0080.

**Best**: 0087-recipe at 3-sample mean 0.872. Effective ceiling under prompt-only at ~0.87-0.88.

**Pattern after 27 post-0062 experiments**: ceiling reached. Mean accuracy 0.872-0.8745 across two variants. No-tool baseline 0.880. Gap is ~0.5-1.0pp, well within noise.

**Next ideas**:
1. **Declare current as final**.
2. **Compare to no-tool baseline directly with new variance check** — confirm 0.880 holds at multiple samples.
3. **Try a truly orthogonal direction**: e.g., a different tool name semantic.
4. **Try mini-RL** — out of scope/expensive.

Picking #2: variance-check no-tool baseline (NO_TOOL=1) to confirm 0.880 isn't itself a single-sample artifact. Critical for honest comparison.
