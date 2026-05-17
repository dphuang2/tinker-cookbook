# 0041 0036-rerun (variance check)

**Hypothesis**: T=0.6 with N=500 gives ~1.7pp std on accuracy. Was 0036's 0.810 vs 0024's 0.798 real or noise? Re-run 0036's config.

**Result**: accuracy **0.794** (vs 0036's 0.810). Cadence `0:483, 1:17` (same shape).

**Conclusion**: 0036 and 0024 are within noise. Two-run average for 0036 = (0.810 + 0.794) / 2 = 0.802. This is statistically indistinguishable from 0024's single-run 0.798.

**Calibration**: the eval has ~1.7pp std. Differences below ~3pp aren't statistically significant.

**Best (point estimate)**: 0036 average ~0.80.
**Best (statistically distinguishable from baselines)**: 0036 ≈ 0024 ≈ ~0.80.
**Tertiary goal trade-off**: 0036 has 3% tool use (degenerate); 0024 has 18% (healthy).

If the user prefers non-degenerate cadence, **0024 is the preferred recipe** — within noise of 0036 on accuracy, with much healthier tool use behavior.

If pure accuracy: 0036 by ~0.4pp average, but noise-bounded.

**Status**: analytical (variance check).
