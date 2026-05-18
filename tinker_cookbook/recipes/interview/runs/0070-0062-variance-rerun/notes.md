# 0070 0062-variance-rerun

**Hypothesis**: 0062's single-sample 0.870 needs variance corroboration. Re-run exact 0062 config to estimate true mean and std.

**Diff**: comment-only marker; no functional change. Config identical to 0062.

**Result**: accuracy **0.864**, cadence `0:395, 1:65, 2:26, 3:6, 4:4, 5:1, 6:2, 8:1`. 79% 0-call.

| Run | accuracy | 0-call % |
|-----|----------|----------|
| 0062 (original) | 0.870 | 75.6% |
| 0070 (rerun) | 0.864 | 79.0% |
| **2-sample mean** | **0.867** | 77.3% |

Std estimate (n=2): ~0.004 / sqrt(2) → single-sample std ≈ 0.003 (0.3pp). Tight.

**Status**: `variance` — corroborates 0062 as true baseline.

**Implications for prior discards**:
- 0064 (0.876, cadence degenerate): genuinely +1pp but cadence-disqualified.
- 0065 (0.862): within ~0.5pp of mean — possibly within noise.
- 0066 (0.850): -1.7pp — genuinely worse (>3σ).
- 0067 (0.856): -1.1pp — likely worse (~3σ).
- 0068 (0.856): -1.1pp — likely worse.
- 0069 (0.860): -0.7pp — within ~2σ, possibly noise.

**Bottom line**: 0062 baseline is ~0.866 ± 0.3pp. Several recent discards are genuine regressions. 0065 and 0069 may have been near-equivalent.

**Best remains 0062-class config at 0.866 (2-sample).**

**Next ideas**:
1. **Try non-tool-related changes** — e.g. different temperature (0.4 vs 0.6) at the eval level. (Caveat: temp is FIXED per PROGRAM.md.)
2. **Try a different USER_INSTRUCTION_SUFFIX ordering** — put tool guidance first, boxed-answer last.
3. **System prompt with terse reasoning rubric** — separate channel.

Picking #2: simple ordering swap. Tests whether front-loading the tool guidance changes cadence/accuracy.
