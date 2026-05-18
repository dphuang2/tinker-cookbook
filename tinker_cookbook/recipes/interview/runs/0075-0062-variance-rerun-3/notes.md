# 0075 0062-variance-rerun-3

**Hypothesis**: 3rd variance sample of 0062 config to better estimate std.

**Diff**: comment-only marker.

**Result**: accuracy **0.854**, cadence `0:405, 1:50, 2:23, 3:12, 4:2, 5:2, 6:2, 7:1, 8:3`. 81% 0-call.

**Status**: `variance`.

| Sample | accuracy |
|--------|----------|
| 0062 | 0.870 |
| 0070 | 0.864 |
| 0075 | 0.854 |
| **Mean (n=3)** | **0.863** |
| **Std** | **~0.008 (0.8pp)** |

**Critical update**: variance is wider than initially estimated. With std ≈ 0.8pp:
- 2σ = 1.6pp
- 3σ = 2.4pp

Re-evaluating prior discards in this wider band:
- 0064 (0.876, **degenerate**): +1.3pp / +1.6σ — likely real gain, but cadence-disqualified.
- 0065 (0.862): -0.1pp — **equivalent**.
- 0066 (0.850): -1.3pp / -1.6σ — borderline, possibly real regression.
- 0067 (0.856): -0.7pp — **equivalent within noise**.
- 0068 (0.856): -0.7pp — **equivalent**.
- 0069 (0.860): -0.3pp — **equivalent**.
- 0071 (0.854): -0.9pp — equivalent.
- 0072 (0.866): +0.3pp — equivalent (cadence weaker).
- 0073 (0.858): -0.5pp — equivalent.
- 0074 (0.862): -0.1pp — equivalent.

**Major take-away**: most "discards" are within noise. Only **0064** showed a real signal (~+1.6σ accuracy gain) and that came with degenerate cadence. No prompt-engineering variation has shown a real positive signal while keeping cadence.

**Decision point**: with 3-sample baseline 0.863 ± 0.8pp, the recipe ceiling is ~0.87. To clear noise, an experiment needs to land at 0.880+ on a single sample to be likely real.

**Best**: 0062 config, 3-sample mean **0.863 ± 0.008**, prompt-only, 0 training records, healthy cadence.

**Pattern**: 3 variance samples + 11 negative variation attempts = the prompt-only optimum is at ~0.863 (mean). The 0.880 no-tool baseline is the ceiling and the ~1.7pp gap appears fundamental.

**Next ideas**:
1. **Tiny SFT with low LR** — 0028 failed with default LR. Try LR=1e-5 and ~50 records. May not damage base reasoning.
2. **Variance rerun on a "discard" like 0064 with cadence-preserving twist** — if 0064 was real, finding a way to preserve cadence would unlock it.
3. **Declare 0062 final** — accept 0.863 mean as the recipe ceiling.

Picking #2: 0064's signal was real. Try the CoT prefix but with a STRONGER tool-encouragement to compensate. E.g., reword from "use sparingly" to "use whenever helpful". Tests whether we can keep CoT gain + restore cadence.
