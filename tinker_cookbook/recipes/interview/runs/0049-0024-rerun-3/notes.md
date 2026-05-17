# 0049 0024-rerun-3 (variance)

**Result**: acc **0.790**, cadence `0:407, 1:63, 2:21, 3:4, ...`. 81% 0-call.

**Three samples of 0024 config**:
| Run | Accuracy |
|-----|----------|
| 0024 | 0.798 |
| 0045 | 0.804 |
| 0049 | 0.790 |
| **mean** | **0.797** |
| std | 0.007 |

**Three samples of 0036 config** (CoT variant):
| Run | Accuracy |
|-----|----------|
| 0036 | 0.810 |
| 0041 | 0.794 |
| 0044 | 0.790 |
| **mean** | **0.798** |
| std | 0.011 |

**Definitive comparison**: 0024 and 0036 have indistinguishable means (~0.797-0.798). 0024 has lower std (0.007 vs 0.011).

**Final recipe = 0024**:
- Accuracy: 0.797 ± 0.007 (95% CI roughly [0.78, 0.81])
- Cadence: 17-19% tool-call rate (healthy, non-degenerate)
- Training records: 0

This is the converged optimum. After 49 experiments, no recipe statistically beats it within eval noise.
