# 0055 0024-rerun-5

**Result**: acc **0.800**, cadence `0:402, 1:63, 2:20, 3:8, 4:2, 6:2, 8:3`. 80% 0-call.

**Five samples of 0024 config**:
| Run | Accuracy |
|-----|----------|
| 0024 | 0.798 |
| 0045 | 0.804 |
| 0049 | 0.790 |
| 0051 | 0.802 |
| 0055 | 0.800 |
| **mean** | **0.799** |
| std | 0.005 |

95% CI: **[0.789, 0.809]**. Very tight.

**0024 is the converged final recipe**: accuracy 0.799 ± 0.005, 0 training records, 17-21% tool-call rate.

Diminishing returns on further reps. The 0.80 ceiling for prompt-only is real and reproducible.
