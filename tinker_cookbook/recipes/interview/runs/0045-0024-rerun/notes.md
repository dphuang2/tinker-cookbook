# 0045 0024-rerun (variance)

**Result**: accuracy **0.804**, cadence `0:413, 1:49, 2:23, ...`. 83% 0-call.

**Two samples of 0024 config**:
| Run | Accuracy |
|-----|----------|
| 0024 | 0.798 |
| 0045 | 0.804 |
| **mean** | **0.801** |

**Three samples of 0036 config**:
| Run | Accuracy |
|-----|----------|
| 0036 | 0.810 |
| 0041 | 0.794 |
| 0044 | 0.790 |
| **mean** | **0.798** |

**Statistical comparison**: 0024 (n=2, mean 0.801) and 0036 (n=3, mean 0.798) are within noise. 0024 actually has a *slightly* higher point estimate but the difference is meaningless.

**Cadence (non-degenerate criterion)**:
- 0024: 17-18% tool-call rate (healthy)
- 0036: 3% tool-call rate (borderline degenerate)

**Final recipe**: **0024** wins on combined goals.
- Accuracy: ~0.80 (tied with 0036)
- Training records: 0
- Cadence: 18% (healthy non-degenerate)

**Status**: analytical (variance check).
