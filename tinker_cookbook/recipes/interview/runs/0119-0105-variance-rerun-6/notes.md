# 0119 0105-variance-rerun-6

**Result**: accuracy **0.864**.

**Status**: `variance`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| **0105** | **6** | 0.890, 0.878, 0.866, 0.880, 0.858, 0.864 | **0.8727** | **0.011** |

6-sample mean 0.8727 ± 1.1pp std. Converged. Recipe at parity with 0100 (0.8775 mean, n=8) and ~0.7pp below no-tool baseline 0.880.

**Definitive picture**: prompt-only recipe gives 0.873 ± 0.011 on DeepMath 500. Statistically indistinguishable from no-tool baseline (0.880) within ~1σ. Healthy cadence (50% 0-call, 40% use exactly 3 calls, peak strongly bimodal).
