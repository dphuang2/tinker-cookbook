# 0106 0105-variance-rerun

**Result**: accuracy **0.878**, cadence `0:263, 1:5, 2:4, 3:189, 4:24, 5:5, 6:7, 12:3`. 52.6% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0100 (n=8) | 8 | various | 0.8775 |
| **0105 (n=2)** | 2 | 0.890, 0.878 | **0.884** |

Both 0105 samples ≥ 0100 mean. Cadence consistently bimodal — 187/189 problems use exactly 3 calls when they use the tool.

**Possibly real signal.** +0.7pp vs 0100 baseline. Need 3rd sample to confirm.

**Action**: keep, 3rd variance sample next.
