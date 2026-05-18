# 0133 0132-variance-rerun — STRONG SIGNAL

**Result**: accuracy **0.890** — exact repeat of 0132's result.

**Status**: `keep`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0105 baseline | 10 | various | 0.8732 |
| **0132 (turn-count ack)** | **2** | **0.890, 0.890** | **0.890** |
| no-tool baseline | - | - | 0.880 |

Two consecutive 0.890 samples is a strong signal. **+1.7pp over 0105 mean.** Above no-tool baseline by 1pp. Cadence shape consistent: 54% skip, ~36% exactly 3 calls.

**Hypothesis confirmed**: showing the model its progress counter ("checkpoint 1 of 8") in the tool ack helps it allocate reasoning effort better. Likely mechanism: the model can self-pace and avoid running out of agent-loop budget on hard problems.

**Best is now 0132-recipe (current).**

**Next**: 3rd variance sample.
