# 0132 tool-ack-turn-count

**Hypothesis**: tool ack content `"ok"` → `f"ok (checkpoint {N} of 8)"`. Explicit progress meta-state may help pacing.

**Result**: accuracy **0.890**, cadence: 54% skip, 37.6% exactly 3 calls. Single sample above no-tool baseline 0.880.

**Status**: `keep` (tentative, +1.7pp vs 0105 mean 0.8732; needs corroboration).

**Hypothesis on mechanism**: showing turn count in ack ("checkpoint 1 of 8", "checkpoint 2 of 8") gives the model a sense of how much budget remains, potentially helping it allocate reasoning effort better.

**Note**: 0118 tested "continue" ack (parity), 0074 tested echoing summary (parity). Both null. Adding *numeric progress* meta is a different signal.

**Action**: keep, variance rerun.
