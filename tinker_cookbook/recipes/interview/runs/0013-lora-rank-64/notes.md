# 0013 lora-rank-64

**Hypothesis**: 0011 (rank 32) beats 0005 (rank 8) and 0012 (rank 16). If rank is monotonic, rank 64 should be even better.

**Diff**: `LORA_RANK = 32 → 64`. Everything else same as 0011.

**Training**: 142 steps, final NLL 0.239 (same as 0011 — rank ≥ 32 doesn't move loss).

**Result**: accuracy **0.732**, cadence `0:142, 1:222, 2:128, 3:8`. **Worse than 0011** by 0.8pp.

| Run | accuracy | lora_rank |
|-----|----------|-----------|
| 0005 | 0.728 | 8 |
| 0012 | 0.724 | 16 |
| **0011** | **0.740** | **32** |
| 0013 | 0.732 | 64 |

**Status**: `discard`. Rank curve is non-monotonic with a peak at 32. Larger rank doesn't help.

**Rank sweep done** — stay at 32.

**Pattern**: all 4 LoRA-rank values (8, 16, 32, 64) cluster in 0.724-0.740 range, with 0.736-0.740 the apparent ceiling on the message-only format. The rank knob accounts for ~1pp of variation, well within eval noise.

**Next ideas (untried)**:
1. **Lengthen training to 1.5 epochs** — 1 epoch gave 0.740, 2 epochs gave 0.704 (0008 overfit). Maybe 1.5 lands in between.
2. **Stronger "you don't talk to user" framing** in tool description.
3. **Higher MIN_TOTAL_THINKING_CHARS** (8000-12000) — the 4000 filter was too gentle. A real filter that drops 25-50% may help cadence.
4. **System prompt addition** — beyond the tool spec, add an explicit instruction line about cadence.

Picking #3 (MIN_TOTAL_THINKING_CHARS=8000): structurally different from rank/LR/epoch knobs (which are all dead), and 0006 was a no-op because the threshold was too low.
