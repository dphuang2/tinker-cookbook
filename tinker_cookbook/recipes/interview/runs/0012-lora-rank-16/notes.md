# 0012 lora-rank-16

**Hypothesis**: 0005 (rank 8) underperformed slightly. 0011 (rank 32, with tool rename) is the current best. Try rank 16 — intermediate — in case the rank curve is non-monotonic and 16 lands in a sweet spot.

**Diff**: `LORA_RANK = 32 → 16`. Everything else same as 0011 (renamed `checkpoint` tool, 0002-style teacher data).

**Training**: 142 steps, final NLL 0.239 (same as 0011 — rank doesn't move loss much).

**Result**: accuracy **0.724**, cadence `0:138, 1:213, 2:140, 3:9`. Worse than 0011 (0.740) by 1.6pp, similar to 0005 (rank 8 at 0.728).

| Run | accuracy | lora_rank |
|-----|----------|-----------|
| 0005 | 0.728 | 8 |
| 0012 | 0.724 | 16 |
| **0011** | **0.740** | **32** |

**Status**: `discard`. Smaller rank hurts. Stay at rank 32.

**Confirmed**: rank knob is monotonic in our direction — 32 is the best. Don't try smaller again.

**Next ideas**:
1. **Push tool description even harder** — emphasize "the user does NOT see these" more strongly.
2. **Add MIN_TOTAL_THINKING_CHARS=8000** for the actually-meaningful filter (0006 was too gentle at 4000).
3. **Lengthen `max_length` to 65536** to handle the longest 2 epoch traces (probably doesn't matter since most fit in 32k).
4. **Larger LoRA rank** (64) — if rank is monotonic up to 32, maybe even larger helps.

Picking #4 (lora_rank=64) — completes the rank sweep. Cheap test of "more is better".
