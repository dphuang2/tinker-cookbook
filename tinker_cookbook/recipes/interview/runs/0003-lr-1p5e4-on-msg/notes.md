# 0003 lr-1p5e4-on-msg

**Hypothesis**: 0001 showed LR 1.5e-4 had no effect when stacked on v3's `reasoning`-arg format. But on 0002's simpler `message`-only format, lower LR might help — the message-only loss is harder (0.27 NLL vs 0.11), so maybe the default LR is now over-fitting more aggressively.

**Diff**: sft_train.py LR change `get_lr(...)` → `1.5e-4` (3.3x smaller).

**Training**: 142 steps. Final train NLL 0.278 (≈ same as 0002 at 0.270).

**Result**: accuracy **0.718**. Worse than 0002 (0.736), barely above v3 (0.708). Cadence: `0:287, 1:175, 2:35, 3:3` — shifted toward fewer tool calls vs 0002.

| Run | accuracy | nll | LR | format |
|-----|----------|------|------|---------|
| v3 | 0.708 | 0.109 | get_lr (~5e-4) | summary+reasoning |
| 0001 | 0.708 | 0.107 | 1.5e-4 | summary+reasoning |
| 0002 | **0.736** | 0.270 | get_lr | **message-only** |
| 0003 | 0.718 | 0.278 | 1.5e-4 | message-only |

**Status**: `discard`. LR 1.5e-4 underperforms get_lr regardless of format. The previous 0001 "LR had no effect" was a coincidence — actually 1.5e-4 is just *worse*, but at the v3 format the noise hid it.

**Conclusion**: keep LR at `get_lr`. Don't try lower LR again unless we have a different reason.

**Next idea**: bigger structural changes now that the LR knob is dead.

1. **Mix in pure-math no-tool data** — 0002 has 28% of problems emitting 0 tool calls. Those should match baseline accuracy (~0.880) but presumably they're not. Mixing in plain Qwen3 traces (no tool calls) as part of the training data should help preserve the "just answer when confident" behavior.
2. **Smaller `lora_rank`** (8 from 32) — less capacity to corrupt baseline.
3. **Subsample training data** (e.g. 800 records from 2301) — directly addresses data-efficiency goal.

Picking #1 (mix in plain data) — it directly addresses the most likely failure mode (baseline reasoning erosion on no-tool problems) AND is structurally distinct from the LR knob we just ruled out.
