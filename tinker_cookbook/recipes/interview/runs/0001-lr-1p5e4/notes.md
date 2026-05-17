# 0001 lr-1p5e4

**Hypothesis**: v3's `get_lr` default (~5e-4) was over-aggressive for a format-imitation task and is the main cause of the 14pp regression. Lowering LR to 1.5e-4 should preserve more of the base reasoning capability.

**Diff**: sft_train.py line ~234, `"learning_rate": hyperparam_utils.get_lr(MODEL_NAME, is_lora=True)` → `"learning_rate": 1.5e-4`.

**Training**: 142 steps × batch 16 over 2301 records (post-flatmap ~9000 datums). Final train NLL 0.107. Mid-step LR around 1.5e-4 (linear schedule).

**Result**: accuracy 0.708 (identical to v3 baseline). Cadence shifted toward more 0-call problems: `0:377, 1:98, 2:21, 3:4` (vs v3 `0:342, 1:137, 2:18, 3:2, 4:1`). So the lower LR didn't damage reasoning *less*, it just shifted the cadence distribution slightly.

**Status**: `discard`. LR was not the main cause of the regression — the loss curves and final accuracy are basically identical to v3.

**Next idea**: since LR isn't the bottleneck, try a more structural change. Candidates:
- **Smaller `lora_rank`** (8 from 32) — less capacity = less corruption.
- **Drop the duplicated `reasoning` arg** — back to v1's `message`-only format. v1 hit 0.736 with that. The duplication may be teaching the model to be over-confident about emitting verbose tool calls.
- **Mix in pure-math no-tool data** — half the batch plain `<think>` + boxed answer, half with tool calls.

Picking "drop reasoning arg" for 0002: it directly tests whether the duplication is hurting (v1 hit 0.736, v3 hit 0.708 — so the duplication added ~3pp damage). If 0002 reproduces v1's 0.736 or better, we have confirmation. Layer LR change on top if it works.
