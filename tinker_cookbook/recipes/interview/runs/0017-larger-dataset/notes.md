# 0017 larger-dataset

**Hypothesis**: 0011 hit 0.740 with 2302 records. Maybe the ceiling is data-bound. Doubling to ~4700 records should push accuracy if so.

**Diff**:
- sample_deepmath_train.py: END_INDEX 3000 → 5500 (samples ~5000 raw traces, was ~2500).
- Re-ran sample_deepmath_train.py: 5000 raw traces, 4794 clean.
- Re-ran teacher_rewrite.py: 4787 SFT records (vs 2402). Cadence shape unchanged: 65% 3-call.
- Reverted SYSTEM_PROMPT to "" (0016 discard).

**Training**: 290 steps × batch 16 on 4687 train records (~2x previous). Final NLL 0.252.

**Result**: accuracy **0.738**, cadence `0:174, 1:219, 2:95, 3:12`. Within noise of 0011's 0.740.

| Run | accuracy | training_records |
|-----|----------|------------------|
| 0011 | **0.740** | 2302 |
| 0017 | 0.738 | 4687 |

**Status**: `discard`. 2x data didn't move accuracy. The 0.74 ceiling is **structural**, not data-bound.

This is a meaningful negative result: we don't need to invest in more data. The bottleneck is elsewhere (SFT process itself, format choice, the LoRA adapter, or the fundamental difficulty of teaching this behavior to Qwen3 without damaging its reasoning).

For the secondary "data efficiency" goal: 0011 (2302 records) is already as good as 0017 (4687 records), so 2302 is the better recipe on records-per-accuracy.

**Take-away after 17 experiments**: the recipe at 0.740 with 2302 records is the durable best. Closing the remaining 14pp gap to no-SFT baseline (0.880) probably requires a methodological change (RL, DPO, surgical loss masking) rather than continued knob-tuning.

**Next ideas (untried, methodological)**:
1. **Mask loss on `<think>` tokens** — surgical SFT where only the tool_call + final-answer tokens contribute to loss. Preserves base reasoning by not training on thinking. Requires custom weight computation after `conversation_to_datum`.
2. **DPO on preference pairs** — collect (correct, incorrect) pairs from 0011's outputs on a held-out dataset slice and DPO. More complex.
3. **Two-stage SFT** — train briefly on plain-math, then on tool-call data. Different curriculum.

Picking #1 (mask thinking tokens). It's the most surgical: tests the hypothesis that training-on-thinking is what's eroding reasoning. If true, masking out thinking should let the model keep its reasoning ability AND learn the tool format.
