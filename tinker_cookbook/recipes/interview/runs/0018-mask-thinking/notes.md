# 0018 mask-thinking

**Hypothesis**: training on thinking tokens damages base reasoning. Mask the loss on `<think>...</think>` content so the model trains only on:
- the tool_call args at the end of each turn
- the final boxed answer

Should preserve reasoning quality while still teaching the format.

**Diff**: sft_train.py
- Added `THINK_OPEN_TOKEN = 151667`, `THINK_CLOSE_TOKEN = 151668`, `MASK_THINKING_LOSS = True`.
- Added `_mask_thinking_weights(datum)` helper: walks the input token sequence, tracks whether inside a `<think>...</think>` span, zeros out loss weights at positions that predict thinking-block content.
- Wrapped both `conversation_to_datum` calls in `record_to_datums` with `_mask_thinking_weights(...)`.
- Reverted `sample_deepmath_train.py` END_INDEX 5500 → 3000 (constant only — data file is still the larger 4787-record one from 0017).

Smoke test confirmed: loss tokens per datum dropped from ~1000 to 50-650 (just the tool_call args / final answer).

**Training**: 287 steps × batch 16. Final NLL 0.225.

**Result**: accuracy **0.658** — *worst result in 18 experiments*. Cadence exploded: `0:4, 2:2, 3:149, 4:256, 5:66, 6:10, 7:3, 8:10`. Median tool calls per problem = 4, max = 8 (the eval cap). 99% of problems emit at least 3 calls.

| Run | accuracy | cadence summary | training_records |
|-----|----------|-----------------|------------------|
| 0011 | **0.740** | 1-call median | 2302 |
| 0017 | 0.738 | 1-call median | 4787 |
| **0018** | **0.658** | **4-call median** | 4687 (mask-thinking) |

**Diagnosis**: masking thinking-token loss meant the model never trained on the "thinking → tool_call transition" pattern. Without that gradient signal, the model has no learned policy for *when to stop thinking and emit a call*. It just emits calls reflexively, often hitting the 8-turn agent cap before reaching a final answer.

**Status**: `discard`. The mask was structurally wrong — the thinking tokens are load-bearing for learning the cadence boundary, not just for reproducing thinking content.

**Take-away**: this rules out the "training-on-thinking erodes reasoning" hypothesis. The thinking tokens are necessary signal. The 0.74 ceiling cannot be cracked by surgical loss masking.

**Pattern after 18 experiments**:
- Best: 0011 (0.740) with reasoning-arg-dropped + checkpoint-rename + mid-framing description.
- 14pp gap to no-SFT baseline persists.
- All structural changes (LR, rank, epochs, data composition, system prompt, dataset size, loss mask) have failed.

**Remaining untried (radical)**:
1. **Two-stage SFT** — first epoch on plain math (no tools, no SFT damage to reasoning), second epoch on the tool-call data.
2. **Different base model checkpoint** — e.g. evaluate an earlier 0011-style training checkpoint (step 80) instead of `final`.
3. **DPO** — much more complex.

Picking #2 (early-stop eval): cheapest test (just changes which sampler the eval uses), and the v3-era notes mentioned this idea. The intermediate checkpoint might have learned the format without fully eroding reasoning.

To do this, modify eval_deepmath_agent.py to read a specific checkpoint name (e.g. step 80) from checkpoints.jsonl. Then re-eval the *existing* 0011 sampler path that was saved at step 80 — wait, 0011's checkpoint history only saved step 20, 40, ..., 140 + final. Let me look.

Actually it's simpler: re-train 0011 (revert to MASK_THINKING_LOSS=False, smaller dataset) and snapshot at step 80 by setting `eval_every=80`. Then eval the step-80 checkpoint.

Or simpler still: just modify eval_deepmath_agent.py's `find_final_sampler_path` to pick a different row. Use 0011's saved checkpoints (they're in the 0011 run dir).

Looking at 0011's run dir: `tinker_cookbook/recipes/interview/runs/0011-rename-checkpoint/checkpoints.jsonl`. That has all the periodic checkpoints. Let me grep.
