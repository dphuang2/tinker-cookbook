# 0016 system-prompt-directive

**Hypothesis**: 0011's tool description sits in the tool-spec block, which is structurally separate from any general system directive. Adding an explicit cadence instruction at the *top* of the system prompt — before the tools section — might bias cadence more strongly than the tool description alone.

**Diff**: sft_train.py
- Reverted tool description to 0011's "your own bookkeeping" wording.
- Added a `SYSTEM_PROMPT` constant: `"You are working through a math problem and should use the checkpoint tool every time you finish a sub-step, change approach, or make a key observation. ... do not skip it."`
- Passed `system_prompt=SYSTEM_PROMPT` to `create_conversation_prefix_with_tools` in both training (`record_to_datums`, `pure_math_record_to_datum`) and eval (`run_agent`).

**Training**: 142 steps, final NLL 0.238 (essentially the same as 0011's 0.239).

**Result**: accuracy **0.708**, cadence `0:155, 1:215, 2:119, 3:11`. Worse than 0011 by 3.2pp.

| Run | accuracy | system prompt | tool description |
|-----|----------|---------------|------------------|
| 0011 | **0.740** | empty | mid framing ("for your bookkeeping") |
| 0015 | 0.722 | empty | strong framing ("private scratchpad") |
| 0016 | 0.708 | mandatory directive | mid framing |

**Status**: `discard`. Stacking prompt-side biases on top of 0011 makes things worse. Pattern: any prompt change beyond 0011's exact wording regresses.

**Take-away**: prompt engineering has hit a local maximum at 0011. The model is highly sensitive to prompt phrasing and small changes both up and down don't help.

**Pattern after 16 experiments**:
- Single winning change: drop `reasoning` arg (0002, +2.8pp over v3).
- Marginal winning change: rename tool + mid-framing (0011, +0.4pp over 0002).
- All hyperparam knobs (LR, lora_rank, epochs): dead or worse.
- All data composition knobs (mix, filter, subsample): worse or cadence-collapse.
- All prompt knobs beyond 0011: worse.

The 0.740 ceiling appears robust. The remaining 14pp gap to no-SFT baseline (0.880) is likely structural — caused by the SFT process itself (over-fitting on a niche format) rather than any specific knob.

**Next ideas (most radical untried)**:
1. **Larger dataset (~5000 records via more raw traces)** — tests whether the ceiling is data-bound. Expensive (~25 min for extra sampling + teacher).
2. **Two-stage SFT** — first train on plain math (preserve reasoning), then a quick second pass on tool data (learn format). Tests whether order matters.
3. **Mask loss on `<think>` tokens** — train only on the tool_call + final answer tokens. Surgical SFT that explicitly preserves base reasoning.

Picking #3 (mask loss on thinking): structurally distinct from all prior experiments. Tests the hypothesis that training on the thinking tokens is what's eroding reasoning — surgically excluding them might fix it.

This requires modifying how loss weights are computed. Need to override `conversation_to_datum` or write a custom weight-zeroing pass after it. Will be in `record_to_datums` — after getting the datum, zero out weights on the thinking tokens.

Actually that's harder to do cleanly without parsing tokens. Let me think of an easier version of #3:

Easier: use `LAST_ASSISTANT_MESSAGE` for tool-call turns but train ONLY on the tool_call portion, by setting `train_on_what` differently. But this isn't supported.

Alternative: instead of #3, try #2 (two-stage SFT). But that adds infrastructure.

OK alternatively #1 (larger dataset) - just sample more raw traces.

Let me pick #1 as the next concrete step. It's a known well-understood operation: extend `sample_deepmath_train.py` to cover indices 3000-5500 (2500 more), then rerun teacher_rewrite on all 5000 traces, then train.
