# 0065 cot-no-suppressor

**Hypothesis**: 0064's CoT prefix gave +0.6pp accuracy but collapsed cadence (95% 0-call). Was the collapse driven by the "Think step by step" prefix or by the strong "for simple problems, just think and answer directly" suppressor sentence? Test: keep CoT prefix, drop suppressor.

**Diff**: USER_INSTRUCTION_SUFFIX = `"Think step by step, then write your final answer in \boxed{} format. Don't think for too long unnecessarily, especially when you have a reasonable degree of confidence. The checkpoint tool is available for tracking progress on hard multi-step problems -- call it when you genuinely change approach or finish a substantial sub-task."`

**Result**: accuracy **0.862**, cadence `0:315, 1:131, 2:32, 3:18, 4:1, 7:1, 8:2`. 63% 0-call.

| Run | accuracy | 0-call % | description |
|-----|----------|----------|-------------|
| 0062 | 0.870 | 75.6% | no CoT, sparing directive (best non-degenerate) |
| 0064 | 0.876 | 95.4% | CoT + sparing directive (degenerate) |
| 0065 | **0.862** | **63%** | CoT + no suppressor (this) |

**Status**: `discard`. Accuracy -0.8pp vs 0062 (within noise but slightly worse). Cadence is *healthier* than 0062 (37% tool use vs 24%), but priority #1 is accuracy.

**Interpretation**: dropping the "for simple problems just think and answer directly" suppressor restored tool use cleanly, confirming that sentence was the cadence-killer in 0064. But the CoT prefix alone doesn't recover the full +0.6pp gain — the gain in 0064 came specifically from suppressing the tool on easy problems (paying near-zero tool-spec cost). When the tool is used on ~37% of problems, the tool-spec cost reasserts itself and we land slightly below 0062.

**Take-away**: the 0064 gain was structural, not a wording improvement. There's no free lunch — you either pay the tool-spec cost (and keep cadence) or suppress it (and lose cadence). 0062 sits at an apparently optimal mix.

**Action**: revert sft_train.py USER_INSTRUCTION_SUFFIX to 0062's wording (sparing directive). Keep eval at max_tokens=24576.

**Best remains 0062 at 0.870 with non-degenerate cadence (75.6% 0-call).**

**Next ideas (running thin)**:
1. **Reduce tool-spec verbosity** — shorter `description` field in PROGRESS_TOOL_SPEC. Less spec to "pay for" per call. May tilt the accuracy/cadence tradeoff favorably.
2. **Sample more variance on 0062** — confirm 0.870 isn't a high-variance peak; with healthy variance ~0.6-1.7pp, single-sample 0.870 may overstate.
3. **Try only the CoT prefix, no other changes** — i.e. add CoT to 0024-style sparing directive without other edits (this is what 0064 was, but with mt=8192 perhaps different).

Picking #1 (shorter tool spec description): reducing prompt overhead is a structural lever that should compound with high max_tokens.
