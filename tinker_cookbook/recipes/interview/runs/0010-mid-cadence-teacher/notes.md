# 0010 mid-cadence-teacher

**Hypothesis**: 0002 had teacher data biased toward 3 calls (over-use); 0009 had teacher data with 30% no-calls (eval cadence collapsed to 0). The sweet spot might be a teacher that *always* uses tools but only 1-2 times per record. This should bias the model toward "use the tool" without over-fitting to 3-call pattern.

**Diff**:
- teacher_rewrite.py: prompt rewritten to "ALWAYS emit at least 1 split, mostly 1 or 2, never 0, never 3+".
- Regenerated `/tmp/tinker-examples/interview/sft_dataset.json` (2402 records).

**New training-data cadence**:
| 1 call | 2 calls | 0 calls | 3+ |
|--------|---------|---------|-----|
| 195 (8%) | 2207 (92%) | 0 | 0 |

Kimi gravitated to 2 calls. No 0-call or 3+ records.

**Training**: 142 steps, final NLL 0.243.

**Result**: accuracy **0.702**, cadence `0:218, 1:250, 2:32`. The eval cadence is dramatically different from training — 44% emit 0 calls even though training has 0 such records.

| Run | acc | training mode | eval cadence (0-call %) |
|-----|-----|---------------|-------------------------|
| 0002 | **0.736** | mostly 3-call | 28% |
| 0009 | 0.716 | 31% 0-call | 96% |
| 0010 | 0.702 | 0% 0-call | 44% |

**Status**: `discard`. Even with 0% 0-call training data, the model drifts to 44% 0-call at eval. The model has a strong learned default toward no-calls that LoRA SFT can shift only partially.

**Major insight**: training cadence ≠ eval cadence by a wide margin. The model's pretrained "just answer" bias breaks through SFT regardless of how the training distribution is tuned. The reason 0002 holds the lead (0.736) isn't because its cadence is right — it's because its tool calls *carry less harmful supervision per turn* (no duplicated reasoning arg).

**Lessons after 10 experiments**:
- Cadence engineering via teacher prompt: limited control. Eval cadence drifts toward 0-call.
- The "drop reasoning arg" change (0002, +2.8pp) remains the only successful improvement.
- Hyperparam knobs: dead. Data composition knobs: brittle.
- The ceiling of 0.736 may be near the cap of what LoRA SFT can do on this dataset / format.

**Next ideas (more orthogonal)**:
1. **Tool name + description rewrite** — rename `progress_update` to `checkpoint` with description "for your own bookkeeping; the user does NOT see this". Reduces the model's instinct to suppress calls "because user doesn't need to hear it".
2. **Prompt-only baseline** — no SFT at all, just expose tool in prompt. Establishes the natural-cadence ceiling.
3. **Train on a different signal: only mass on the tool-call tokens, not the thinking** — surgical SFT that only teaches the format, leaving reasoning untouched. Different `train_on_what` logic.

Picking #1 (tool name + description rewrite). Format-level tweak, cheap to test, complements 0002's format win.
