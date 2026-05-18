# 0147 tiny-sft-20

**Hypothesis**: minimum-viable SFT — 20 records. 0028 with 100 records was catastrophic; 20 might be either too small to damage OR even more under-fit.

**Diff**: MAX_TOOL_RECORDS = 20 (was 0). Trained 1 batch (~20 records / batch_size=16 = 2 steps), 1 epoch.

**Result**: accuracy **0.870**, cadence `0:497, 3:2, 4:1`. **99.4% 0-call** — degenerate.

**Status**: `discard`.

| Run | training records | accuracy | tool use |
|-----|------------------|----------|----------|
| 0105 prompt-only | 0 | 0.876 (n=15) | ~50% |
| **0147 (20 records SFT)** | **20** | **0.870** | **0.6%** |
| 0028 (100 records SFT) | 100 | 0.482 | 100% degenerate |

**Critical finding**: even 20 SFT records is enough to **kill cadence**. Accuracy survives (within noise), but the model learns "this is a math problem, answer directly" from the small dataset, completely overriding the prompt's "three checkpoints typical" anchor.

**Take-away**: SFT signal dominates prompt signal even at 20 records. The teacher trajectories (Qwen3 raw thinking → boxed answer) don't include tool calls, so the model learns to skip them. To use SFT we'd need teacher trajectories WITH checkpoint calls (which is what 0011 attempted with mixed results).

**Per PROGRAM.md priorities**: (1) accuracy parity, (2) records WORSE (20 > 0), (3) cadence MUCH WORSE.

**Action**: revert MAX_TOOL_RECORDS = 0. Best remains 0105 prompt-only.
