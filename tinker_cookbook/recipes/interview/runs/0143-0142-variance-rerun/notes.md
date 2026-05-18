# 0143 0142-variance-rerun

**Result**: accuracy **0.866**, cadence: 57% skip, 40% exactly 3 calls.

**Status**: `variance`. 0142-recipe 2-sample mean **0.873** — at parity with 0105.

**Finding**: under the 0105 recipe (with "three checkpoints" anchor), the "Don't think for too long" anti-rumination sentence is NOT load-bearing. It was originally added when max_tokens=8192 was binding; with max_tokens=24576 and the strong cadence anchor, it's redundant.

**Action**: keep simpler recipe (no anti-rumination sentence).

**New final recipe**:
```
SYSTEM_PROMPT = "You are solving competition math problems. You have access to a checkpoint tool for tracking progress."

USER_INSTRUCTION_SUFFIX = " Think step by step, then write your final answer in \\boxed{} format. Use the checkpoint tool when it helps you organize hard multi-step problems -- three checkpoints is typical for a multi-step problem."
```

Cleaner, shorter, at parity.
