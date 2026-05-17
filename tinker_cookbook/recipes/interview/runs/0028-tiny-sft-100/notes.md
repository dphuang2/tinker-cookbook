# 0028 tiny-sft-100

**Hypothesis**: tiny SFT (100 records) on top of 0024's prompt-only optimum might nudge the format learning without breaking base reasoning the way bigger SFT runs did.

**Diff**: MAX_TOOL_RECORDS = 100. Used 0024's USER_INSTRUCTION_SUFFIX. Trained on 100 records (post-flatmap ≈ 300 datums, ~18 steps).

**Result**: catastrophic — accuracy **0.482**, cadence `0:500` (100% emit 0 tool calls). The 100-record SFT damaged base reasoning AND collapsed tool use entirely.

| Run | accuracy | training_records | 0-call % |
|-----|----------|------------------|----------|
| **0024 (prompt-only)** | **0.798** | 0 | 82% |
| 0028 (tiny SFT) | **0.482** | 100 | 100% |

**Status**: `discard`. Strongly negative result.

**Diagnosis**: 18 training steps on a tiny dataset with the default LR caused the LoRA adapter to *over-adjust* on a few specific examples without consolidating. The result is a model that:
1. Has weights pulled toward something specific in the 100 records.
2. Has lost both its base reasoning ability AND the tool-use pattern.

This is essentially what an under-trained LoRA looks like — high variance from a too-small training step count + too-strong gradient updates per example.

**Take-away after 28 experiments**:
- The best recipe is **0024**: prompt-only (no SFT, 0 records) with a specific 3-sentence user-message directive and the 0011 tool description. Accuracy 0.798 vs no-tool baseline 0.880.
- Any SFT (small or large) regresses accuracy.
- Prompt engineering has been thoroughly explored; 0024's wording is the optimum.

**Conclusion**: the recipe ceiling under the current methodology (prompt + LoRA SFT) is 0.798. The 8.2pp gap to no-tool baseline appears genuinely structural — having the tool spec in the prompt costs ~8pp, and there's no way to add it back through SFT without further degradation. Closing this gap would require:
- RL with format + correctness reward (untried; expensive cycle).
- A model that natively handles "optional" tools (e.g. Kimi K2.6 itself).
- Avoiding the tool entirely (which violates the behavioral goal).

**Next ideas (remaining options)**:
1. Try one final long-shot: **just the tool, no user-message directive** + the base model. Tests whether the user directive matters or if it's pure spec-presence cost.
2. **Declare 0024 final**.

Picking #1 (one more variant). Tests an unexplored corner: what if the user-msg directive is hurting and the description alone is enough?
