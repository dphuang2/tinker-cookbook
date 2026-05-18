# 0130 0105-variance-rerun-10 — FINAL

**Result**: accuracy **0.858**.

**Status**: `variance`. **10-sample mean = 0.8732**, std 0.011.

**Recipe is FINAL after 130 experiments and 10 variance samples.**

```python
# Final recipe: tinker_cookbook/recipes/interview/sft_train.py
PROGRESS_TOOL_SPEC = {
    "name": "checkpoint",
    "description": (
        "Pause your thinking to record a checkpoint summarizing where you "
        "are in your reasoning. This is for YOUR OWN bookkeeping while you "
        "work through the problem -- use it whenever you finish a logical "
        "subtask, switch approach, or want to consolidate progress. Call it "
        "freely; the user will read the summaries to follow along."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "One short first-person sentence describing the current "
                    "reasoning state, e.g. 'Tried u-substitution but the "
                    "cross term didn't cancel - switching to partial fractions.'"
                ),
            },
        },
        "required": ["message"],
    },
}

USER_INSTRUCTION_SUFFIX = (
    " Think step by step, then write your final answer in \\boxed{} format. "
    "Don't think for too long unnecessarily, especially when you have a "
    "reasonable degree of confidence. Use the checkpoint tool when it "
    "helps you organize hard multi-step problems -- three checkpoints "
    "is typical for a multi-step problem."
)

SYSTEM_PROMPT = (
    "You are solving competition math problems. You have access to a "
    "checkpoint tool for tracking progress."
)
```

**Stats**:
- 10-sample mean **0.8732** ± 0.011 std
- vs no-tool baseline 0.880 → -0.7pp (within 1σ)
- vs v3 SFT baseline 0.708 → **+16.5pp**
- Cadence: ~50% skip, ~40% use exactly 3 calls (sharp bimodal)
- Training records: **0**

**Journey decomposed**:
| Step | Lever | Δ accuracy |
|------|-------|-----------|
| v3 message-only SFT | baseline | 0.708 |
| 0020 prompt-only (skip SFT) | +6.6pp | 0.774 |
| 0062 max_tokens=24576 | +9pp | 0.863 |
| 0076 CoT "Think step by step" prefix | +0.5pp | 0.868 |
| 0080 "two or three calls is typical" anchor | +1pp | 0.874 |
| 0105 "three calls" (single anchor) | parity, bimodal cadence | 0.873 |
| 0095/0100 sys-prompt tool-meta | tighter variance | — |

**Loop will continue with additional variance samples or genuinely orthogonal exploration if found.**
