# 0102 0100-variance-rerun-3 — RECIPE LOCKED

**Result**: accuracy **0.882**, cadence `0:302, 1:11, 2:9, 3:137, 4:32, 5:8, 6:1`. 60.4% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| **0100/0095** | **7** combined | 0.880, 0.874, 0.876, 0.882, 0.882, 0.870, 0.882 | **0.878** | **0.005** |
| no-tool baseline | - | - | 0.880 | - |

(Combining 0095 + 0100 since they're equivalent — only difference is system prompt qualifier.)

**Final mean: 0.878 ± 0.5pp on 7 samples.** Statistically AT no-tool baseline 0.880.

**Recipe LOCKED.**

```python
# tinker_cookbook/recipes/interview/sft_train.py
SYSTEM_PROMPT = (
    "You are solving competition math problems. You have access to a "
    "checkpoint tool for tracking progress."
)

USER_INSTRUCTION_SUFFIX = (
    " Think step by step, then write your final answer in \\boxed{} format. "
    "Don't think for too long unnecessarily, especially when you have a "
    "reasonable degree of confidence. Use the checkpoint tool when it "
    "helps you organize hard multi-step problems -- two or three calls "
    "is typical for a multi-step problem."
)

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
```

- **Base**: Qwen3-30B-A3B (no SFT, 0 training records)
- **Eval (FIXED)**: temp=0.6, max_tokens_per_turn=24576, max_turns=8, agent loop with "ok" tool ack
- **Accuracy**: 0.878 ± 0.005 (n=7 across 0095/0100 variants)
- **Cadence**: ~60% 0-call, peak at 3 calls (~28%) — 40% of problems engage the tool
- **vs no-tool baseline** (0.880): **at parity** (within 1σ noise)
- **Training records**: **0** (zero data, pure prompt engineering)

**Definitive journey**:
| Cell | Δ | Cumulative |
|------|----|------|
| message-only SFT (v3, 2302 records) | baseline | 0.708 |
| 0020 prompt-only (0 records) | +6.6pp | 0.774 |
| 0062 max_tokens=24576 eval | +9pp | 0.863 |
| 0076 CoT prefix in user msg | +0.5pp | 0.868 |
| 0080 "two or three calls typical" anchor | +1pp | 0.874 |
| 0095/0100 identity+tool-meta system prompt | +0.3pp | **0.878** |

The journey was: discovery that SFT hurts → discovery that the eval cap was the real bottleneck → discovery that prompt anchoring tool cadence is the only remaining lever.
