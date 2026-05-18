# 0098 0095-variance-rerun-3 — RECIPE FINAL

**Result**: accuracy **0.876**, cadence `0:292, 1:8, 2:8, 3:155, 4:26, 5:4, 6:4, 7:1, 8:2`. 58.4% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean | std |
|--------|---|---------|------|-----|
| 0062 | 3 | 0.870, 0.864, 0.854 | 0.863 | 0.008 |
| 0076 | 3 | 0.876, 0.854, 0.852 | 0.861 | 0.014 |
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 | 0.007 |
| 0092 | 3 | 0.878, 0.876, 0.862 | 0.872 | 0.009 |
| **0095** | **3** | **0.880, 0.874, 0.876** | **0.877** | **0.003** |
| no-tool baseline | - | - | 0.880 | - |

**0095-recipe is the FINAL**:
- 3-sample mean **0.877** (within 0.3pp of no-tool baseline 0.880).
- Tightest variance (std 0.003 — most reproducible).
- Healthy cadence: ~58% 0-call, peak at 3 calls (~31%), 42% use the tool.

**Decomposed gains over journey**:
| Step | Lever | Δ accuracy |
|------|-------|-----------|
| v3 baseline | message-only SFT | 0.708 |
| 0020 | prompt-only (skip SFT) | +6.6pp → 0.774 |
| 0062 | max_tokens 8192→24576 | +9pp → 0.863 |
| 0080 | "two or three calls is typical" anchor | +1.1pp → 0.874 |
| **0095** | + identity & tool-meta system prompt | **+0.3pp + variance tightening → 0.877** |

**Final recipe**:
```python
SYSTEM_PROMPT = (
    "You are solving competition math problems. You have access to a "
    "checkpoint tool for tracking your reasoning progress on hard "
    "multi-step problems."
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
    "description": <verbose, ~60 words encouraging "call it freely">,
    "parameters": {"message": {"type": "string", "description": <u-sub example>}},
}
```

- **Base**: Qwen3-30B-A3B (no SFT, 0 training records)
- **Eval**: temp=0.6, max_tokens=24576, max_turns=8
- **Cadence**: 58% 0-call, 42% use tool (peak at 3 calls)
- **Accuracy**: 0.877 ± 0.003 (n=3)

**Pattern (all 38 post-0062 experiments)**: ceiling firmly at ~0.87-0.88. The 0.880 no-tool baseline is the structural ceiling; the recipe gap closed via careful directive design. No prompt-engineering variation achieves >0.88 reliably.

**Closing notes**:
- 0080-recipe (no system prompt) at 0.8745 mean is essentially equivalent — would also be a fine final.
- 0095 chosen for tighter variance and cleaner system-prompt framing.
- All "discards" within ~1pp of 0.87 were within noise; the optimum is robust.
