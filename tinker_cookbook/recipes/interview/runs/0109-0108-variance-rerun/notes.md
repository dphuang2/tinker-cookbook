# 0109 0108-variance-rerun

**Result**: accuracy **0.860**, cadence `0:253, 1:7, 2:7, 3:195, 4:18, 5:2, 6:11, 8:1, 9:4, 12:1, 18:1`. 50.6% 0-call.

**Status**: `variance`.

| Recipe | n | mean |
|--------|---|------|
| 0100 (sysprompt + 2-3 calls) | 8 | 0.8775 |
| 0105 (sysprompt + 3 calls) | 3 | 0.878 |
| **0108 (no-sys + 3 calls)** | **2** | **0.874** |

All three variants at parity. The 0.888 from 0108 was variance.

**Take-away**: 5+ recipe variants all converge to ~0.876 ± 0.5pp. Recipe is statistically saturated.

**Final summary**: prompt-only ceiling ≈ 0.876 across many wordings. No-tool baseline 0.880. The 0.4pp gap is the structural cost of having any tool spec in the prompt; cannot be closed via prompt engineering alone.

Recipe is essentially DONE. Future tweaks within noise.

**Current canonical recipe** (0105/0108 family — sharply bimodal cadence):
- Optional minimal SYSTEM_PROMPT
- USER_INSTRUCTION_SUFFIX with "three calls is typical" anchor
- 0 training records
- ~50% tool use, ~50% direct answer
