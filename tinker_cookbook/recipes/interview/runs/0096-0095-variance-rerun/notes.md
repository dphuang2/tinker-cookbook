# 0096 0095-variance-rerun

**Result**: accuracy **0.874**, cadence `0:289, 1:10, 2:13, 3:142, 4:34, 5:5, 6:3, 8:2, 12:1, 24:1`. 57.8% 0-call.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0092 (identity-only system) | 3 | 0.878, 0.876, 0.862 | 0.872 |
| **0095 (tool-meta system)** | **2** | **0.880, 0.874** | **0.877** |

Within noise. Tool-meta in system prompt is essentially neutral on accuracy (~+0.5pp non-significant). Cadence shape unchanged.

**Take-away**: the SYSTEM_PROMPT channel is largely inert for this task — neither identity-only nor tool-meta meaningfully shifts performance. The user-message directive carries the actual signal.

**Action**: keep current (tool-meta system prompt) since it's marginally cleaner framing. But note the 0080-recipe with no system prompt is equally good.

**Effective ceiling**: all 0080-family recipes converge to ~0.87-0.88. The structural prompt-only accuracy is at no-tool baseline.

**Pattern**: extensive exploration confirms 0.87 ± 0.01 ceiling regardless of minor prompt variations.

**Next ideas**:
1. **Declare final**.
2. **Try the recipe combined with the agent loop using shorter MAX_TOKENS** — already FIXED.
3. **Test ack content variations** — already tested.

Picking: try removing the user-message tool-directive entirely now that system prompt mentions the tool. Tests minimal recipe.
