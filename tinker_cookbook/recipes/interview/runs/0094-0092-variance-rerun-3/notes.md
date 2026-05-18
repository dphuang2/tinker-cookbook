# 0094 0092-variance-rerun-3

**Result**: accuracy **0.862**.

**Status**: `variance`.

| Recipe | n | samples | mean |
|--------|---|---------|------|
| 0080 | 4 | 0.876, 0.868, 0.884, 0.870 | 0.8745 |
| **0092** | **3** | **0.878, 0.876, 0.862** | **0.872** |

3-sample mean for 0092 dropped to 0.872 — at parity with 0080. The "tight variance" from 2 samples was an artifact.

**Take-away**: identity-only SYSTEM_PROMPT is neutral on accuracy. Doesn't help, doesn't hurt. Cadence slightly less tool use (~60% 0-call).

**Decision**: keep the system prompt anyway (clean framing, no cost). Recipe ceiling firmly established at ~0.872-0.875 across multiple variants.

**Pattern after ~33 post-0062 experiments**: every prompt-engineering lever converges to the same ~0.87 accuracy ceiling. The recipe is robust to minor variations on multiple axes. The structural ceiling on accuracy under prompt-only is ~0.87, with no-tool baseline 0.88.

**Next ideas**:
1. **Declare final** — recipe ceiling thoroughly mapped.
2. **Test on a different DeepMath slice** — generalization (out of bounds per "FIXED slice").
3. **Try mini-RL** — out of scope/expensive.

Picking #1 effectively — but keep trying orthogonal angles since loop is autonomous. Next: try varying the eval's HELDOUT_START. Wait — that's FIXED. 

Going to test: removing the anti-rumination sentence (0078 tested under 0076 wording, but not under 0080). Already tested... let me try something else.

Actually, let me try one more orthogonal: combine 0080 + 0064-style suppressor (the strict "for simple problems just answer"). Counter to 0085 finding maybe. Wait, 0064 had this suppressor. Was degenerate cadence at 95% 0-call. Bad idea.

Best path: try removing the "Don't think for too long unnecessarily..." sentence under 0080+0092 recipe. Tests interaction with the 2-3 anchor.
