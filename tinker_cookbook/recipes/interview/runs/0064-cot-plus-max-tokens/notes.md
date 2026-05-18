# 0064 cot-plus-max-tokens-24576

**Hypothesis**: combine 0036's "Think step by step, then" CoT prefix in USER_INSTRUCTION_SUFFIX with 0062's max_tokens=24576 sweet spot. Tests whether CoT framing compounds with the unlocked thinking budget.

**Diff**:
- `eval_deepmath_agent.py`: MAX_TOKENS_PER_TURN 28672 → 24576 (revert 0063's saturation point back to 0062 sweet spot; this part kept).
- `sft_train.py` USER_INSTRUCTION_SUFFIX: added "Think step by step, then " prefix before "write your final answer..."

**Result**: accuracy **0.876**, cadence `0:477, 1:20, 2:1, 3:2`.

| Run | accuracy | 0-call % | description |
|-----|----------|----------|-------------|
| 0062 (no CoT, mt=24576) | 0.870 | 75.6% | baseline |
| 0063 (no CoT, mt=28672) | 0.866 | 79.0% | saturation |
| 0064 (CoT, mt=24576) | **0.876** | **95.4%** | this run |
| no-tool baseline | 0.880 | 100% | upper bound |

**Status**: `discard` (cadence degenerate). Accuracy is +0.6pp over 0062 and within 0.4pp of the no-tool baseline, but only 23/500 problems use the tool at all. Per PROGRAM.md priority 3, cadence must stay non-degenerate; this is degenerate.

**Interpretation**: the explicit "Think step by step" prefix sharpens the chain-of-thought directive and pulls the model toward "answer directly" mode, completely suppressing tool calls on simple problems. The accuracy gain comes from no longer paying the tool-spec cost (close to no-tool baseline). This is essentially a "stealth no-tool" recipe.

**Take-away**: there's a clear tradeoff between accuracy and cadence retention. The CoT prefix breaks the cadence floor.

**Action**: revert sft_train.py to keep 0062's wording (no CoT prefix). Keep eval_deepmath_agent.py at MAX_TOKENS_PER_TURN=24576 (0062's setting), since 0063 confirmed 28672 is past saturation.

**Best non-degenerate remains 0062 at 0.870 (cadence 75.6% 0-call, 122/500 problems use the tool).**

**Next ideas**:
1. **Weaker CoT prefix** — instead of "Think step by step, then", try a softer "Reason carefully, then" or even just whitespace — see if any portion of the 0.876 gain transfers without collapsing cadence.
2. **Cadence-stabilizing reward** — out of scope (would need RL).
3. **Tool-call-required floor** — explicit minimum tool calls. Likely degrades accuracy.

Picking #1 (softer CoT phrasing): probes whether the "Think step by step" specifically is what suppresses tool use, or if any reasoning-emphasis prefix does it.
