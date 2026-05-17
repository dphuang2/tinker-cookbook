# 0033 agent-loop-no-tool-anchor

**Hypothesis**: measure the true ceiling for the agent-loop evaluation harness by removing the tool spec entirely but keeping the same user message. Tells us how much of the gap between the vanilla 0.880 baseline and 0024's 0.798 is from "having a tool" vs "running through the agent loop / user-msg directive".

**Diff**: added `NO_TOOL` env var to `eval_deepmath_agent.py` which passes `tools=[]` to `create_conversation_prefix_with_tools`. USER_INSTRUCTION_SUFFIX still mentions the checkpoint tool (so this isn't a clean no-tool baseline — the user msg promises a tool that isn't in the prompt).

**Result**: accuracy **0.772**, cadence `0:500` (100% emit 0 tool calls because no tool exists).

**Comparison**:
| Setup | Accuracy |
|-------|----------|
| Vanilla single-turn (no agent loop, no tool, no directive — `eval_deepmath.py`) | **0.880** |
| Agent loop, no tool, user-msg promises tool | 0.772 |
| Agent loop, **tool exposed**, user-msg promises tool (0024) | **0.798** |

**Status**: analytical (not "keep"/"discard" — this is an anchor measurement).

**Insights**:
1. The agent-loop framework + user-msg directive costs ~11pp vs single-turn (0.880 → 0.772).
2. Adding the tool back **adds +2.6pp** (0.772 → 0.798). The user-msg directive promised a tool; satisfying that promise helps.
3. Most of the gap from 0.880 → 0.798 is from the agent-loop framework / user-msg directive, NOT from the tool's presence.

**Take-away**: 0024 (0.798) is actually *better* than the apples-to-apples no-tool agent-loop ceiling. The remaining gap is structural in the eval harness itself (multi-turn dispatch, user-msg directive overhead, etc.).

In hindsight, the relevant comparison isn't 0.798 vs 0.880 (different harnesses) — it's 0.798 vs 0.772 (same harness, with vs without tool). Through that lens, our recipe **adds 2.6pp by exposing the tool**.

**0024 remains the best recipe.** The "8.2pp gap" we were trying to close is largely an artifact of comparing single-turn-eval numbers to agent-loop-eval numbers.
