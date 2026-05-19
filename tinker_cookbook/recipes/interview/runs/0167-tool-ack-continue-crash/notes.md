# 0167 tool-ack-continue (CRASH)

**Hypothesis**: changing the tool ack content from "ok" to
"noted; continue your reasoning" would nudge the model to keep
reasoning rather than emit another tool_call, improving the cadence
shape without losing accuracy.

**Diff**: `eval_deepmath_agent.py:125` — `"content": "ok"` →
`"content": "noted; continue your reasoning"`.

**Result**: `crash`. The entire eval failed with:
```
tinker.BadRequestError: 400 Prompt length plus max_tokens exceeds
the model's context window: 9734 prompt tokens + 24576 max_tokens
> 32768.
```

**Why**:
1. `asyncio.gather` propagates the first exception, killing all 499
   other concurrent rollouts.
2. The longer ack ("...continue your reasoning" vs "ok") plus
   resulting verbose history pushed at least one rollout's prompt
   past the 32768 - 24576 = 8192 token budget.
3. Fundamental limit: with `max_tokens_per_turn=24576` against
   Qwen3-30B-A3B's 32k context, any conversation accumulating >8k
   tokens of history fails on the next turn.

**Reverted** ack to "ok" + added per-problem exception handling
(`return_exceptions=True` + error fill) so future evals don't die
on a single bad rollout. Added `num_errored` to summary.

**Status**: `crash`. No results.tsv row (no metric to record).
Reverting causal change; the eval-robustness fix is layered on top
because the next experiment would have the same risk otherwise.

**Next idea**: now that the eval is robust, retry the tool-ack
continue experiment AND begin sketching RL infra. The plateau at
0.53 primary_score with prompt-only is the real signal — RL is
the next real lever.
