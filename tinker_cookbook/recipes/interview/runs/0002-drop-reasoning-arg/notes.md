# 0002 drop-reasoning-arg

**Hypothesis**: v3's tool args (`{summary, reasoning}` with `reasoning` = full thinking content) duplicated thinking content into the tool call. The duplication trained the model to be verbose and confident about emitting tool calls, *and* made the loss curve artificially easy (lots of predictable repeated tokens). Reverting to v1's `{message}` only — short summary, no reasoning content — should recover v1's 0.736 accuracy.

**Diff**: sft_train.py
- `PROGRESS_TOOL_SPEC` parameters: dropped `reasoning`, kept only `message`.
- `_assistant_turn_with_update`: tool_call args = `{"message": summary}` instead of `{"summary": summary, "reasoning": thinking}`.

Eval already handles both formats (`args.get("summary", args.get("message", ""))`), no eval edits needed.

**Training**: 142 steps × batch 16 over 2301 records. Final train NLL **0.270** (vs v3's 0.109 — much higher because no duplicated content makes loss harder, which is correct).

**Result**: accuracy **0.736**. Cadence: `0:138, 1:250, 2:102, 3:10`.

| Run | accuracy | cadence (0:1:2:3) | nll | notes |
|-----|----------|-------------------|------|-------|
| v3 baseline | 0.708 | 342:137:18:2 | 0.109 | +reasoning arg |
| 0001 | 0.708 | 377:98:21:4 | 0.107 | LR 1.5e-4 (no effect) |
| **0002** | **0.736** | **138:250:102:10** | **0.270** | drop reasoning arg |

**Status**: `keep`. +2.8pp over v3 baseline. Reproduces the v1 number we'd hoped to see. The duplicated `reasoning` arg WAS the main regression cause — not LR.

**Cadence is healthier**: 72% of problems now use 1+ tool calls (vs v3's 32%). Model is using the tool naturally.

**Next idea**: now that we're on the right track at 0.736 (still 14pp below the 0.880 no-SFT baseline), try to recover more. Two angles:
1. **Lower LR** on top of 0002 — earlier 0001 (LR 1.5e-4 on v3 reasoning-arg) had no effect. But on the simpler `message`-only format, lower LR may now help. Cheap retry.
2. **Mix in pure-math no-tool data** to preserve base reasoning — model regressed on 28% of problems that emit 0 tools, suggesting baseline capability is still being damaged.

Picking #1 first (cheaper): retry LR 1.5e-4 on top of 0002. Then #2 if no improvement.
