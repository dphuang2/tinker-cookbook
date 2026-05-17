# 0024 directive-in-user-msg

**Hypothesis**: 0022's sparing-use directive worked (+1.8pp). Move it to the user message instead of the system prompt — Qwen3 may weight user instructions more heavily than system prompts since the latter is "infrastructure" framing.

**Diff**: sft_train.py
- SYSTEM_PROMPT = "" (reverted to empty).
- USER_INSTRUCTION_SUFFIX extended with the same "sparing use" instruction that was in 0022's system prompt.

Same prompt-only setup, no training.

**Result**: accuracy **0.798** — **new best!** +0.6pp over 0022 (0.792).

Cadence: `0:412, 1:58, 2:12, 3:11, 4:3, 5:2, 8:2`. 82% 0-call, very similar to 0022.

| Recipe | accuracy | 0-call % |
|--------|----------|----------|
| 0020 (no instruction) | 0.774 | 45% |
| 0022 (sys-prompt sparing) | 0.792 | 79% |
| **0024 (user-msg sparing)** | **0.798** | **82%** |
| no-tool baseline | 0.880 | n/a |

**Status**: `keep`. New best at 0.798, 0 training records. Gap to no-tool baseline now 8.2pp.

**Take-away**: positioning of the same instruction matters. Moving it to the user message (closer in attention to the question itself) gave +0.6pp.

**Next ideas**:
1. **Tighten the user-message directive further** — see if there's a shorter/sharper phrasing that works better.
2. **Vary instruction strength** — try "DO NOT call the tool unless absolutely needed" vs current "use it sparingly".
3. **Combine 0024 + soft directive in system prompt** — both positions reinforcing each other.

Picking #3 (both positions): cheap, tests if redundancy helps.
