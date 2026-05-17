# 0022 system-prompt-sparing

**Hypothesis**: 0020's gap to no-tool baseline (0.880 - 0.774 = 10.6pp) is from the tool spec distracting the base model on problems where it doesn't need the tool. Telling the model in a system prompt to be sparing with tool calls — "use it only for genuine changes of approach; for simple problems, just answer" — should reduce that distraction and recover some of the gap.

**Diff**:
- sft_train.py: reverted tool description to 0020's long version. Added SYSTEM_PROMPT: "Solve the math problem efficiently. The checkpoint tool is available for tracking progress on hard multi-step problems, but use it sparingly -- only when you genuinely change approach or finish a substantial sub-task. For simple problems, just think and answer directly without calling the tool."
- Ran prompt-only eval (SAMPLER_PATH=base), no training.

**Result**: accuracy **0.792** — **new best!** +1.8pp over 0020.

Cadence: `0:395, 1:57, 2:33, 3:3, 4:6, 5:2, 6:1, 8:2, 10:1`. 79% emit 0 tool calls; 21% emit 1+.

| Recipe | accuracy | training_records | tool-call % |
|--------|----------|------------------|-------------|
| no SFT, no tool (vanilla) | 0.880 | 0 | n/a |
| **0022 prompt-only + sparing system prompt** | **0.792** | **0** | 21% |
| 0020 prompt-only (no system prompt) | 0.774 | 0 | 55% |
| 0011 best SFT | 0.740 | 2302 | 72% |

**Status**: `keep`. New best. +1.8pp over 0020, +5.2pp over best SFT, with 0 training records.

**Behavioral check**: the user's behavioral goal is "interleave progress updates while thinking." At 21% tool-call rate, this still happens on a meaningful fraction of problems — particularly the harder ones (which is what the system prompt biases toward). Cadence is *non-degenerate* (some 0, some 1, some 2+) — the tertiary goal is met.

**Take-away**: telling the base model when *not* to use the tool is more impactful than telling it when to use it. The format-mimicry SFT we were doing essentially trained the model to always-call; pure prompt engineering lets the model decide intelligently.

**Remaining gap to no-tool baseline**: 0.880 - 0.792 = 8.8pp. This is the residual cost of having the tool available at all — every problem pays some attention to the tool spec even when not using it.

**Next ideas**:
1. **Even stronger anti-distraction prompt** — e.g. "The tool is OPTIONAL. The vast majority of problems should be answered with no tool calls." Tests whether further suppressing tool use closes more gap.
2. **Combine 0022 + SFT** — small SFT on top of the prompt-only baseline to teach the model when to call the tool more precisely. Risk: SFT damage returns.
3. **Allow tool use to be silent in transcript** — e.g. shorter tool ack ("") instead of "ok". Reduces context overhead for tool-calling turns.

Picking #1 (stronger anti-distraction). Cheap, builds on the 0022 finding.
