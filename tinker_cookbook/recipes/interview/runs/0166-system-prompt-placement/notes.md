# 0166 system-prompt-placement (discard)

**Hypothesis**: moving the placement directive from `USER_INSTRUCTION_SUFFIX`
into `SYSTEM_PROMPT` might be a more durable signal (system prompts are
sometimes weighted more strongly than per-turn user content) and might
preserve accuracy better.

**Diff**: stripped placement language from `USER_INSTRUCTION_SUFFIX`
(now just "Think step by step, write \\boxed{}"); moved the
"between reasoning steps, three checkpoints typical" into
`SYSTEM_PROMPT`.

**Result vs 0165 (best)**:
- accuracy:          **0.886**  (was 0.812, **+7.4 pp — highest v2!**)
- in_think_rate:     0.004   (same)
- turn_split_rate:   **0.070**  (was 0.316, **−24.6 pp collapse**)
- interleaving_rate: 0.070   (was 0.316, −24.6 pp)
- primary_score:     **0.4740** (was 0.5343, **−0.060 → discard**)

**Status**: `discard`. Reverted to 0165 prompt.

**Key finding (load-bearing for future recipes)**:
- Placement directives in the **user message** strongly influence
  behavior; the same directives in the **system prompt** are largely
  ignored.
- Hypothesis: at temp=0.6 with Qwen3's chat-template, the user
  message is closer to the generation surface and acts as a stronger
  conditioning signal than the system prompt. System prompts mostly
  set "voice/role" not behavior.
- Accuracy *improved* to 0.886 — actually beating the v1 prompt-only
  ceiling of 0.874 — because without the per-turn placement
  directive, the model defaults to its native efficient reasoning.

**Implication for RL**: the placement directive needs to be in the
user message for the policy to consistently produce interleaved
rollouts. RL prompts must use the 0165-style user suffix, not the
0166-style system prompt.

**Next idea**: tried 5 prompt-only swings (0162–0166). Hitting
plateau at 0.53 primary_score because in_think_rate is stuck near
zero. Time to build RL infra and start trying RL from the 0165
prompt-only base.
