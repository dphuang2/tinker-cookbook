# 0175 phaseA-teacher-score (TODO 2 from 0173)

**Hypothesis**: `score_with_teacher` can compute privileged-info
teacher logprobs on the student's full multi-turn rollout, including
the historical `<think>` blocks that Qwen3's default renderer strips.

**Key finding (bug → fix)**: First attempt (smoke2) returned
`sequence_len=2781` for a rollout with `n_assistant_tokens=23704`.
The Qwen3 chat template strips historical `<think>...</think>` blocks
by default (matching HF). Fix: instantiate a `Qwen3Renderer` with
`strip_thinking_from_history=False` inside `score_with_teacher`. After
the fix, smoke3 shows `sequence_len=31911`, `n_assistant_tokens=31349`
— the prefix is ~562 tokens (system + privileged user) and the rest
is the student's full content including thinking. Working.

**Smoke3 result** (idx=500, answer="Yes"):
- 5 turns / 4 calls / 4 splits / 31349 student tokens
- Teacher scored 31911 token positions, 31349 marked as student
- Mean teacher logprob over student tokens: **−0.288**
  (≈ exp(−0.288) = 0.75 average per-token probability under the
   privileged-info teacher)

**Status**: `keep` (infra). No model yet.

**Why the teacher logprob is high but not 1.0**:
- Same base model so most tokens are close-to-MAP under both
  distributions.
- The privileged context (ground-truth answer + "spread checkpoints
  evenly" directive) shifts a minority of tokens — exactly the ones
  we want to push the student toward. Those will dominate the
  reverse-KL contribution.

**Next TODO (3)**: assemble training data and run a single
forward_backward step. The cookbook's `incorporate_kl_penalty` shows
the formula:
```
reverse_kl = sampled_logprobs - teacher_logprobs
advantage  = -kl_penalty_coef * float_mask * reverse_kl
```
We need `sampled_logprobs` (the student's per-token logprobs at
sample-time — sampling_client returns these). Then wrap in a
`tinker.Datum` and call `training_client.forward_backward_async` +
`optim_step_async`.
