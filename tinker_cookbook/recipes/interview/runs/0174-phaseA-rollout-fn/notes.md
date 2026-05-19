# 0174 phaseA-rollout-fn (TODO 1 from 0173)

**Hypothesis**: a clean reusable `roll_out_student` function — mirror-
ing eval_deepmath_agent.py exactly — can be tested standalone on a
handful of train problems before being wired into the OPSD loop.

**What landed**:
- `tinker_cookbook/recipes/interview/opsd_train.py`:
  added `roll_out_student(...)` that does multi-turn agent loop with
  PROGRESS_TOOL_SPEC, SYSTEM_PROMPT, USER_INSTRUCTION_SUFFIX, and the
  0170 state-aware ack throttle (4-call threshold). Returns
  `(all_tokens, turn_token_ranges, decoded, n_tool_calls, n_turn_splits,
  in_think_calls, tool_call_char_positions)` — enough to compute
  per-rollout teacher logprobs (TODO 2) and primary_score components
  (TODO 4 eval hook).
- `tinker_cookbook/recipes/interview/opsd_smoke_test.py`: standalone
  smoke test that runs N student rollouts in parallel and dumps a
  shape summary.

**Smoke result** (3 problems, DeepMath train idx 500–502):
```
idx=500 turns=1 calls=0 splits=0 tokens=13231  (skip — base mixture)
idx=501 turns=5 calls=4 splits=4 tokens=11165  (interleaved!)
idx=502 turns=4 calls=3 splits=3 tokens=5002   (interleaved)
```

idx=501 and 502 show the natural cross-turn cadence the base model
produces with the 0170 throttled ack — `n_turn_splits` matches
`n_tool_calls`, i.e. each call lives in its own turn rather than
being batched.

**Status**: `keep` (infra). No metric to record.

**Next TODO (2)**: teacher logprob scoring on these rollouts using
the privileged prompt. The teacher sees
`question + USER_INSTRUCTION_SUFFIX + TEACHER_PRIVILEGED_SUFFIX.format(answer=gt)`
and the student's generated `all_tokens`; we want logprob per token
to compute reverse-KL = log p_student − log q_teacher.

Concrete shape: write `score_teacher(rollout, answer) -> list[float]`
that builds the teacher prompt and calls
`sampling_client.compute_logprobs_async` (or equivalent) on the
concatenation of teacher-prompt-tokens + student-generated tokens.
Mask logprobs to the assistant-token range only.
