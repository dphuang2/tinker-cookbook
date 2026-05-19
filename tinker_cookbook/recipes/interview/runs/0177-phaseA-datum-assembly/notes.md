# 0177 phaseA-datum-assembly (TODO 3 from 0173)

**Hypothesis**: with student logprobs captured during rollout and the
teacher's privileged-info logprobs in hand, we can assemble a
`tinker.Datum` ready for `forward_backward`. Verify the resulting
advantage distribution is sensible.

**What landed (cumulative since 0173)**:
- `roll_out_student` (0174) captures `all_logprobs` per token now.
- `score_with_teacher` (0175) returns `sequence_tokens` so the datum
  builder can reconstruct model_input.
- `assemble_opsd_datum` (this run) constructs:
  - `model_input  = sequence_tokens[:-1]`
  - `target_tokens = sequence_tokens[1:]`
  - `logprobs  = pad(student_logprobs, prefix_len) | aligned to targets`
  - `advantages = -kl_coef * mask * (student_lp - teacher_lp)`
  - `mask = 1 on student-generated positions only`
- Teacher modes parameterized (Option A default, B and C plumbed).

**Smoke5 (N=4)**:
```
idx=500 turns=5 calls=4 splits=4 tokens=27692   scored, datum built
idx=501 turns=5 calls=4 splits=4 tokens=10724
idx=502 turns=5 calls=4 splits=4 tokens=5895
idx=503 turns=4 calls=3 splits=3 tokens=2894
```
All 4 rollouts cross-turn interleaved. Datum on idx=500:
- model_input.length=28249
- n_masked_positions=27692 (= n_assistant_tokens, correct)
- adv_mean(masked)=−0.175  (mild average pull toward teacher)
- adv_std(masked)=1.179
- adv range [−29.06, +8.34]

Asymmetric range is the actual signal: positive-adv tokens are ones
the teacher is much more confident in than the student (likely the
checkpoint-related tokens we want the student to emit more), and
negative-adv tokens are ones the student is over-confident in
(non-checkpoint tokens at positions where the teacher would have
checkpointed).

**Known gotcha**: rollouts exceeding 28K tokens overflow the
teacher's 32K context window. Added a token-budget skip in the
smoke test; training pipeline must do the same per rollout.

**Status**: `keep` (infra). Phase A's data layer is complete.

**Next TODO (4)**: actually launch a `forward_backward` step on a
single Datum. Use `tinker.ServiceClient.create_lora_training_client_async`
or equivalent, with `lora_rank=32` and `learning_rate=1e-4`.
Verify weights update without NaN.

**Next TODO (5, smaller)**: also extract `score_with_teacher` and
`assemble_opsd_datum` reasonable-error handling — e.g. wrap the
context-overflow case so the training loop can drop a single
overlong rollout without dying.
