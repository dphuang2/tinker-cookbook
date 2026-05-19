# 0173 phaseA-scaffold (v3 Phase A bootstrap — infra-only)

**Hypothesis (none — scaffolding tick)**: Phase A (OPSD,
Self-Distilled Reasoner style) cannot be built off the cookbook's
distillation pipeline as-is because it ties the teacher's logprob
scoring to the student's prompt. We need a custom training loop here
in `recipes/interview/`.

**What landed**:
- `tinker_cookbook/recipes/interview/opsd_train.py` — scaffold with:
  - `TEACHER_PRIVILEGED_SUFFIX` defining the answer-conditioned
    addendum the teacher sees and the student does not.
  - `make_teacher_user_message(question, answer)` /
    `make_student_user_message(question)` helpers — both reuse
    `USER_INSTRUCTION_SUFFIX` from `sft_train.py` so train/eval
    match exactly.
  - `@chz.chz OPSDConfig` with all hyperparameters defaulted to
    reasonable starting points (LoRA-32, lr=1e-4, group_size=4,
    groups_per_batch=32, kl_penalty_coef=1.0, max_steps=100).
  - `main(config)` raises NotImplementedError — explicit list of
    TODOs in the module docstring.

**Open TODOs (one per future tick)**:
1. Multi-turn agent loop for student rollouts (must mirror
   eval_deepmath_agent.py: tool spec exposed, MAX_TURNS=8, the 0170
   throttled ack content). Reuse where possible.
2. Teacher logprob scoring on student sequences, but with the teacher
   prompt being the *privileged* prompt + the student's generated
   tokens.
3. Reverse-KL advantage computation per token → forward_backward
   training step.
4. Checkpoint emission with `sampler_path` recorded in
   `checkpoints.jsonl` so eval can re-evaluate any intermediate
   sampler.
5. Held-out eval hook every N steps using `eval_deepmath_agent.py`.

**Status**: `keep` — pure scaffolding; no metric to report. No
results.tsv row (no model trained).

**Next idea**: implement TODO 1 (multi-turn student rollouts). This
is independently testable: run the rollout loop, dump 10 sample
trajectories to a JSON file, confirm shape matches eval. Once
trajectories are in hand, hook them into TODO 2 (teacher scoring).
