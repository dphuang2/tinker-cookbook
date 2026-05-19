# 0178 phaseA-fwdbwd (TODO 4 from 0173) — END-TO-END WORKING

**Hypothesis**: stitching together rollout + teacher scoring + datum
assembly + LoRA training_client + forward_backward + optim_step
produces a single valid OPSD training step.

**Bug fix vs smoke6**:
- The `importance_sampling` loss does NOT accept the `mask` field;
  it's used only during datum-construction (advantage shaping) then
  must be stripped. See `tinker_cookbook/rl/train.py:_remove_mask`.
  Smoke7 wraps the Datum and removes `mask` before calling
  `forward_backward_async`.

**Smoke7 (N=4) — final results**:
```
Rolling out 4 student trajectories...
  idx=500 turns=5 calls=4 splits=4 tokens=13952
  idx=501 turns=5 calls=4 splits=4 tokens=10160
  idx=502 turns=5 calls=4 splits=4 tokens=6786
  idx=503 turns=4 calls=3 splits=3 tokens=2546

score_with_teacher (idx=500, answer='Yes'):
  teacher_logprobs len=14508 n_assistant_tokens=13952
  mean teacher logprob on student tokens: -0.330

assemble_opsd_datum:
  model_input.length=14507 n_masked_positions=13952
  adv_mean=-0.193 adv_std=1.304 adv_min=-29.75 adv_max=+7.01

forward_backward + optim_step:
  loss outputs: logprobs[14507] mean=-0.407
  optim_step: completed
```

**Status**: `keep` — Phase A infrastructure complete. All five
primitives chain end-to-end:
1. roll_out_student
2. score_with_teacher (preserve-thinking renderer, teacher_mode A/B/C)
3. assemble_opsd_datum (correct sign convention, leftshifted targets)
4. training_client.forward_backward_async (drop mask first!)
5. training_client.optim_step_async

**Now ready for**:
- TODO 5: assemble `opsd_train.main(config)` — proper outer loop:
  - sample N problems per step, run rollouts concurrently
  - filter overlong rollouts (>28k tokens, see 0177)
  - score each, build datum, batch as list, forward_backward
  - save sampler checkpoint every `save_every` steps
  - eval hook every `eval_every` steps using
    eval_deepmath_agent.py with sampler_path pointed at the saved
    LoRA weights
- TODO 6: run a small training (e.g. max_steps=20,
  groups_per_batch=8) and inspect the resulting sampler's
  primary_score vs the v2.1 baseline (0.3444).

**Estimate**: 8 problems × ~10k avg tokens × ~30s per
forward_backward + ~3 min per sample-rollout → first 20-step
training is ~30 min sample + ~10 min training = 40 min.
Plus 10 min for held-out eval. Total ~50 min for the first
real Phase A signal.
