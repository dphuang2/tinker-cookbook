# 0180 phaseA-outerloop-smoke (TODO 5/6 partial — verified 2-step run)

**Hypothesis**: `opsd_train.main` outer loop works end-to-end at
small scale. Sample 4 problems per step, filter overlong, score,
build datums, forward_backward+optim_step, save sampler, repeat.

**Config**: `max_steps=2 groups_per_batch=4 save_every=1
train_index_end=520` (20-problem pool).

**Result**:
```
step 1: rollouts kept=3 skipped_overlong=1
        trained on 3 datums, mean_training_logprob=-0.619
        Saved sampler at step 1
step 2: rollouts kept=4 skipped_overlong=0
        trained on 4 datums, mean_training_logprob=-0.556
        Saved sampler at step 2
OPSD training complete. 2 sampler checkpoints saved.
```

mean_training_logprob is moving toward 0 (−0.619 → −0.556) —
healthy direction. The student is learning to produce its own
tokens at higher probability after the reverse-KL nudge toward
the teacher distribution.

**Status**: `keep` — outer-loop infrastructure verified.

**Checkpoints saved** (for next-tick eval):
- step_1: `tinker://a2923ff2-...:train:0/sampler_weights/step_1`
- step_2: `tinker://a2923ff2-...:train:0/sampler_weights/step_2`

**Now ready for the first real Phase A signal**:
The step_2 sampler can be evaluated against the held-out 500
DeepMath problems via:
```
SAMPLER_PATH="tinker://...:train:0/sampler_weights/step_2" \\
LD_PRELOAD=... .venv/bin/python -m \\
  tinker_cookbook.recipes.interview.eval_deepmath_agent
```

Expected at 2-step LoRA: minimal change vs base prompt-only.
Need a real training run (max_steps=50-100, groups_per_batch=32+)
to see meaningful primary_score movement.

**Next idea**: launch a real Phase A training run:
- max_steps=20 (cheap first try)
- groups_per_batch=8
- save_every=10
- train_index_end=2000 (1500-problem pool)
- All other defaults

Estimated wall time: ~20 × (8 problems × ~5min rollout + 30s
training) ≈ ~3 hours. Worth letting it run overnight and
evaluating step_10, step_20 against the v2.1 baseline (0.3444).
