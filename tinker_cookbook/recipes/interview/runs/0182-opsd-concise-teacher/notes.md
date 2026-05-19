# 0182 opsd-concise-teacher (knob 2 from 0181 — discard)

**Hypothesis**: appending "Be concise: do not restate the problem, do
not repeat work already established, do not pad to make chunks equal"
to TEACHER_SUFFIX_A will shorten the teacher's rollout distribution,
which propagates through reverse-KL into a more efficient student.

**Diff**: TEACHER_SUFFIX_A only, in opsd_train.py.

**Training (4 steps, batch=8, save@2,4)**:
```
step 1: n_datums=7 mean_lp=-0.586 (overlong=1)
step 2: n_datums=6 mean_lp=-0.549 (overlong=2)
step 3: n_datums=8 mean_lp=-0.435 (overlong=0)
step 4: n_datums=8 mean_lp=-0.420 (overlong=0)
```
Note: overlong-skip went to 0 after step 2 — small positive sign that
the policy is producing shorter rollouts. But...

**Eval result (step_4 vs 0181 step_2)**:
| metric              | 0181 step_2 | 0182 step_4 | Δ |
|---------------------|------------:|------------:|---:|
| accuracy            | 0.842       | 0.842       | 0.0 pp |
| in_think_rate       | 0.000       | 0.002       | ~0 |
| turn_split_rate     | 0.976       | 0.974       | −0.2 pp |
| interleaving_rate   | 0.976       | 0.974       | −0.2 pp |
| mean_split_balance  | 0.443       | 0.441       | −0.002 |
| mean_total_tokens   | 12078       | 12245       | +1.4% (worse) |
| efficiency_factor   | 0.455       | 0.449       | −0.006 |
| primary_score       | 0.2746      | **0.2703**  | **−0.004** |

**Status**: `discard` — concise teacher prompt is essentially neutral.
2 more LoRA steps with the new prompt produced identical metrics.

**Why this didn't work**:
1. The privileged-info teacher is *already* writing long traces by
   intrinsic policy behavior — appending "be concise" wording isn't
   strong enough to override the fact that the teacher has the answer
   and a placement directive.
2. KL_penalty_coef=1.0 is wholesale distribution replacement; small
   prompt nudges to the teacher don't survive that.
3. Possibly the rollout-length saturation comes from the 0170
   state-aware ack throttle (kicks at 4 calls): once 4 calls have
   happened, the student gets "finalize" and must spend tokens
   writing the answer. With the new "between steps" cadence pattern
   distilled in, that 4-call ceiling locks in a long structure.

**Next idea (knob 1)**: lower `kl_penalty_coef` from 1.0 to 0.25.
A gentler nudge toward the teacher distribution should let the
student keep more of its native (shorter) reasoning while still
acquiring the placement pattern. Same recipe otherwise; reset to
default TEACHER_SUFFIX_A (revert the conciseness addition since
it doesn't help).

Train 4 steps × batch=8 with `kl_penalty_coef=0.25`, eval, compare.
