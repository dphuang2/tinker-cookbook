# 0183 opsd-kl025 (knob 1 from 0181 — discard)

**Hypothesis**: lowering `kl_penalty_coef` from 1.0 to 0.25 will be
a gentler nudge toward the teacher, letting the student keep more
of its native (shorter) reasoning while still acquiring the
placement pattern.

**Diff**: only `kl_penalty_coef=0.25` on command line.

**Training (4 steps, batch=8)**:
```
step 1: n_datums=7 mean_lp=-0.542
step 2: n_datums=6 mean_lp=-0.525
step 3: n_datums=7 mean_lp=-0.444
step 4: n_datums=7 mean_lp=-0.462
```

**Eval (step_4 vs 0181 step_2 with kl_coef=1.0)**:
| metric              | kl=1.0 (0181) | kl=0.25 (0183) | Δ |
|---------------------|--------------:|---------------:|---:|
| accuracy            | 0.842         | **0.814**      | **−2.8 pp** |
| in_think_rate       | 0.000         | 0.000          | 0 |
| turn_split_rate     | 0.976         | 0.972          | −0.4 pp |
| mean_split_balance  | 0.443         | 0.429          | −0.014 |
| mean_total_tokens   | 12078         | 12359          | **+2.3% (worse)** |
| efficiency_factor   | 0.455         | 0.445          | −0.010 |
| primary_score       | 0.2746        | **0.2567**     | **−0.018** |

**Status**: `discard` — kl_coef=0.25 is *worse* on every metric.

**Why this didn't help**:
1. The teacher's privileged-info distribution intrinsically produces
   long traces — KL just controls how much the student moves toward
   the teacher, not what the teacher is.
2. Lower KL means the student spends more steps drifting weakly,
   gets stuck in a worse local optimum at step 4 (mean_lp -0.462)
   than the high-KL version (mean_lp ≈ -0.42 in 0182).
3. **Both knobs 1 (lower KL) and 2 (concise teacher) failed.** The
   problem is structural: distilling from a privileged-info teacher
   that knows the answer produces a student trained to write verbose
   justifications.

**Next idea — try teacher mode B (placement_only, no answer)**:
The teacher sees the same problem as the student plus only the
placement directive. No ground-truth answer. The teacher's
distribution will be much closer to the student's, so:
- Pro: traces should be natural-length (no "I know the answer, let
  me explain" bias)
- Con: teacher might not differ enough from student to produce
  useful KL signal

This is the cleanest next experiment because it tests a different
hypothesis (is the issue distribution mismatch from privileged info,
or fundamental to OPSD?). And we already have it implemented —
just need `teacher_mode=placement_only`.
