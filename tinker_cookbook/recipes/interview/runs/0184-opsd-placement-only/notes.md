# 0184 opsd-placement-only (teacher_mode=B; discard)

**Hypothesis**: removing the ground-truth answer from the teacher's
privileged prompt will eliminate the "I know the answer, let me
explain" verbose-trace bias, producing shorter rollouts and a more
efficient student.

**Diff**: `teacher_mode=placement_only` (teacher sees the same prompt
as the student plus only the even-split directive, no answer).

**Training (4 steps, batch=8)**:
```
step 1: n_datums=7 mean_lp=-0.594
step 2: n_datums=7 mean_lp=-0.497
step 3: n_datums=7 mean_lp=-0.428
step 4: n_datums=7 mean_lp=-0.433
```

**Eval vs 0181 (mode=A) and 0182 (mode=A+concise) and 0183 (mode=A,kl=0.25)**:
| metric              | 0181 A | 0182 A+concise | 0183 A kl=.25 | **0184 B** |
|---------------------|-------:|--------------:|--------------:|-----------:|
| accuracy            | 0.842  | 0.842         | 0.814         | **0.828**  |
| mean_split_balance  | 0.443  | 0.441         | 0.429         | **0.419**  |
| mean_total_tokens   | 12078  | 12245         | 12359         | **13028**  |
| efficiency          | 0.455  | 0.449         | 0.445         | **0.422**  |
| primary_score       | 0.2746 | 0.2703        | 0.2567        | **0.2462** |

**Status**: `discard` — worst primary_score yet.

**Critical finding**: every OPSD variant (4 runs now) converges to
nearly identical metrics — split_balance ~0.42–0.44, mean_tokens
~12–13k, primary_score ~0.25. **The privileged-info teacher is not
the problem; the placement directive itself produces verbose
rollouts.** Telling the model to "split into 3 equal chunks" causes
it to pad reasoning to fill the chunks.

**Cookbook OPSD doesn't optimize for efficiency** — the reverse-KL
loss treats every token equally regardless of whether the rollout
is too long. Three remaining options:

1. **Add explicit token-length penalty to the OPSD advantage**
   (knob 3 from 0181). Per-rollout:
   `r += -lambda * max(0, n_tokens - REF_TOKENS) / REF_TOKENS`
   added to the per-token KL advantage on a uniform spread.
2. **Switch to Phase B (RFT)** on the OPSD-pretrained model.
   The OPSD checkpoints have broken the chat-template prior
   (split_balance jumped from 0.029 to 0.44). RFT can now refine:
   sample many rollouts from OPSD step_4, filter by primary_score,
   SFT on the survivors. This is exactly the OPSD → RFT sequence
   PROGRAM.md v3 prescribes.
3. **Different placement directive** that doesn't drive verbosity —
   e.g. "checkpoint after the *first natural reasoning transition*
   you would already make" rather than "spread into 3 equal chunks".

**Decision**: skip ahead to option 2 (Phase B RFT). The OPSD
infrastructure works — it produces a model that interleaves at
split_balance 0.44 just at the cost of doubling tokens. RFT will
let us filter for accuracy+efficiency directly using the primary
metric, without needing more OPSD tuning.

The Phase A best checkpoint to bootstrap from is **0181 step_2**
(primary_score 0.2746, lowest token overhead among OPSD variants).
