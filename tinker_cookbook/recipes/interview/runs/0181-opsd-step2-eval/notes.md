# 0181 opsd-step2-eval — FIRST PHASE A SIGNAL (mixed)

**Sampler**: `tinker://a2923ff2-...:train:0/sampler_weights/step_2`
(from 0180 — 2 LoRA steps × 7 datums via OPSD with teacher_mode=A
and the v2.1 metric live).

**Result vs v2.1 baseline (run 0172)**:
| metric              | baseline | OPSD step_2 | Δ |
|---------------------|---------:|------------:|---:|
| accuracy            | 0.862    | 0.842       | −2.0 pp |
| in_think_rate       | 0.012    | 0.000       | −1.2 pp |
| turn_split_rate     | 0.280    | **0.976**   | **+69.6 pp** |
| interleaving_rate   | 0.280    | **0.976**   | **+69.6 pp** |
| mean_split_balance  | 0.029    | **0.443**   | **+0.414 (15×)** |
| mean_total_tokens   | 6939     | 12078       | +74% (worse) |
| efficiency_factor   | 0.793    | 0.455       | −0.338 |
| **primary_score**   | **0.3444** | **0.2746** | **−0.0698** |

**Status**: `keep` — primary_score down but the placement signal is
real and unambiguous. Two LoRA steps decisively broke the chat-
template prior at the cross-turn level (turn_split_rate 0.976 — nearly
every rollout now spans multi-turn cadence). The student is now
actually splitting the CoT into roughly equal chunks
(mean_split_balance 0.443 ≈ each segment ≥ 44% of the largest).

**The tradeoff**:
- OPSD teaches "do multi-step structured reasoning with checkpoints"
  but the privileged teacher prompt elicited LONG traces (since it
  knows the answer, the teacher can afford to show full derivation).
- Distilling toward that distribution doubled token consumption.
- efficiency_factor halved → primary_score lost more than placement
  gained.

This is exactly the Goodhart escape the user predicted when adding
efficiency_factor to v2.1: "fake interleave by doing CoT 3 times".
OPSD ran into it because the teacher's privileged-prompt rollouts
naturally trend long.

**Cadence histogram (notable)**:
- Mode at exactly **4 calls (327/500 = 65%)** — sharp lock-in to the
  state-aware ack throttle (which kicks in at the 4th call).
- 132/500 at 3 calls; only 7/500 skip.
- Long tail collapsed (max=8).

**Three knobs to try next** (one per future tick):
1. **Lower `kl_penalty_coef`** to 0.5 or 0.25 — gentler nudge toward
   teacher, less wholesale distribution replacement.
2. **Tighter teacher prompt** — add "be concise; don't repeat the
   problem statement; avoid restating already-established facts" to
   the privileged suffix, so the teacher's traces are shorter.
3. **Add token-length penalty** to the OPSD advantage:
   `r = -kl - lambda * (n_tokens - baseline_tokens)+`
   directly penalizing overlong rollouts.

I'd try (2) first — cheapest and tests whether the issue is
*intrinsic* to OPSD or just *prompt design*. If the teacher can be
prompted to produce shorter traces, the student will distill toward
shorter behavior.

**Next ideal experiment**: 0182 = teacher_mode A with appended
"be concise" directive, max_steps=10, groups_per_batch=8, then eval
step_10 against v2.1.
