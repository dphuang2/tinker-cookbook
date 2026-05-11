# opd-rl: on-policy distillation as a fast-start for RL

Adapted from Karpathy's `autoresearch/program.md`. The goal is to test a
specific hypothesis about On-Policy Distillation (OPD), not to maximize a
single metric. The experiment loop (try → measure → keep/discard, log in a
TSV, never stop) carries over; the metrics, the acceptance criteria, **and
the search over model pair / env design** are different. Picking the right
teacher–student pair and shaping the env are part of the research, not
prerequisites done once before the loop starts.

## Hypothesis

Given:

- a *teacher* model that already performs well on RL env `E`, and
- a *student* model that is smaller and weaker on `E`,

OPD (student samples on-policy, teacher provides per-token reverse-KL targets)
can be used to **hill-climb on `E` much faster than RL-from-scratch on the
student**, while **avoiding the catastrophic forgetting** seen with pure RL or
pure SFT on a narrow task. We then ask the harder question: after OPD,
**continued RL on the distilled student can surpass the teacher** on `E`,
because (a) the teacher is fixed and (b) the env is narrow enough that a
smaller student has enough capacity once primed.

Measured per checkpoint:

1. `task_reward` on `E` (held-out prompts)
2. `forgetting_score` on a *general* eval set (IFEval + an MMLU slice)
3. `kl_to_teacher` on rollouts from `E`
4. `vs_teacher_gap` = `student_reward − teacher_reward` on `E`

Claims pass/fail as follows:

- **Claim A (fast start):** OPD reaches teacher-level `task_reward` in
  ≲ ½ the steps that RL-from-scratch needs.
- **Claim B (no forgetting):** OPD'd student keeps `forgetting_score` within
  noise of the base student; pure-RL student drops noticeably.
- **Claim C (past the teacher):** RL continued from the OPD checkpoint reaches
  `vs_teacher_gap > 0` on `E` within the compute budget. Pure-RL student from
  scratch either does not get there or takes substantially more steps.

If only A and B hold but not C, that is still a useful result.

## Finding the right pair and env is part of the research

A negative result on any claim is **only informative if the pair and env
make that claim discoverable in the first place.** Before treating a failed
claim as evidence against the hypothesis, check that the setup actually
makes it observable:

| Claim | Setup precondition that must hold for the claim to be discoverable |
|---|---|
| A — fast start | Teacher gap is large at student init (so there's room for OPD to close), but student has capacity to learn from KL within the compute budget. If student is at near-zero reward forever, you can't see "faster than RL". |
| B — no forgetting | Student is **instruct-tuned** with non-trivial IFEval/MMLU at init (≳50% IFEval), so there is *something to forget*. Pure-RL ablation must visibly degrade those metrics on this env. If pure RL doesn't degrade general capability either, the claim is unfalsifiable on this env. |
| C — surpass teacher | Teacher is **not at the env ceiling** — i.e. teacher has not been RL-tuned on `E`, and the env is narrow enough that a specialized smaller student can plausibly exceed teacher zero-shot. If teacher is frontier-reasoning + thinking-on, Countdown-like envs are near-trivial for it and Claim C is closed off by construction. |

Iterating on the pair and the env is therefore in scope:

- **Pair search.** If the teacher gap is too small (student already
  ~teacher), grow the gap (bigger teacher, smaller student, or both). If
  the gap is so large the student can't learn anything within the budget
  (no signal from KL), shrink it. If the teacher is at-ceiling on the env,
  pick a weaker teacher (e.g. non-thinking, or smaller).
- **Env shaping.** If task reward is too sparse for RL-from-scratch to ever
  move (so claim A is trivially won by OPD on a degenerate baseline), make
  the env easier (fewer sources, smaller targets) or add a partial-credit
  signal. If task reward saturates too fast for any pair (so claims A/C
  collapse), make it harder.
- **Forgetting eval shaping.** If IFEval doesn't move under pure RL, swap
  in a more sensitive eval (e.g. a mix-of-tasks chat eval, a code eval, or
  a domain the env actively pulls away from).

Pair and env changes must be **logged as a step in `results.tsv`** with a
clear `notes` field describing why the change was made. They are not silent
rewrites — they are part of the experimental record.

## Starting point (subject to change by the loop)

These are the *first* picks, not commitments. The loop is allowed and
expected to replace them if they don't make the claims discoverable.

- **Teacher (initial):** `Qwen/Qwen3-8B` (instruct, thinking off). Plausibly
  decent on Countdown without being at ceiling.
- **Student (initial):** `Qwen/Qwen3-1.7B` (instruct). Small enough to
  iterate cheaply.
- **Env (initial):** Countdown numbers game, 4 sources, target ≤ 100,
  +,−,*,/, strict AST verifier. Solvable-by-construction sampler.
- **Forgetting eval (initial):** IFEval + a 500-question MMLU slice.

After the first two or three runs, the loop should explicitly ask "is the
current pair making each claim observable?" before continuing with
hyperparameter sweeps.

## Compute budget

Each "experiment" in the loop is a single training run capped at
**20 minutes wall-clock** (longer than Karpathy's 5 min because RL rollouts
dominate). If a run is still improving at the budget, note it and either
extend (with explicit log entry) or call it done.

## What you CAN do

- Modify anything under `experiments/opd_rl/` and the env file.
- Tune LR, LoRA rank, group size, KL coefficient, batch size, temperature.
- **Swap teacher and/or student** if the current pair doesn't make the
  hypothesis discoverable. Log the swap and the reason.
- **Reshape the env** (difficulty, source count, target range, format) if
  the reward signal is degenerate. Log the change and the reason.
- **Swap or extend the forgetting eval** if it's not sensitive enough to
  the kind of forgetting RL is producing. Log the change and the reason.
- Add ablations: off-policy SFT-from-teacher, RL-only, mixed RL+OPD,
  RL'd-teacher (strict version of Claim C).

## What you CANNOT do

- Do not change the verifier *while comparing runs against each other*.
  Reset baselines when the verifier changes.
- Do not silently change the pair or env mid-comparison; log every change.
- Do not bring chat-data replay into the OPD mixture — the whole point of
  Claim B is that OPD alone preserves general capabilities.

## Output format

Each run appends one row to `results.tsv`:

```
commit	variant	teacher	student	env_version	steps	task_reward_E	forgetting_score	kl_to_teacher	vs_teacher_gap	status	notes
```

`variant` is one of: `teacher_ref`, `student_zero_shot`, `student_rl_only`,
`student_opd`, `student_opd_then_rl`, `student_sft_off_policy`,
`teacher_rl_ref` (optional strict-version teacher). `status` is `keep` /
`discard` / `crash`. `env_version` is a short tag (e.g. `countdown-v1`,
`countdown-v2-harder`) bumped whenever the env shape changes.

## The experiment loop

LOOP FOREVER (or until manually stopped):

1. Look at the table. For each claim (A/B/C), is the current pair + env
   making it observable? If not, the next experiment is a **setup change**
   (swap teacher, swap student, reshape env, or swap forgetting eval),
   not a hyperparameter tweak.
2. Otherwise, pick the claim with the weakest current evidence and design
   the ablation that most directly stress-tests it.
3. Edit code, commit, run.
4. `grep "^FINAL:" run.log` for the summary row.
5. Append to `results.tsv`.
6. If the run changes the picture on one of A/B/C, leave the commit on
   the branch. Otherwise revert to before the experiment.
7. If you start running out of ideas, re-read the OPD blog and the
   cookbook `recipes/distillation/README.md` for unused angles
   (multi-teacher, kl_discount_factor, KL coefficient schedules), or
   reconsider whether a different pair/env would let a claim breathe.

### Stopping conditions

Unlike Karpathy's loop (pure hill-climb on val_bpb), this experiment has a
defined endpoint: once all three claims have been clearly resolved
(supported or falsified) with at least one confirmatory ablation each on a
pair+env where the claim was actually observable, write the final summary
into `experiments/opd_rl/findings.md` and stop.
