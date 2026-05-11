# opd-rl: on-policy distillation as a fast-start for RL

Adapted from Karpathy's `autoresearch/program.md`. Goal of this experiment is to
test a specific hypothesis about On-Policy Distillation (OPD), not to maximize
a single metric. The "experiment loop" idea (try → measure → keep/discard, log
in a TSV, never stop) carries over; the metrics and acceptance criteria are
different.

## Hypothesis

Given:

- a *teacher* model that already performs well on RL env `E`, and
- a *student* model that is much smaller and weak on `E`,

OPD (student samples on-policy, teacher provides per-token reverse-KL targets)
can be used to **hill-climb on `E` much faster than RL-from-scratch on the
student**, while **avoiding the catastrophic forgetting** seen with pure RL or
pure SFT on a narrow task. We then ask the harder question: after OPD,
**continued RL on the distilled student can surpass the teacher** on `E`,
because (a) the teacher is fixed and (b) the env is narrow enough that the
small student has enough capacity once primed.

Concretely we want to measure four numbers across training:

1. `task_reward` on `E` (held-out prompts)
2. `forgetting_score` on a *general* eval set (IFEval + an MMLU slice)
3. `kl_to_teacher` on rollouts from `E`
4. `vs_teacher_gap` = `student_reward − teacher_reward` on `E` (negative means
   student is still below teacher)

The three claims succeed/fail as follows:

- **Claim A (fast start):** OPD reaches teacher-level `task_reward` in
  ≲ ½ the steps that RL-from-scratch needs.
- **Claim B (no forgetting):** OPD'd student keeps `forgetting_score` within
  noise of the base student; pure-RL student drops noticeably.
- **Claim C (past the teacher):** RL continued from the OPD checkpoint reaches
  `vs_teacher_gap > 0` on `E` within the compute budget. Pure-RL student from
  scratch either does not get there or takes substantially more steps.

If only A and B hold but not C, that is still a useful negative result.

## Setup

Models (committed up front, do not change mid-run):

- **Teacher**: `Qwen/Qwen3-8B` (instruct). Strong on compositional reasoning,
  cheap enough to log-prob on Tinker at LoRA-free inference cost.
- **Student**: `Qwen/Qwen3-1.7B` (instruct). Small enough that a single OPD
  step is cheap; weak enough on compositional reasoning that the teacher gap
  on Countdown is large. We deliberately start from the **instruct** model,
  not the base model: Claim B (no catastrophic forgetting on IFEval/MMLU) is
  only meaningful if the student starts with non-trivial instruction-
  following and general-knowledge capability that can be lost. A base model
  scores near zero on IFEval, so "preservation" would be vacuous.

RL environment: **Countdown numbers game.** Given a target integer and 4–6
source integers, the model must produce an arithmetic expression using each
source at most once that evaluates to the target. Reward = 1.0 if the
expression parses, uses only allowed sources, and equals the target; small
format-bonus otherwise. Reasons this env is the right test bed:

- *Cheap, unambiguous verifier* — no LLM-judge noise.
- *Compositional* — needs multi-step reasoning that a 1.7B model is genuinely
  bad at, so the teacher gap is real.
- *Narrow* — the env is small enough that a 1.7B student plausibly has the
  capacity to *exceed* an 8B model after enough RL, which is what makes
  Claim C testable.
- *Not in the OPD blog* — the blog used DeepMath/AIME, so Countdown is a
  genuine OOD test of the OPD recipe.

Forgetting eval: IFEval (instruction following) + a 500-question MMLU slice
(general knowledge). Run before training and at each checkpoint.

Compute budget: each "experiment" in the loop is a single training run capped
at **20 minutes wall-clock** (longer than Karpathy's 5 min because RL rollouts
dominate). If a run is still improving at the budget, note it and either
extend (with explicit log entry) or call it done.

## What you CAN do

- Modify anything under `experiments/opd-rl/` and the new Countdown env file.
- Tune LR, LoRA rank, group size, KL coefficient, batch size.
- Swap student size *within Qwen3 family* (0.6B / 1.7B / 4B) but log the
  change. Do NOT swap teacher; the teacher is the fixed reference.
- Add ablations: off-policy SFT-from-teacher baseline, RL-only baseline,
  mixed RL+OPD.

## What you CANNOT do

- Do not change the verifier in the Countdown env (it is the ground-truth
  metric).
- Do not change the forgetting eval (IFEval + MMLU slice are ground truth for
  Claim B).
- Do not bring in chat data into the OPD mixture — the whole point of Claim B
  is that OPD alone, with no replay, preserves general capabilities.

## Output format

Each run writes a single results row plus a per-step JSONL log. Per-row fields:

```
commit  variant  steps  task_reward_E  forgetting_score  kl_to_teacher  vs_teacher_gap  status  notes
```

`variant` is one of: `teacher_ref`, `student_zero_shot`, `student_rl_only`,
`student_opd`, `student_opd_then_rl`, `student_sft_off_policy`. `status` is
`keep` / `discard` / `crash`.

Header for `experiments/opd-rl/results.tsv`:

```
commit	variant	steps	task_reward_E	forgetting_score	kl_to_teacher	vs_teacher_gap	status	notes
```

## The experiment loop

LOOP FOREVER (or until manually stopped):

1. Decide which of the three claims is currently weakest in evidence. Pick the
   ablation that most directly stress-tests it.
2. Edit code, commit, run.
3. `grep "^FINAL:" run.log` for the summary row.
4. Append to `results.tsv`.
5. If the run *changes the picture* on one of A/B/C, leave the commit on the
   branch. Otherwise revert to before the experiment.
6. If you start running out of ideas, re-read the OPD blog post and the
   tinker-cookbook `recipes/distillation/README.md` for unused angles
   (multi-teacher, kl_discount_factor, KL coefficient schedules).

### Stopping conditions

Unlike Karpathy's loop (pure hill-climb on val_bpb), this experiment has a
defined endpoint: once all three claims have been clearly resolved
(supported or falsified) with at least one confirmatory ablation each, write
the final summary into `experiments/opd-rl/findings.md` and stop.
