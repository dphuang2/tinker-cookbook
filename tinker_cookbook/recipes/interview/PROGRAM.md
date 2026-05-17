# progress-update autoresearch

This is an experiment to have the LLM autonomously iterate on a recipe
that trains a model to interleave `progress_update`-style tool calls in
the middle of its thinking on DeepMath problems. The entire recipe —
training method, data generation, tool design, agent loop — is in scope.

## Goal

**Primary**: maximize agent-eval accuracy on the held-out 500
DeepMath problems. The current best (`v3`) is **0.708**; the no-SFT
baseline is **0.880**. Goal is to close that gap.

**Secondary**: minimize the amount of training data needed to reach a
given accuracy. A recipe that hits 0.85 with 500 records is preferred
over one that hits 0.85 with 2400 records. Always record the
training-record count for the experiment in `notes.md` and the
`results.tsv` description.

**Tertiary**: keep tool-call cadence non-degenerate (some problems
emit 1+ calls, some emit 0 — the model should *vary* by problem, not
always 0 or always 3).

## What is fixed

Almost nothing. The hard constraints are:

1. **Behavioral goal**: the trained model must interleave a
   tool-call-style "pause and report progress" mechanism in the
   middle of its thinking. The tool's name, schema, and what the
   tool response contains are up to you — but the *behavior* must be
   recognizable: the model pauses its thinking, emits a tool call,
   the agent loop responds, and the model continues toward the final
   answer.
2. **Eval slice**: indices 0-499 of `zwhe99/DeepMath-103K` shuffled
   with seed=42. Never train on those indices; always grade on
   exactly those.
3. **Grading**: `extract_boxed` + `grade_answer` (from
   `tinker_cookbook.recipes.math_rl.math_grading`) on the final
   visible answer. Pass/fail. Don't invent a softer metric.
4. **Eval temperature / max_tokens**: keep them at the current values
   in `eval_deepmath_agent.py` (`temperature=0.6`,
   `max_tokens_per_turn=8192`, `max_turns=8`) so accuracy numbers
   stay comparable across experiments. If you have a strong reason
   to change one, do it once, document it loudly, and recognize the
   experiment is no longer comparable to earlier rows.
5. **Base model**: Qwen3-30B-A3B (so all rows compare to the same
   `0.880` baseline).

Everything else — training method (SFT / RL / DPO / prompt-only),
teacher rewriting on/off, which teacher, tool name and schema, system
prompt phrasing, renderer configuration, agent-loop structure (as
long as it faithfully feeds tool responses back), training data
composition and size — is open.

## Editable scope

You can edit anything under `tinker_cookbook/recipes/interview/` and
add new files there. You may **not** edit anything outside that
directory (the renderer, the supervised training core, the eval
grading utility — all out of scope) and may not add new package
dependencies.

If you change which inputs flow into training (e.g. you edit
`teacher_rewrite.py` or `sample_deepmath_train.py`), **regenerate the
corresponding artifact before retraining** and note "regenerated
dataset" in the description.

If you change the *format* of progress updates (rename the tool, add
a new arg), update `eval_deepmath_agent.py` accordingly so it parses
the new format — but never change the eval slice, grading, or the
fixed eval sampling params.

**Simplicity criterion**: simpler is better when results are tied. A
small accuracy gain from a one-line LR change beats the same gain
from 50 lines of brittle data-munging. Deleting code while holding
accuracy flat is a win.

## Setup (run once, before the loop)

1. **Run tag**: today's date (e.g. `may18`).
2. **Branch**: `git checkout -b autoresearch/progress-update-<tag>`
   from current main. All experiments commit on this single branch.
3. **Read** PROGRAM.md (this file), then skim `sft_train.py`,
   `eval_deepmath_agent.py`, `teacher_rewrite.py`,
   `sample_deepmath_train.py`. (The first iteration of the loop will
   read them in depth.)
4. **Verify artifacts exist**:
   - `/tmp/tinker-examples/interview/sft_dataset.json` — 2401 records,
     current SFT dataset. Regenerable via `teacher_rewrite.py`.
   - `/tmp/tinker-examples/interview/deepmath_train_traces.json` —
     2500 raw Qwen3 traces (DeepMath indices 500-2999). Regenerable
     via `sample_deepmath_train.py`.
   - `/tmp/tinker-examples/interview/deepmath_eval.json` — baseline
     n=500 single-turn eval at accuracy 0.880.
5. **Initialize** `tinker_cookbook/recipes/interview/results.tsv`
   with header `commit	accuracy	cadence	test_nll	training_records	status	description`,
   and one row for the v3 baseline:

   ```
   <current-HEAD-short-hash>	0.7080	0:342,1:137,2:18,3:2,4:1	0.109	2301	keep	v3 baseline: reasoning-in-tool-call, LR≈5e-4, lora_rank=32
   ```

   (2301 = 2401 records − 100 held out as the SFT test set.)
6. **Initialize** `tinker_cookbook/recipes/interview/runs/` (empty
   tracked dir; commit a `.gitkeep` if needed).
7. **Commit** the two new files as `autoresearch setup`.

Once setup is done, enter the loop below and don't stop.

## The experiment loop

The branch advances linearly — **every experiment commits**, including
`keep`, `discard`, and `crash`. Never `git reset` to roll back. The
branch history *is* the experimental record.

LOOP FOREVER:

1. **Read state**: latest few `results.tsv` rows + the most recent
   `runs/<NNNN>-*/notes.md` to see what was just tried and what
   pattern is emerging.
2. **Form a hypothesis**. Pick something from "Ideas to try" below
   (avoiding ideas already tested) or invent a new one. One change
   per experiment when possible (easier attribution).
3. **Revert if needed**: if the previous experiment was `discard` or
   `crash` and its changes are still in the working tree, edit the
   relevant files back to the previous `keep` state before layering
   on the new change. Do this by editing, not by `git reset`.
4. **Make the edit** under `recipes/interview/`. If you changed
   data-generation code, regenerate the artifact:
   - Teacher prompt / cadence / validation changed → re-run
     `uv run python -m tinker_cookbook.recipes.interview.teacher_rewrite`.
   - Raw Qwen3 trace generation changed → re-run
     `uv run python -m tinker_cookbook.recipes.interview.sample_deepmath_train`
     and then `teacher_rewrite.py`.
5. **Train**:
   ```
   rm -rf /tmp/tinker-examples/interview/sft_run
   LD_PRELOAD=/work/dylan/Git/dylan-workspace-6/tinker-cookbook-dylan/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 \
       uv run python -m tinker_cookbook.recipes.interview.sft_train \
       > /tmp/dylan/sft_run.log 2>&1
   ```
   (If you swap to a non-SFT training method, point this at the
   appropriate entrypoint.) Use `>` not `tee` — don't let log
   content flood your context.
6. **Sanity check the training**: confirm a `final` row landed in
   `/tmp/tinker-examples/interview/sft_run/checkpoints.jsonl`. If
   not, `tail -n 80 /tmp/dylan/sft_run.log` for the traceback and
   decide fix-and-retry vs. log a `crash` row and move on.
7. **Eval**:
   ```
   LD_PRELOAD=... uv run python -m tinker_cookbook.recipes.interview.eval_deepmath_agent \
       > /tmp/dylan/eval_run.log 2>&1
   ```
8. **Read the summary**:
   ```
   jq -r '.summary' /tmp/tinker-examples/interview/deepmath_agent_eval.json
   ```
9. **Stage artifacts** in `runs/<NNNN>-<slug>/`:
   - `<NNNN>` = next zero-padded sequential index (`0001`, `0002`...).
   - `<slug>` = short kebab-case description.
   - Files: `summary.json` (eval summary), `results.json` (full
     eval JSON with per-problem trajectories), `metrics.jsonl` (copy
     of SFT metrics, if applicable), `sft.log.tail` (last 200 lines),
     `eval.log.tail` (last 200 lines), `config.json` (copy of SFT
     config dump, if applicable), `checkpoints.jsonl` (copy — gives
     the Tinker `sampler_path` so the run is re-evaluatable),
     `notes.md` (hypothesis + one-line diff summary + result + next
     idea).
   - Keep total per-experiment artifacts under ~10 MB to avoid
     branch bloat. Full per-problem eval JSON for 500 problems is
     typically a few MB.
10. **Append a row** to `results.tsv` (tab-separated, 7 columns:
    `commit	accuracy	cadence	test_nll	training_records	status	description`).
    Use `pending` in the commit column; fill the hash in on the
    next iteration's commit (or amend, your choice — but don't
    block on filling it in).
11. **Commit**: `git add -A && git commit -m "<NNNN> <slug>: <status>: <result>"`.
    Example: `0002 lr-1p5e4: keep: 0.7080 -> 0.7820`.
12. **(Optional) Push** if a remote is configured. Don't block on
    missing credentials.
13. Loop back to step 1.

### Status semantics

`status` is a **label**, not a branch operation. Every experiment
commits regardless.

- `keep` — the experiment looks promising. The next experiment
  layers on top of this `sft_train.py` (or whichever files changed)
  state.
- `discard` — the experiment underperformed. The *next* commit
  should revert the causal lines back to the previous `keep` state
  before adding new changes.
- `crash` — the run never produced an eval. Same handling as
  `discard` if the change was causal; otherwise leave the file
  alone (e.g. a config typo that's already been fixed).

## Output format reference

After eval, `summary` looks like:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "sampler_path": "tinker://.../sampler_weights/final",
  "num_problems": 500,
  "num_correct": 354,
  "accuracy": 0.708,
  "tool_call_cadence": [[0, 342], [1, 137], [2, 18], [3, 2], [4, 1]],
  "temperature": 0.6,
  "max_tokens_per_turn": 8192,
  "max_turns": 8
}
```

`tool_call_cadence` format for `results.tsv`: serialize as
`0:342,1:137,2:18,3:2,4:1` (one comma-joined key:value list, no
spaces).

`training_records` for `results.tsv`: count of training datums you
trained on this run (post any filtering). Use `-` for non-SFT
methods or `0` for prompt-only baselines.

Example `results.tsv`:

```
commit	accuracy	cadence	test_nll	training_records	status	description
a1b2c3d	0.7080	0:342,1:137,2:18,3:2,4:1	0.109	2301	keep	v3 baseline reasoning-in-tool-call
b2c3d4e	0.7820	0:280,1:180,2:35,3:5	0.142	2301	keep	drop LR to 1.5e-4
c3d4e5f	0.7050	0:330,1:140,2:25,3:5	0.105	2301	discard	add empty-thinking final turns (reverted next)
d4e5f6g	0.0000	-	-	-	crash	max_length=65536 OOM (reverted to 32768)
e5f6g7h	0.8050	0:300,1:160,2:32,3:8	0.155	500	keep	subsample 500 records + LR 1.5e-4 (data-efficient!)
```

## Operational notes

- **LD_PRELOAD is required.** The system NCCL on `LD_LIBRARY_PATH`
  is older than what torch 2.12 wants. Every training/eval command
  must be prefixed with the LD_PRELOAD shown in step 5/7 above.
  Export it once at the start of the loop to avoid repeating:
  ```
  export LD_PRELOAD=/work/dylan/Git/dylan-workspace-6/tinker-cookbook-dylan/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
  ```
- **Cycle time**: typical SFT cycle ≈ 25-35 min (15 min train,
  10-15 min eval). RL or data regeneration cycles are longer.
- **Timeout**: if a single train or eval phase exceeds 1 hour, kill
  it and treat as a crash.
- **Monitoring**: use a tight Monitor filter on
  `/tmp/tinker-examples/interview/sft_run/metrics.jsonl` while
  training (steps % 10 + errors only). Stop the monitor before the
  next cycle.
- **NEVER STOP**: once the loop has begun, do not pause to ask "is
  this a good stopping point?" The human may be asleep. Continue
  indefinitely until manually interrupted. If you run out of ideas,
  re-read prior `notes.md` for patterns, combine prior near-misses,
  or take a more radical swing.

## Ideas to try (seed list — non-exhaustive; exhaust your own ideas too)

### Training knobs (cheapest — try these first)

- **Lower LR** — current ≈ `get_lr(Qwen3-30B-A3B, lora=True)` ≈ 5e-4.
  Try 1.5e-4, 5e-5. The 14pp regression strongly suggests
  over-aggressive LR damaging base reasoning.
- **Early stopping** — re-eval an intermediate sampler from
  `checkpoints.jsonl` (e.g. step 80 or 100) instead of `final`. Mid
  training may preserve more baseline capability.
- **Smaller `lora_rank`** — try 8 or 16 from 32.
- **Fewer epochs / fewer records** — directly addresses the
  data-efficiency secondary goal: subsample to 500-1000 records and
  see if accuracy holds.

### Data composition

- **Mix in pure-math SFT data** — half the batch with tool calls,
  half plain `<think>` + final answer, so the model retains the
  "just answer" mode. Source:
  `/tmp/tinker-examples/interview/deepmath_train_traces.json`.
- **Filter out short traces** from the teacher dataset — records
  whose original thinking was < 4000 chars probably shouldn't be
  forced to have tool calls.
- **Drop the duplicated `reasoning` arg** — revert to `message`-only
  summary. v1 hit 0.736 with that; worth retesting once other
  knobs are tuned.

### Teacher rewrite

- **Lower the teacher cadence target** — current is ~1 update /
  1000 thinking tokens, cap 3. Try ~1/2000, cap 2.
- **Different teacher** — swap Kimi-K2.6 for another Tinker model,
  or skip the teacher and use rule-based splitting (e.g. split on
  paragraph boundaries every K tokens).
- **Self-distillation** — prompt Qwen3 directly with the tool
  exposed, collect natural tool-using traces, filter to correct
  answers, use as SFT data. No external teacher required.

### Format / tool design

- **System prompt tweak** — "between major reasoning steps" vs
  "only when you change approach" vs "only for uncertain problems".
- **Tool renaming** — `progress_update` → `checkpoint` /
  `note_to_self`, with a description that makes it clear this is
  for the model's bookkeeping, not a user-facing report.
- **No tool, use special tokens** — `<update>...</update>` inside
  the `<think>` block. No agent loop needed at eval. Different
  preservation mechanics.

### Method changes (most radical)

- **Prompt-only baseline** — skip training, expose the tool in the
  prompt, measure zero-shot accuracy. Useful upper bound.
- **DPO** — preference pairs of (good-cadence trace, bad-cadence
  trace) for the same problem.
- **RL with format + correctness reward** — bootstrap with the
  existing `tinker_cookbook.recipes.math_rl` machinery. Format bonus
  for emitting `progress_update` at the right cadence; correctness
  reward dominates.

As an example use case: a user might leave you running while they
sleep. At ~30 min per cycle that's ~16 experiments overnight. The
user wakes up to a populated `results.tsv` and (hopefully) a `keep`
row with accuracy back above 0.85.
