# progress-update autoresearch — v2 (interleaving-aware)

This is an experiment to have the LLM autonomously iterate on a recipe
that teaches a model to **interleave `progress_update` tool calls
*inside* its thinking** on DeepMath problems. The entire recipe —
training method, data generation, tool design, agent loop, reward
shape — is in scope.

## Lessons from v1 (read this before doing anything)

The first 159-experiment loop converged on a prompt-only recipe
(`0105`) that hits 0.8750 ± 0.010 mean accuracy across 26 reruns —
statistically indistinguishable from the no-tool baseline (0.880).

**But it never actually solved the behavioral goal.** A post-hoc
audit of the raw rollouts (`raw_rollouts/`, `interleaved_rollouts.md`)
found:

- **234 / 238 tool-using rollouts (98%)** emit *all* their tool
  calls **after** `</think>` closes, as a back-to-back batch of
  summaries, followed by the final answer. The thinking pass is
  monolithic; the "checkpoints" are decorative post-hoc summaries.
- **Only 10 / 500 rollouts (2%)** had tool calls split across ≥2
  separate assistant turns (the cross-turn proxy for interleaving).
- **Re-sampling those same 10 indices at temp 0.6** reproduced the
  interleaved shape in only **1 / 40** attempts. So even on the
  "easy" indices, true interleaving is a ~2.5% accident, not a
  learned behavior.

### Why the v1 loop didn't catch this

The eval only measured **`num_tool_calls` per problem** (the
`tool_call_cadence` histogram) and **`extract_boxed` accuracy** on
the final visible text. Neither signal distinguishes:

- "think once, dump 3 summaries at the end, then answer" (the
  trained behavior — easy to learn from Qwen3's prior because the
  chat template puts `<tool_call>` *outside* `<think>` blocks)
- "think → pause → call → think → call → think → answer" (the
  intended behavior — fights the chat-template prior)

Both shapes score identically on cadence count and accuracy.
Goodhart did the rest: every "winning" recipe optimized the proxy.

### What v2 must do differently

1. **Measure placement, not just count.** Add an
   `interleaving_rate` metric to the eval summary that counts the
   fraction of rollouts where tool calls actually pause the
   thinking — see "Interleaving metric" below.
2. **Treat interleaving as a first-class objective**, not a tertiary
   nicety. A recipe that scores 0.85 with 40% interleaving beats a
   recipe that scores 0.875 with 2% interleaving.
3. **Use RL.** Successful interleaved trajectories already exist at
   ~2.5% frequency from the prompt-only base. That's enough signal
   to amplify with on-policy RL using a placement-aware reward.
   SFT on rare positives risks the same prior-overfit failure as v1.

## Goal

**Primary (v2.1)**: maximize a **joint metric** that rewards
correctness AND true interleaving AND avoids two Goodhart escapes
(running the CoT N times to look "interleaved"; clustering all
checkpoints at one end of the rollout):

```
primary_score = accuracy
              × (0.5 + 0.5 × interleaving_rate × mean_split_balance)
              × efficiency_factor

where:
  efficiency_factor = clamp(NO_TOOL_REF_TOKENS / mean_total_tokens, 0, 1)
                    # 1.0 if at-or-below baseline; 0.5 if 2× tokens; etc.
  mean_split_balance = mean over rollouts with ≥2 tool calls of
                       (min_segment / max_segment)
                    # 1.0 = checkpoints divide the rollout into equal
                    # parts; ~0 = all batched at the end (the v1 mode).
```

v1 best (0105) under v2.1 scores ≈ `0.87 × (0.5 + 0.5 × 0.02 × 0.0) ×
1.0 ≈ 0.44`. A recipe at accuracy 0.80, interleaving 0.60, balance
0.5, efficiency 1.0 scores `0.80 × (0.5 + 0.5 × 0.60 × 0.5) × 1.0 =
0.520`. A "fake interleaved" recipe doing the CoT 3× (efficiency
factor ≈ 0.33) is heavily penalized.

**Secondary**: minimize training cost. A recipe matching the primary
metric with 0 training records or fewer RL steps wins ties.

**Tertiary**: cadence variety per problem — the model should decide
*per problem* whether and how often to checkpoint, not always 0,
always 3, or always N.

### Required per-rollout instrumentation (v2.1)

For each rollout:

- `in_think_calls`: count of `<tool_call>` markers appearing
  **before** the first `</think>` close in the decoded stream of
  any assistant turn (strictest definition — true mid-thinking
  interleaving; tends to be ~0 due to Qwen3's chat-template prior).
- `n_turn_splits`: number of distinct assistant turns containing
  ≥1 tool call (cross-turn proxy for interleaving).
- `total_tokens`: tokens consumed across all turns (for efficiency).
- `total_chars`: chars across all turns (for split_balance).
- `tool_call_char_positions`: char position of every `<tool_call>`
  marker in the concatenated decoded stream.
- `split_balance`: for rollouts with ≥2 tool calls, the ratio
  `min_segment / max_segment` over the K+1 segments formed by the
  K markers (with implicit boundaries at 0 and total_chars). `None`
  for rollouts with <2 calls (excluded from aggregate).
- `is_interleaved`: `True` iff `in_think_calls ≥ 1` OR
  `n_turn_splits ≥ 2`.

Aggregate (in `summary`):
- `interleaving_rate` = fraction of all 500 rollouts that are
  interleaved.
- `mean_split_balance` = mean of `split_balance` over rollouts
  where it's defined.
- `mean_total_tokens` = mean tokens per rollout.
- `efficiency_factor` = clamp(`NO_TOOL_REF_TOKENS` /
  `mean_total_tokens`, 0, 1). `NO_TOOL_REF_TOKENS` is set to 5500
  (current rough no-tool baseline; re-baseline if you change the
  prompt-only ceiling).
- `primary_score` = formula above.

## What is fixed

Almost nothing. The hard constraints are:

1. **Behavioral goal (now measurable)**: the model must
   *interleave* tool calls with thinking — not batch them after.
   This is now operationalized by the `interleaving_rate` metric
   defined above. A recipe that scores high accuracy but
   `interleaving_rate < 0.1` fails the primary metric by
   construction.
2. **Eval slice**: indices 0-499 of `zwhe99/DeepMath-103K` shuffled
   with seed=42. Never train on those indices; always grade on
   exactly those.
3. **Grading**: `extract_boxed` + `grade_answer` (from
   `tinker_cookbook.recipes.math_rl.math_grading`) on the final
   visible answer. Pass/fail on correctness. Combined with
   `interleaving_rate` into the primary score (see Goal section).
4. **Eval temperature / max_tokens**: keep them at the v1
   end-of-loop values (`temperature=0.6`,
   `max_tokens_per_turn=24576`, `max_turns=8`). Don't change these
   in v2 — comparability with v1's 0.875 and the 0.880 baseline
   matters.
5. **Base model**: Qwen3-30B-A3B (so all rows compare to the same
   `0.880` no-tool baseline and the v1 `0.875` prompt-only
   ceiling).
6. **Eval must compute `interleaving_rate`** and include it in
   `summary` and `results.tsv`. Without this metric, an experiment
   is not comparable to v2 rows.

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

## Setup for v2 (run once, before the loop)

1. **Run tag**: today's date (e.g. `may19`).
2. **Branch**: keep working on the existing
   `autoresearch/progress-update-may17` branch (v1 history is
   preserved; v2 layers on top). All experiments commit linearly.
3. **Read** this PROGRAM.md end-to-end, plus
   `interleaved_rollouts.md` (the 10 v1 examples that interleaved),
   plus a few of the `raw_rollouts/idx*_s*.json` files to see the
   token-level shape of batched vs interleaved trajectories.
4. **Instrument `interleaving_rate` in
   `eval_deepmath_agent.py`** before the first experiment. The
   eval loop already has access to the decoded token stream per
   turn — compute `n_calls_in_think` from the raw text (search
   for `<tool_call>` before `</think>`) and `n_turn_splits` by
   counting turns where `len(tool_calls) > 0`. Persist both per
   problem and aggregate `interleaving_rate` in the `summary`.
5. **Re-run the v1 ceiling recipe (`0105`) under the new metric**
   to establish the v2 baseline. Expected: accuracy ≈ 0.875,
   interleaving_rate ≈ 0.02–0.05, primary score ≈ 0.45.
6. **Re-run the no-tool baseline** for sanity. Expected:
   accuracy ≈ 0.880, interleaving_rate = 0 (no tool exposed),
   primary score = 0 (filtered to 0 because no tool use).
7. **Commit** the eval instrumentation and the two baselines as
   `v2 setup: instrument interleaving_rate + re-baselines`.

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

## Output format reference (v2)

After eval, `summary` must include the new fields:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "sampler_path": "tinker://.../sampler_weights/final",
  "num_problems": 500,
  "num_correct": 437,
  "accuracy": 0.874,
  "tool_call_cadence": [[0, 262], [1, 3], [3, 206], [4, 16]],
  "interleaving_rate": 0.028,
  "in_think_rate": 0.000,
  "turn_split_rate": 0.028,
  "primary_score": 0.4438,
  "temperature": 0.6,
  "max_tokens_per_turn": 24576,
  "max_turns": 8
}
```

Where:
- `in_think_rate` — fraction of rollouts with ≥1 tool call
  inside `<think>...</think>`
- `turn_split_rate` — fraction with tool calls across ≥2 turns
- `interleaving_rate` — fraction satisfying either (union)
- `primary_score = accuracy × (0.5 + 0.5 × interleaving_rate)`

`results.tsv` v2 header (extend, don't break v1 — add columns):

```
commit	accuracy	interleaving_rate	primary_score	cadence	training_records	rl_steps	status	description
```

`training_records` is `0` for prompt-only / RL-from-base. `rl_steps`
is the number of RL gradient steps for RL experiments, `-` for SFT.

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

## Ideas to try (v2 — interleaving-first)

### Start here: RL with placement-aware reward (the main bet)

The natural fit. Successful interleaved trajectories already exist
at ~2.5% frequency from the prompt-only base, so on-policy RL has
real positive signal to amplify. Avoids SFT's chat-template
prior-overfit failure mode that killed v1.

Recommended seed configuration:

- **Base policy**: Qwen3-30B-A3B with the v1 `0105` prompt
  (`SYSTEM_PROMPT` + `USER_INSTRUCTION_SUFFIX`). This already gives
  ~2% interleaved positives — that's the starting distribution
  RL will sharpen.
- **Reward shape**: per-trajectory scalar
  ```
  r = is_correct * (0.5 + 0.5 * is_interleaved)
  ```
  where `is_interleaved` is the binary version of the metric. So
  correct-and-interleaved = 1.0, correct-but-batched = 0.5, wrong
  = 0. Don't reward interleaving on wrong answers — that just
  trains tool-spam.
- **Reward variants to try**:
  - Continuous: scale by `min(n_calls_in_think, 3) / 3`.
  - Anti-spam: penalty `-0.05 * max(0, n_total_calls - 5)` to
    discourage runaway cadence.
  - Format-only first: train briefly with `r = is_interleaved`
    alone to verify placement is learnable, then mix in
    correctness.
- **Algorithm**: `tinker_cookbook.recipes.math_rl` machinery.
  Group size 4–8, KL penalty to base policy (LoRA RL), LR
  modest (1e-6 to 5e-6 — small steps to preserve reasoning).
- **Group semantics**: each group is one DeepMath problem
  sampled `group_size` times. Centering advantages within the
  group already exploits the rare-interleaved-positive signal.
- **Reference policy**: use the base model (no SFT) as KL
  reference to keep reasoning intact.

### Prompt-only knobs (still worth a swing)

- **Stronger placement directive**: explicit "emit your first
  `progress_update` after roughly 1000 tokens of thinking, before
  finishing your derivation."
- **Demonstration in system prompt**: a tiny worked example
  showing think → call → think → call → answer in-context. May
  unlock the shape without RL.

### Renderer / format hacks (if RL stalls)

- **Custom renderer that allows `<tool_call>` inside `<think>`** —
  the Qwen3 chat template treats tool calls as
  post-thinking by default. A small renderer override that
  permits in-think tool calls and re-tokenizes accordingly might
  be the only way to get true within-think placement.
- **Special-token alternative**: `<checkpoint>...</checkpoint>`
  inside `<think>`. No agent loop needed; placement is purely a
  text-pattern. Different definition of "interleaved" but matches
  the spirit of the goal.

### Out-of-scope for v2 (already tried, regressed in v1)

- SFT on rewritten teacher traces — every attempt collapsed
  cadence and/or hurt reasoning. Don't retry without a fresh
  hypothesis about why this time is different.
- Increasing `MAX_TOOL_RECORDS` past 0 — confirmed harmful in v1
  runs 0011–0150.

As an example use case: a user might leave you running while they
sleep. RL cycles are longer than SFT (60–120 min each), so expect
~6–10 experiments overnight. The user wakes up to a populated
`results.tsv` and (hopefully) a `keep` row with `primary_score`
above 0.5 — meaning real interleaving, not just decorative
checkpoints.
