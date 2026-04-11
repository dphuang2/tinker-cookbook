# golf_forecasting autoresearch

This recipe is designed for autonomous experimentation by a coding agent such as Cursor or Claude Code.

## Mission

Build a more accurate golf forecasting system.

Almost everything is flexible:

- model choice
- training approach
- prompt design
- output format
- reward function
- data pipeline
- feature engineering
- retrieval or tool use
- priors
- train, validation, and held-out definitions
- benchmark design
- recipe structure
- overall system architecture

The core objective is fixed: improve golf forecasting.

## In-Scope Files

Read these files for full context before starting:

- `tinker_cookbook/recipes/golf_forecasting/README.md`
- `tinker_cookbook/recipes/golf_forecasting/data.py`
- `tinker_cookbook/recipes/golf_forecasting/build_dataset.py`
- `tinker_cookbook/recipes/golf_forecasting/env.py`
- `tinker_cookbook/recipes/golf_forecasting/train.py`
- `tinker_cookbook/recipes/golf_forecasting/eval.py`

## Edit Scope

You may edit anything inside `tinker_cookbook/recipes/golf_forecasting/`.

You may also edit nearby shared cookbook infrastructure if it is genuinely necessary to support a better golf forecasting system, but prefer keeping changes local to this recipe.

## Research Freedom

You are explicitly allowed to:

- search the web
- call public HTTP APIs
- fetch public webpages
- discover new public data sources
- redesign the dataset schema
- change the forecasting output format
- switch models
- replace RL with SFT, DPO, distillation, prompt optimization, or hybrids
- add retrieval, tools, or external priors
- redefine the research benchmark
- add hole-by-hole scorecard data for current and historical rounds
- add tournament-specific historical statistics (course history per player)
- add player pressure profiles (leading-by-X hold/collapse rates)

You must:

- use real public data rather than hard-coded fake golf examples
- cache fetched raw data under recipe-local artifacts
- record source URLs and fetch timestamps
- prefer reproducible public sources over brittle ad hoc scraping
- log what changed and why
- log whether each change affected data, model, prompt, training, evaluation, or overall system design
- when adding scorecard, tournament-history, or pressure data, cache the raw responses under `artifacts/raw/` by source+date so reruns are reproducible

## Two Evaluation Tracks

You must maintain two evaluation tracks.

### 1. Anchor Eval

Create one frozen anchor benchmark as soon as you have a minimally viable dataset.

Rules:

- once created, the anchor eval manifest must never change
- use it as the stable apples-to-apples benchmark for long-term progress
- always record anchor metrics for every experiment

### 2. Research Eval

You may redesign the research eval at any time.

Use it to explore:

- richer data
- new task formulations
- alternative output formats
- better priors
- better metrics
- larger or more realistic datasets

You may replace or expand the research eval whenever doing so helps build a stronger forecaster.

## Setup

1. Create a fresh branch for the run, for example `autoresearch/golf-overnight`.
2. Discover public golf data sources and write a source manifest at, for example, `/tmp/golf_sources.json`.
3. Build an initial dataset:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.build_dataset \
  source_manifest_path=/tmp/golf_sources.json \
  output_dir=tinker_cookbook/example_data/golf_forecasting \
  fetch_online=true
```

1. Freeze a clean anchor eval manifest as soon as you have a minimally viable dataset.
2. Define an initial research eval setup.
3. Create an untracked `results.tsv` with header:

```tsv
commit	anchor_log_loss	anchor_brier	research_score	status	change_type	description
```

1. Establish a baseline on both anchor eval and research eval before changing anything.

## Baseline

Start with a short, bounded baseline run using your current best judgment for:

- model
- dataset
- training approach
- output format
- evaluation setup

If the current recipe implementation is not the right shape, change it.

## Experiment Loop

LOOP FOREVER:

1. Inspect the current best system and experiment history.
2. Form one concrete hypothesis.
3. Implement the change.
4. If data logic changed, rebuild or enrich the dataset.
5. If benchmark logic changed, update only the research eval, never the frozen anchor eval.
6. Train, fine-tune, prompt-optimize, distill, or otherwise update the system.
7. Run both anchor eval and research eval.
8. Record the metrics and what changed.
9. Keep the change if it improves the overall system by your judgment.
10. Always preserve the anchor metric history, even if the research eval changes.
11. Revert changes that are not worth keeping.
12. Continue immediately with the next hypothesis.

## Good Hypotheses

Prefer hypotheses with a clear reason they might help:

- better public data sources
- more complete leaderboard history
- stronger player priors
- improved normalization or entity matching
- better prompt structure
- more robust output parsing
- alternative output formats
- improved reward shaping
- better model selection
- a different training objective
- retrieval or tool-augmented forecasting
- ensemble strategies
- hole-by-hole scorecard data for the current round (see below)
- tournament-specific historical statistics (see below)
- leading-by-X pressure data per player (see below)

### Hole-by-Hole Scorecard Data

Including the full per-hole scorecard for every player in the snapshot is a high-value feature because:

- A player at -10 who birdied the last 4 holes is in very different shape from one who bogeyed the last 4 and is riding a lucky eagle from hole 3.
- Scoring momentum (streaks, recent struggles) is invisible in a summary `score_to_par` number.
- Remaining holes have different difficulty profiles; knowing which holes are left lets the model apply implicit course knowledge.

**Schema extension** — add a `holes` list to each player entry:

```json
{
  "name": "Scottie Scheffler",
  "score_to_par": -11,
  "holes_completed": 10,
  "holes": [
    {"hole": 1, "par": 4, "score": 4, "to_par": 0},
    {"hole": 2, "par": 5, "score": 4, "to_par": -1},
    {"hole": 3, "par": 4, "score": 3, "to_par": -1},
    {"hole": 4, "par": 3, "score": 2, "to_par": -1},
    {"hole": 5, "par": 4, "score": 5, "to_par": 1},
    {"hole": 6, "par": 4, "score": 4, "to_par": 0},
    {"hole": 7, "par": 4, "score": 4, "to_par": 0},
    {"hole": 8, "par": 5, "score": 4, "to_par": -1},
    {"hole": 9, "par": 4, "score": 3, "to_par": -1},
    {"hole": 10, "par": 4, "score": 4, "to_par": 0}
  ]
}
```

Also include the `course_scorecard` at the snapshot level so the model knows par, yardage, and difficulty rank for every hole:

```json
"course_scorecard": [
  {"hole": 1, "par": 4, "yards": 445, "handicap_rank": 11},
  {"hole": 2, "par": 5, "yards": 575, "handicap_rank": 15},
  ...
]
```

**Data sources:** Most public golf APIs (DataGolf, ESPN, PGA Tour JSON feeds) expose hole-by-hole scoring at the round level. Fetch the per-round scorecard for each player alongside the leaderboard snapshot.

**Prompt design:** Represent the scorecard compactly — e.g., a comma-separated list like `E -1 -1 -1 +1 E E -1 -1 E` rather than full JSON — to save tokens. Always include the remaining-holes count and par values so the model can reason about what is left.

### Tournament-Specific Historical Statistics

Course history at a specific venue is strongly predictive. A player with five top-10 finishes at Augusta is a very different prospect from a player with five missed cuts. Key data to include:

**Per-player tournament history** in each example:

```json
"tournament_history": {
  "Scottie Scheffler": {
    "years_played": 4,
    "best_finish": 1,
    "avg_finish": 8.2,
    "made_cut_rate": 1.0,
    "avg_score_to_par_r4": -1.3,
    "win_count": 1,
    "top5_count": 2,
    "top10_count": 3,
    "scoring_avg_this_course": 69.4
  }
}
```

**Course-level historical norms** at the snapshot level:

```json
"tournament_historical_context": {
  "typical_winning_score": -12,
  "avg_r4_scoring": 71.2,
  "largest_r4_comeback_to_win": 5,
  "avg_leader_after_r3_wins": 0.68,
  "course_style": "precision iron play, penalty for distance off fairway"
}
```

**Data sources:** DataGolf historical results API, ESPN historical results pages, PGA Tour stats portal. Fetch once per tournament type and cache under `artifacts/tournament_history/<tournament_id>.json`. Normalize player names consistently across years using a canonical name map.

**Why it helps:** Models without this context treat a first-time major starter the same as a five-time champion. Historical tournament records correct that blind spot at essentially zero label cost.

### Leading-by-X Pressure Data

A player leading by 5 strokes with 18 holes to play sounds comfortable, but some players consistently collapse large leads while others extend them. Encoding "how does this player perform when holding a lead" is a direct proxy for clutch performance.

**Per-player pressure profile** (add to player entries or to a global `player_profiles` dict in the snapshot):

```json
"pressure_profile": {
  "Scottie Scheffler": {
    "lead_hold_rate": 0.82,
    "win_pct_leading_after_r3": 0.79,
    "avg_r4_score_when_leading_by_1_3": 69.1,
    "avg_r4_score_when_leading_by_4_plus": 68.6,
    "blown_leads_last_5_years": 1,
    "comebacks_from_deficit_last_5_years": 3,
    "strokes_gained_pressure_situations": 0.41
  },
  "Rory McIlroy": {
    "lead_hold_rate": 0.61,
    "win_pct_leading_after_r3": 0.58,
    "avg_r4_score_when_leading_by_1_3": 70.2,
    "avg_r4_score_when_leading_by_4_plus": 72.1,
    "blown_leads_last_5_years": 4,
    "comebacks_from_deficit_last_5_years": 2,
    "strokes_gained_pressure_situations": -0.14
  }
}
```

Key fields:

| Field | Description |
|---|---|
| `lead_hold_rate` | Fraction of final-round leads converted to wins |
| `win_pct_leading_after_r3` | Win % when holding 54-hole lead |
| `avg_r4_score_when_leading_by_X` | Scoring average in final round, stratified by lead size |
| `blown_leads_last_5_years` | Count of 54-hole leads not converted |
| `strokes_gained_pressure_situations` | SG differential vs. field in head-to-head final-round duels |

**Data sources:** Compute from historical PGA Tour round-by-round results (DataGolf, ESPN, or ShotLink derivatives). Build a script under `artifacts/pressure_profiles/` that ingests historical final-round results and emits a per-player JSON. Refresh annually or when adding new seasons.

**Prompt design:** Summarize concisely — e.g., `"Rory: leads after R3 = 58% win rate, has blown 4 leads in 5 years"` — rather than dumping the full profile JSON. Let the model use it as context when interpreting the current leaderboard.

**Why it helps:** Aggregate scoring stats are symmetric and don't distinguish between players who clutch up and those who wilt. Lead-hold data directly captures the asymmetry and should improve calibration in tight final-round scenarios.

## Acceptance Rule

Use judgment, but always record both scoreboards:

- anchor eval: stable, frozen, comparable over time
- research eval: flexible, evolving, exploratory

If a change improves research performance but hurts the anchor benchmark, think carefully before keeping it. If a change meaningfully improves the anchor benchmark, it is usually worth strong consideration even if the research eval has changed.

## Failure Handling

- If a source is flaky, replace it.
- If parsing is brittle, tighten the format or redesign the output.
- If a training run crashes, fix it quickly and continue.
- If a benchmark redesign helps research, keep it in the research eval only.
- If you need a cleaner anchor eval, create it once and then freeze it permanently.

## Never Stop

Do not stop to ask whether you should continue once the loop has started. Keep researching, gathering better public data, redesigning the system, and iterating until you are manually interrupted.