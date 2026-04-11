"""Forecast the 2026 Masters winner using the golf forecasting recipe.

Uses the best configuration from exp15 (top-5 candidates, 70B model).
Leaderboard data sourced from ESPN API on 2026-04-11 (after Round 2).

Usage:
    # Heuristic-only (no heavy deps needed):
    python -m tinker_cookbook.recipes.golf_forecasting.forecast_masters_2026

    # Full LLM inference (requires TINKER_API_KEY + torch + tinker):
    python -m tinker_cookbook.recipes.golf_forecasting.forecast_masters_2026 --llm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any

# ── Masters 2026 leaderboard after Round 2 (source: ESPN API, 2026-04-11) ────
# Rory McIlroy holds the largest 36-hole lead in Masters history at 6 shots.

PLAYERS_RAW: list[dict[str, Any]] = [
    {"name": "Rory McIlroy", "position": "1", "score_to_par": -12,
     "strokes_behind": 0, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-5, -7]},
    {"name": "Patrick Reed", "position": "T2", "score_to_par": -6,
     "strokes_behind": 6, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-3, -3]},
    {"name": "Sam Burns", "position": "T2", "score_to_par": -6,
     "strokes_behind": 6, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-5, -1]},
    {"name": "Tommy Fleetwood", "position": "T4", "score_to_par": -5,
     "strokes_behind": 7, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-1, -4]},
    {"name": "Justin Rose", "position": "T4", "score_to_par": -5,
     "strokes_behind": 7, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-2, -3]},
    {"name": "Shane Lowry", "position": "T4", "score_to_par": -5,
     "strokes_behind": 7, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-2, -3]},
    {"name": "Tyrrell Hatton", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [+2, -6]},
    {"name": "Cameron Young", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [+1, -5]},
    {"name": "Wyndham Clark", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -4]},
    {"name": "Kristoffer Reitan", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -4]},
    {"name": "Haotong Li", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-1, -3]},
    {"name": "Jason Day", "position": "T7", "score_to_par": -4,
     "strokes_behind": 8, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-3, -1]},
    {"name": "Chris Gotterup", "position": "T13", "score_to_par": -3,
     "strokes_behind": 9, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -3]},
    {"name": "Brooks Koepka", "position": "T13", "score_to_par": -3,
     "strokes_behind": 9, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -3]},
    {"name": "Ben Griffin", "position": "T13", "score_to_par": -3,
     "strokes_behind": 9, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -3]},
    {"name": "Jake Knapp", "position": "T16", "score_to_par": -2,
     "strokes_behind": 10, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [+1, -3]},
    {"name": "Max Homa", "position": "T16", "score_to_par": -2,
     "strokes_behind": 10, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -2]},
    {"name": "Hideki Matsuyama", "position": "T16", "score_to_par": -2,
     "strokes_behind": 10, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [0, -2]},
    {"name": "Xander Schauffele", "position": "T16", "score_to_par": -2,
     "strokes_behind": 10, "holes_completed": 36, "holes_remaining": 36,
     "round_scores": [-2, 0]},
    {"name": "Matt Fitzpatrick", "position": "T20", "score_to_par": -1,
     "strokes_behind": 11, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Collin Morikawa", "position": "T20", "score_to_par": -1,
     "strokes_behind": 11, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Michael Brennan", "position": "T20", "score_to_par": -1,
     "strokes_behind": 11, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Nick Taylor", "position": "T20", "score_to_par": -1,
     "strokes_behind": 11, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Patrick Cantlay", "position": "T24", "score_to_par": 0,
     "strokes_behind": 12, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Ludvig Åberg", "position": "T24", "score_to_par": 0,
     "strokes_behind": 12, "holes_completed": 36, "holes_remaining": 36},
    {"name": "Harris English", "position": "T24", "score_to_par": 0,
     "strokes_behind": 12, "holes_completed": 36, "holes_remaining": 36},
]

MAX_CANDIDATES = 5  # exp15 best config


def heuristic_forecast(
    players: list[dict[str, Any]],
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, float]:
    """Position-weighted exponential decay forecast.

    Matches the recipe's baseline_eval.py heuristic: exp(-0.5 * strokes_behind).
    """
    top_names = [p["name"] for p in players[:max_candidates]]
    labels = [*top_names, "other"]

    raw_weights: dict[str, float] = {}
    for player in players:
        raw_weights[player["name"]] = math.exp(-0.5 * player["strokes_behind"])

    total_weight = sum(raw_weights.values())

    probs: dict[str, float] = {}
    for label in labels:
        if label == "other":
            field_weight = sum(
                w for name, w in raw_weights.items() if name not in top_names
            )
            probs[label] = field_weight / total_weight
        elif label in raw_weights:
            probs[label] = raw_weights[label] / total_weight
        else:
            probs[label] = 0.0

    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def build_prompt_text(
    players: list[dict[str, Any]],
    max_candidates: int = MAX_CANDIDATES,
) -> str:
    """Build the exact system + user prompt that the recipe sends to the model."""
    if max_candidates > 0 and len(players) > max_candidates:
        top_names = [p["name"] for p in players[:max_candidates]]
    else:
        top_names = [p["name"] for p in players]
    candidates = [*top_names, "other"]

    # Leaderboard table
    header = (
        "| Player | Pos | To Par | Behind | Hole | Done | Remaining | Prior | Recent |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for p in players[:max_candidates]:
        rows.append(
            f"| {p['name']} | {p['position']} | {p['score_to_par']:+d} "
            f"| {p['strokes_behind']:.1f} | - | {p['holes_completed']} "
            f"| {p['holes_remaining']} | - | - |"
        )
    table = "\n".join([header, *rows])

    # Field analysis
    scores = [p["score_to_par"] for p in players]
    leader_score = scores[0]
    gap_to_2nd = scores[1] - leader_score
    within_3 = sum(1 for s in scores if s - leader_score <= 3)
    within_5 = sum(1 for s in scores if s - leader_score <= 5)

    calibration_hint = (
        "After round 2, the leader wins roughly 25-35% of the time. "
        "Comebacks are common -- spread probability across 5-10 contenders and the 'other' bucket."
    )

    n_shown = min(max_candidates, len(players))
    n_total = len(players)

    prompt = (
        f"Tournament: The Masters\n"
        f"Course: Augusta National\n"
        f"Round: 2\n"
        f"Event day: Saturday\n"
        f"Snapshot time: 2026-04-11T00:00:00Z\n\n"
        f"Leaderboard snapshot (top {n_shown} of {n_total} players):\n"
        f"{table}\n\n"
        f"Extra context:\n"
        f"- Weather: Typical Augusta spring conditions, light wind expected Saturday\n\n"
        f"Field analysis:\n"
        f"- Leader's margin: {gap_to_2nd:.0f} stroke(s) over 2nd place\n"
        f"- Players within 3 strokes of lead: {within_3}\n"
        f"- Players within 5 strokes: {within_5}\n\n"
        f"Calibration guidance: {calibration_hint}\n\n"
        f"Return a JSON object with a single key `winner_probs`. "
        f"Each key must be one of the candidate labels below and the probabilities must sum to 1. "
        f"Use `other` for all players not listed individually. "
        f"Assign non-zero probability to at least 5 candidates.\n"
        f"Candidate labels: {', '.join(candidates)}\n"
        f"Do not include explanations, markdown fences, or extra keys.\n"
        f'Example: {{"winner_probs": {{"Player A": 0.30, "Player B": 0.20, '
        f'"Player C": 0.15, "Player D": 0.10, "Player E": 0.08, "other": 0.17}}}}'
    )
    return prompt


def print_forecast(forecast: dict[str, float], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    ranked = sorted(forecast.items(), key=lambda x: -x[1])
    for i, (player, prob) in enumerate(ranked, 1):
        bar = "#" * int(prob * 50)
        print(f"  {i:>2}. {player:<25s} {prob:>6.1%}  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast 2026 Masters winner")
    parser.add_argument(
        "--llm", action="store_true",
        help="Run LLM inference via Tinker API (requires TINKER_API_KEY + torch)",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  2026 MASTERS TOURNAMENT - WINNER FORECAST")
    print("  Leaderboard: After Round 2 | 2026-04-11")
    print("  Course: Augusta National Golf Club")
    print("  Leader: Rory McIlroy (-12), 6-shot lead (Masters record)")
    print("  Config: exp15 best (top-5 candidates + other)")
    print("=" * 60)

    print("\n  LEADERBOARD (top 15):")
    print(f"  {'Pos':<5} {'Player':<25} {'Score':>6} {'Behind':>7}")
    print(f"  {'-' * 5} {'-' * 25} {'-' * 6} {'-' * 7}")
    for p in PLAYERS_RAW[:15]:
        score_str = f"{p['score_to_par']:+d}" if p["score_to_par"] != 0 else "E"
        behind_str = f"+{p['strokes_behind']:.0f}" if p["strokes_behind"] > 0 else "-"
        print(f"  {p['position']:<5} {p['name']:<25} {score_str:>6} {behind_str:>7}")

    # Heuristic forecast
    h_forecast = heuristic_forecast(PLAYERS_RAW, max_candidates=args.max_candidates)
    print_forecast(h_forecast, "HEURISTIC BASELINE (position-weighted exponential decay)")

    # Show the prompt
    prompt = build_prompt_text(PLAYERS_RAW, max_candidates=args.max_candidates)
    print("-" * 60)
    print("  PROMPT THAT WOULD BE SENT TO THE 70B MODEL:")
    print("-" * 60)
    print()
    print("  [system]: You are a calibrated golf forecasting assistant.")
    print("            Read the live leaderboard snapshot and produce a")
    print("            probability distribution over likely winners.")
    print("            Return JSON only.")
    print()
    print("  [user]:")
    for line in prompt.split("\n"):
        print(f"    {line}")
    print()

    if args.llm:
        import asyncio
        from tinker_cookbook.recipes.golf_forecasting.data import GolfForecastExample
        from tinker_cookbook.recipes.golf_forecasting.env import (
            build_messages,
            parse_forecast_response,
        )
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        import tinker
        from tinker import types

        example = GolfForecastExample.from_dict({
            "example_id": "masters-2026-r2",
            "tournament_id": "masters-2026",
            "tournament_name": "The Masters",
            "course_name": "Augusta National",
            "round_number": 2,
            "event_day": "Saturday",
            "snapshot_timestamp": "2026-04-11T00:00:00Z",
            "players": PLAYERS_RAW,
            "target_winner": "",
        })

        async def run_llm() -> dict[str, float]:
            renderer_name = model_info.get_recommended_renderer_name(args.model)
            tokenizer = get_tokenizer(args.model)
            renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(base_model=args.model)
            messages = build_messages(
                example, include_other_bucket=True, max_candidates=args.max_candidates,
            )
            model_input = renderer.build_generation_prompt(messages)
            result = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=256, temperature=0.0,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            text = renderers.get_text_content(
                renderer.parse_response(result.sequences[0].tokens)[0]
            )
            top_names = [p["name"] for p in PLAYERS_RAW[:args.max_candidates]]
            allowed = [*top_names, "other"]
            forecast, _ = parse_forecast_response(text, allowed_labels=allowed)
            return forecast

        print("Running LLM forecast with", args.model, "...")
        llm_fc = asyncio.run(run_llm())
        print_forecast(llm_fc, f"LLM FORECAST ({args.model.split('/')[-1]})")

    print("=" * 60)
    print("  To run with the actual 70B model via Tinker:")
    print("    export TINKER_API_KEY=your_key")
    print("    python -m tinker_cookbook.recipes.golf_forecasting.forecast_masters_2026 --llm")
    print("=" * 60)


if __name__ == "__main__":
    main()
