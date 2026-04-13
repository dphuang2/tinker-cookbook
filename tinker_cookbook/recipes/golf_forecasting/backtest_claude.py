"""Backtest Claude Opus 4.6 on the 2026 Masters data using the same prompts as the golf forecasting eval."""

from __future__ import annotations

import json
import math
import os
import sys

import anthropic

from tinker_cookbook.recipes.golf_forecasting.data import GolfForecastExample, load_examples
from tinker_cookbook.recipes.golf_forecasting.env import (
    FORECAST_SYSTEM_PROMPT,
    build_messages,
    parse_forecast_response,
    score_forecast,
    compute_log_loss,
)

BACKTEST_PATH = os.path.join(os.path.dirname(__file__), "masters_2026_backtest.jsonl")
MAX_CANDIDATES = 3


def run_backtest(model: str = "claude-opus-4-6") -> None:
    client = anthropic.Anthropic()
    examples = load_examples(BACKTEST_PATH)

    all_results = []
    for ex in examples:
        messages = build_messages(ex, include_other_bucket=True, max_candidates=MAX_CANDIDATES)
        system_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text

        # Build allowed labels
        if MAX_CANDIDATES > 0 and len(ex.players) > MAX_CANDIDATES:
            top_names = [p.name for p in ex.players[:MAX_CANDIDATES]]
        else:
            top_names = ex.candidate_names
        allowed = [*top_names, "other"]

        effective_target = ex.target_label
        if effective_target not in allowed:
            effective_target = "other"

        try:
            forecast, diagnostics = parse_forecast_response(text, allowed_labels=allowed)
            scores = score_forecast(forecast, target_label=effective_target)
            result = {
                "example_id": ex.example_id,
                "target_winner": ex.target_winner,
                "target_label": effective_target,
                "forecast": forecast,
                "log_loss": scores["log_loss"],
                "brier": scores["brier"],
                "target_prob": scores["target_prob"],
                "top1_correct": scores["top1_correct"],
                "format_valid": 1.0,
                "raw_response": text,
            }
        except Exception as e:
            print(f"  PARSE FAILED: {e}")
            invalid_forecast = {label: 0.0 for label in allowed}
            result = {
                "example_id": ex.example_id,
                "target_winner": ex.target_winner,
                "target_label": effective_target,
                "forecast": invalid_forecast,
                "log_loss": compute_log_loss(invalid_forecast, target_label=effective_target),
                "brier": 2.0,
                "target_prob": 0.0,
                "top1_correct": 0.0,
                "format_valid": 0.0,
                "raw_response": text,
            }

        sorted_fc = sorted(result["forecast"].items(), key=lambda x: -x[1])
        pred_str = ", ".join(f"{k}: {v:.1%}" for k, v in sorted_fc)
        print(f"{ex.example_id:20s} | target_prob={result['target_prob']:.3f} | ll={result['log_loss']:.2f} | top1={result['top1_correct']:.0f} | [{pred_str}]")
        all_results.append(result)

    # Aggregate metrics
    n = len(all_results)
    avg_ll = sum(r["log_loss"] for r in all_results) / n
    avg_brier = sum(r["brier"] for r in all_results) / n
    avg_top1 = sum(r["top1_correct"] for r in all_results) / n
    avg_fv = sum(r["format_valid"] for r in all_results) / n

    print(f"\n{'='*60}")
    print(f"Claude {model} — Masters 2026 Backtest")
    print(f"{'='*60}")
    print(f"  Log-loss:      {avg_ll:.4f}")
    print(f"  Brier:         {avg_brier:.4f}")
    print(f"  Top-1 acc:     {avg_top1:.1%}")
    print(f"  Format valid:  {avg_fv:.1%}")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results", f"masters2026_backtest_{model.replace('-', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
        for r in all_results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "raw_response"}) + "\n")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"eval/log_loss": avg_ll, "eval/brier": avg_brier, "eval/top1_accuracy": avg_top1, "eval/format_valid_rate": avg_fv}, f, indent=2)
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-6"
    run_backtest(model)
