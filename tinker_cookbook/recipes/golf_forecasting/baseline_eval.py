"""Run baseline evaluation on anchor and research eval sets.

This script computes:
1. A heuristic baseline (position-weighted probability distribution)
2. An LLM baseline using Tinker API

Usage:
    python -m tinker_cookbook.recipes.golf_forecasting.baseline_eval
"""

from __future__ import annotations

import asyncio
import json
import math
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from tinker_cookbook.recipes.golf_forecasting.data import (
    GolfForecastExample,
    candidate_labels,
    load_examples,
    load_dataset_manifest,
)
from tinker_cookbook.recipes.golf_forecasting.env import (
    compute_log_loss,
    compute_multiclass_brier,
    score_forecast,
)

logger = logging.getLogger(__name__)


def heuristic_forecast(example: GolfForecastExample) -> dict[str, float]:
    """Create a probability distribution based on leaderboard position.

    Uses an exponential decay from position: the leader gets the most
    probability mass, decreasing exponentially with strokes behind.
    """
    labels = candidate_labels(example)
    probs: dict[str, float] = {}

    # Assign probability based on strokes behind leader using exponential decay
    raw_weights = {}
    for player in example.players:
        weight = math.exp(-0.5 * player.strokes_behind)
        raw_weights[player.name] = weight

    total_weight = sum(raw_weights.values())
    # Reserve 5% for "other" (field players not on leaderboard)
    other_mass = 0.05
    named_mass = 1.0 - other_mass

    for label in labels:
        if label == "other":
            probs[label] = other_mass
        elif label in raw_weights:
            probs[label] = named_mass * raw_weights[label] / total_weight
        else:
            probs[label] = 0.0

    # Ensure normalization
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def evaluate_heuristic(examples: list[GolfForecastExample]) -> dict[str, float]:
    """Run heuristic baseline evaluation over a set of examples."""
    log_losses = []
    briers = []
    top1_correct = []
    top3_recall = []

    for example in examples:
        forecast = heuristic_forecast(example)
        target = example.target_label
        scores = score_forecast(forecast, target_label=target)
        log_losses.append(scores["log_loss"])
        briers.append(scores["brier"])
        top1_correct.append(scores["top1_correct"])
        top3_recall.append(scores["top3_contains_target"])

    return {
        "eval/log_loss": float(np.mean(log_losses)),
        "eval/brier": float(np.mean(briers)),
        "eval/top1_accuracy": float(np.mean(top1_correct)),
        "eval/top3_recall": float(np.mean(top3_recall)),
        "eval/format_valid_rate": 1.0,  # heuristic always produces valid output
    }


async def evaluate_llm(
    examples: list[GolfForecastExample],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> dict[str, float]:
    """Run LLM baseline evaluation using Tinker API."""
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.recipes.golf_forecasting.env import (
        build_messages,
        parse_forecast_response,
    )

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    sampling_params = types.SamplingParams(
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    semaphore = asyncio.Semaphore(16)

    async def eval_one(example: GolfForecastExample) -> dict[str, float]:
        async with semaphore:
            messages = build_messages(example, include_other_bucket=True)
            prompt = renderer.build_generation_prompt(messages)
            try:
                response = await sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                text = renderers.get_text_content(
                    renderer.parse_response(response.sequences[0].tokens)[0]
                )
                allowed = candidate_labels(example)
                forecast, _ = parse_forecast_response(text, allowed_labels=allowed)
                return score_forecast(forecast, target_label=example.target_label)
            except Exception as e:
                logger.warning("LLM eval failed for %s: %s", example.example_id, e)
                return {
                    "brier": 2.0,
                    "log_loss": compute_log_loss({}, target_label=example.target_label),
                    "target_prob": 0.0,
                    "top1_correct": 0.0,
                    "top3_contains_target": 0.0,
                    "brier_reward": 0.0,
                }

    results = await asyncio.gather(*(eval_one(ex) for ex in examples))
    return {
        "eval/log_loss": float(np.mean([r["log_loss"] for r in results])),
        "eval/brier": float(np.mean([r["brier"] for r in results])),
        "eval/top1_accuracy": float(np.mean([r["top1_correct"] for r in results])),
        "eval/top3_recall": float(np.mean([r["top3_contains_target"] for r in results])),
        "eval/format_valid_rate": float(np.mean([1.0 if r["brier"] < 2.0 else 0.0 for r in results])),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest_path = "tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json"
    manifest = load_dataset_manifest(manifest_path)

    anchor_examples = load_examples(manifest.heldout_path)
    research_examples = load_examples(manifest.val_path)

    # Heuristic baseline
    logger.info("Running heuristic baseline on anchor eval (%d examples)...", len(anchor_examples))
    anchor_heuristic = evaluate_heuristic(anchor_examples)
    logger.info("Anchor heuristic: %s", json.dumps(anchor_heuristic, indent=2))

    logger.info("Running heuristic baseline on research eval (%d examples)...", len(research_examples))
    research_heuristic = evaluate_heuristic(research_examples)
    logger.info("Research heuristic: %s", json.dumps(research_heuristic, indent=2))

    # LLM baseline
    logger.info("Running LLM baseline on anchor eval...")
    anchor_llm = asyncio.run(evaluate_llm(anchor_examples))
    logger.info("Anchor LLM: %s", json.dumps(anchor_llm, indent=2))

    logger.info("Running LLM baseline on research eval...")
    research_llm = asyncio.run(evaluate_llm(research_examples))
    logger.info("Research LLM: %s", json.dumps(research_llm, indent=2))

    # Save results
    results_dir = Path("tinker_cookbook/recipes/golf_forecasting/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = results_dir / f"baseline_{timestamp}"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "heuristic": {
            "anchor": anchor_heuristic,
            "research": research_heuristic,
        },
        "llm_baseline": {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "anchor": anchor_llm,
            "research": research_llm,
        },
    }
    (baseline_dir / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("Saved baseline results to %s", baseline_dir)

    print("\n=== BASELINE RESULTS ===")
    print(f"\nHeuristic (position-weighted):")
    print(f"  Anchor:   log_loss={anchor_heuristic['eval/log_loss']:.4f}  brier={anchor_heuristic['eval/brier']:.4f}  top1={anchor_heuristic['eval/top1_accuracy']:.4f}")
    print(f"  Research: log_loss={research_heuristic['eval/log_loss']:.4f}  brier={research_heuristic['eval/brier']:.4f}  top1={research_heuristic['eval/top1_accuracy']:.4f}")
    print(f"\nLLM Baseline (Llama-3.1-8B-Instruct, zero-shot):")
    print(f"  Anchor:   log_loss={anchor_llm['eval/log_loss']:.4f}  brier={anchor_llm['eval/brier']:.4f}  top1={anchor_llm['eval/top1_accuracy']:.4f}")
    print(f"  Research: log_loss={research_llm['eval/log_loss']:.4f}  brier={research_llm['eval/brier']:.4f}  top1={research_llm['eval/top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()
