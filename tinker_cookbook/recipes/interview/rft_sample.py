"""
v3 Phase B — Rejection-Sampling Fine-Tuning (RFT): sampling step.

Samples K rollouts per problem from an OPSD-bootstrapped sampler,
scores each with the v2.1 primary-score formula, and writes a filtered
dataset of top-K-per-problem positives to a JSONL file.

Output JSONL format (one line per kept rollout):
{
    "index": int (deepmath shuffled index),
    "question": str,
    "ground_truth": str,
    "all_tokens": list[int] (assistant tokens, concatenated),
    "history": list[Message] (full conversation with tool acks),
    "n_tool_calls": int,
    "n_turn_splits": int,
    "is_interleaved": bool,
    "is_correct": bool,
    "split_balance": float,
    "n_tokens": int,
    "efficiency": float,
    "score": float,  // is_correct * (0.5 + 0.5 * interleaved * balance) * efficiency
}

Usage:
    LD_PRELOAD=... .venv/bin/python -m \\
      tinker_cookbook.recipes.interview.rft_sample \\
      sampler_path="tinker://...sampler_weights/step_2" \\
      n_problems=200 group_size=4 score_threshold=0.5
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import chz
import tinker
from datasets import load_dataset

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.interview.opsd_train import roll_out_student
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class RFTSampleConfig:
    sampler_path: str  # e.g. tinker://...sampler_weights/step_2 from OPSD
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str | None = None

    # Data
    train_index_start: int = 500
    train_index_end: int = 1500   # 1000 problems by default
    n_problems: int = 200          # subsample for cycle time
    group_size: int = 4

    # Sampling (mirror eval/OPSD)
    max_tokens_per_turn: int = 24576
    temperature: float = 0.6
    max_turns: int = 8

    # Filter
    score_threshold: float = 0.5
    keep_top_per_problem: int = 1

    # IO
    no_tool_ref_tokens: int = 5500  # for efficiency_factor
    out_path: str = "/tmp/tinker-examples/interview/rft_positives.jsonl"
    stats_path: str = "/tmp/tinker-examples/interview/rft_stats.json"


def _score_rollout(
    *, rollout: dict, is_correct: bool, no_tool_ref_tokens: int
) -> dict:
    """Compute per-rollout v2.1 score and components."""
    tool_calls = rollout["tool_call_char_positions"]
    n_tokens = sum(e - s for s, e in rollout["turn_token_ranges"])
    n_turn_splits = rollout["n_turn_splits"]
    in_think = rollout["in_think_calls"]
    is_interleaved = (in_think >= 1) or (n_turn_splits >= 2)
    # split_balance
    if len(tool_calls) >= 2:
        # we don't have total_chars in the rollout return; approximate via
        # decoded length
        total_chars = len(rollout["decoded"])
        if total_chars > 0:
            boundaries = [0] + sorted(tool_calls) + [total_chars]
            segments = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
            sb = (min(segments) / max(segments)) if max(segments) > 0 else 1.0
        else:
            sb = 0.0
    else:
        sb = 0.0
    efficiency = min(1.0, no_tool_ref_tokens / max(n_tokens, 1))
    # v2.2: pure multiplicative. To even have a positive score the
    # rollout must be correct AND interleaved AND have non-zero
    # split_balance. Skipping the tool now scores 0.
    score = (
        (1.0 if is_correct else 0.0)
        * (1.0 if is_interleaved else 0.0)
        * sb
        * efficiency
    )
    return {
        "is_correct": is_correct,
        "n_tool_calls": rollout["n_tool_calls"],
        "n_turn_splits": n_turn_splits,
        "in_think_calls": in_think,
        "is_interleaved": is_interleaved,
        "split_balance": sb,
        "n_tokens": n_tokens,
        "efficiency": efficiency,
        "score": score,
    }


async def main(config: RFTSampleConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=42)
    pool = list(range(config.train_index_start, config.train_index_end))
    # subsample
    import random
    rng = random.Random(0xCAFE)
    indices = rng.sample(pool, min(config.n_problems, len(pool)))
    problems = [(i, ds[i]) for i in indices]
    logger.info(f"Sampling {config.group_size} rollouts on {len(problems)} problems "
                f"from {config.sampler_path}")

    # Clients
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service = tinker.ServiceClient()
    sc = await service.create_sampling_client_async(model_path=config.sampler_path)
    params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Build all (problem, sample_id) tasks
    tasks = []
    for idx, problem in problems:
        for s in range(config.group_size):
            tasks.append((idx, problem, s, roll_out_student(
                sampling_client=sc, renderer=renderer, tokenizer=tokenizer,
                sample_params=params, question=problem["question"],
                max_turns=config.max_turns,
            )))

    logger.info(f"Launching {len(tasks)} rollouts concurrently...")
    results = await asyncio.gather(*[t[3] for t in tasks], return_exceptions=True)

    # Score + group by problem
    by_problem: dict[int, list[dict]] = {}
    n_err = 0
    n_total = 0
    for (idx, problem, s, _coro), r in zip(tasks, results):
        n_total += 1
        if isinstance(r, Exception):
            n_err += 1
            continue
        gt = str(problem.get("final_answer", problem.get("ground_truth", "?")))
        # Extract answer from final visible
        # Find last assistant message's text content
        final_text = ""
        for msg in reversed(r["history"]):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    final_text = "".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
                else:
                    final_text = content
                break
        try:
            predicted = extract_boxed(final_text)
            is_correct = bool(predicted is not None and grade_answer(predicted, gt))
        except ValueError:
            is_correct = False
        scored = _score_rollout(
            rollout=r, is_correct=is_correct, no_tool_ref_tokens=config.no_tool_ref_tokens,
        )
        record = {
            "index": idx, "sample": s, "question": problem["question"],
            "ground_truth": gt, "all_tokens": r["all_tokens"],
            "history": r["history"], **scored,
        }
        by_problem.setdefault(idx, []).append(record)

    logger.info(f"Sampled {n_total - n_err}/{n_total} rollouts successfully ({n_err} errors)")

    # Filter top-K per problem above threshold
    kept = []
    for idx, recs in by_problem.items():
        recs.sort(key=lambda r: -r["score"])  # highest first
        for r in recs[:config.keep_top_per_problem]:
            if r["score"] > config.score_threshold:
                kept.append(r)

    # Write outputs — serialize history properly: ToolCall objects to
    # dicts so rft_train can rebuild them as native ToolCall.
    def _serialize_msg(msg):
        out = {}
        for k, v in msg.items():
            if k == "tool_calls" and v is not None:
                out[k] = [
                    {
                        "id": getattr(tc, "id", None) or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", "checkpoint"),
                            "arguments": getattr(tc.function, "arguments", "{}"),
                        },
                    } for i, tc in enumerate(v)
                ]
            elif k == "content" and isinstance(v, list):
                out[k] = [
                    dict(p) if hasattr(p, "items") else p
                    for p in v
                ]
            else:
                out[k] = v
        return out

    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in kept:
            r_dump = {k: v for k, v in r.items() if k != "history"}
            r_dump["history_json"] = json.dumps(
                [_serialize_msg(m) for m in r["history"]],
            )
            f.write(json.dumps(r_dump) + "\n")
    logger.info(f"Kept {len(kept)} positives (threshold={config.score_threshold}, "
                f"top-{config.keep_top_per_problem}-per-problem). Wrote {out_path}")

    # Stats
    scores_all = [r["score"] for recs in by_problem.values() for r in recs]
    correct_all = [r for recs in by_problem.values() for r in recs if r["is_correct"]]
    interleaved_all = [r for recs in by_problem.values() for r in recs if r["is_interleaved"]]
    stats = {
        "sampler_path": config.sampler_path,
        "n_total": n_total,
        "n_errored": n_err,
        "n_problems": len(by_problem),
        "n_kept": len(kept),
        "score_mean": (sum(scores_all) / len(scores_all)) if scores_all else 0.0,
        "score_max": max(scores_all) if scores_all else 0.0,
        "frac_correct": len(correct_all) / max(len(scores_all), 1),
        "frac_interleaved": len(interleaved_all) / max(len(scores_all), 1),
    }
    Path(config.stats_path).write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    cfg = chz.entrypoint(RFTSampleConfig)
    asyncio.run(main(cfg))
