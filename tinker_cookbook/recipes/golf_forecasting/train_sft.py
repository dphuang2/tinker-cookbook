"""Supervised fine-tuning for golf forecasting via DeepSeek distillation.

Pipeline:
  1. Generate teacher labels from DeepSeek-V3.1 on the training set.
  2. Build a supervised dataset from the (prompt, completion) pairs.
  3. Fine-tune Llama-3.1-8B-Instruct with LoRA via tinker supervised training.
  4. Evaluate on anchor and research eval sets.

Usage:
    python -m tinker_cookbook.recipes.golf_forecasting.train_sft \
        [teacher_model=deepseek-ai/DeepSeek-V3.1] \
        [student_model=meta-llama/Llama-3.1-8B-Instruct] \
        [max_candidates=20] \
        [n_epochs=5] \
        [learning_rate=2e-5] \
        [exp_name=exp52_scorecard_sft]
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker import types

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.golf_forecasting.data import (
    load_dataset_manifest,
    load_examples,
)
from tinker_cookbook.recipes.golf_forecasting.env import build_messages, parse_forecast_response
from tinker_cookbook.recipes.golf_forecasting.eval import (
    GolfForecastEvalConfig,
    run_eval,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GolfSFTDataset(SupervisedDataset):
    """Supervised dataset built from (prompt, completion) pairs."""

    def __init__(
        self,
        records: list[dict],
        renderer: renderers.Renderer,
        batch_size: int = 16,
        max_length: int = 1024,
    ):
        self.records = records
        self.renderer = renderer
        self.batch_size = batch_size
        self.max_length = max_length

    def __len__(self) -> int:
        return math.ceil(len(self.records) / self.batch_size)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.records))
        datums = []
        for record in self.records[start:end]:
            messages = record["messages"]
            completion = record["completion"]
            # Build conversation: system+user prompt from messages, then assistant completion
            conversation = list(messages) + [{"role": "assistant", "content": completion}]
            datum = conversation_to_datum(
                conversation,
                self.renderer,
                max_length=self.max_length,
            )
            datums.append(datum)
        return datums


@chz.chz
class GolfSFTDatasetBuilder(SupervisedDatasetBuilder):
    sft_data_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int = 16
    max_length: int = 1024

    def __call__(self) -> tuple[GolfSFTDataset, None]:
        records = []
        with open(self.sft_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        dataset = GolfSFTDataset(
            records=records,
            renderer=renderer,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        logger.info("Loaded %d SFT records from %s", len(records), self.sft_data_path)
        return dataset, None


# ---------------------------------------------------------------------------
# Teacher generation
# ---------------------------------------------------------------------------


async def generate_teacher_labels(
    *,
    teacher_model: str,
    train_examples_path: str,
    output_path: str,
    max_candidates: int,
    max_tokens: int = 512,
    max_parallel: int = 16,
    include_pressure: bool = False,
    include_player_history: bool = False,
    include_tournament_history: bool = False,
    include_player_quality: bool = False,
    teacher_include_pressure: bool | None = None,
    teacher_include_player_history: bool | None = None,
    teacher_include_tournament_history: bool | None = None,
    teacher_include_player_quality: bool | None = None,
) -> None:
    """Run the teacher model on training examples and write (messages, completion) pairs.

    The student messages (saved as training context) use include_* flags.
    The teacher prompt (used to generate labels) uses teacher_include_* if provided,
    otherwise falls back to the student flags. This enables "rich teacher → plain student"
    distillation where the teacher sees extra features but the student learns from plain prompts.
    """
    output_file = Path(output_path)
    if output_file.exists():
        count = sum(1 for l in output_file.read_text().splitlines() if l.strip())
        logger.info(
            "SFT data already exists at %s (%d records). Skipping generation.", output_path, count
        )
        return

    renderer_name = model_info.get_recommended_renderer_name(teacher_model)
    tokenizer = get_tokenizer(teacher_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    client = tinker.ServiceClient()
    sc = client.create_sampling_client(base_model=teacher_model)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    examples = load_examples(train_examples_path)
    logger.info("Generating teacher labels for %d examples (max_candidates=%d)...", len(examples), max_candidates)

    # Determine teacher vs student feature flags
    t_pressure = teacher_include_pressure if teacher_include_pressure is not None else include_pressure
    t_history = teacher_include_player_history if teacher_include_player_history is not None else include_player_history
    t_tournament = teacher_include_tournament_history if teacher_include_tournament_history is not None else include_tournament_history
    t_quality = teacher_include_player_quality if teacher_include_player_quality is not None else include_player_quality
    teacher_differs = (t_pressure != include_pressure or t_history != include_player_history or
                       t_tournament != include_tournament_history or t_quality != include_player_quality)
    if teacher_differs:
        logger.info(
            "Rich teacher → plain student: teacher(pressure=%s,quality=%s) student(pressure=%s,quality=%s)",
            t_pressure, t_quality, include_pressure, include_player_quality,
        )

    semaphore = asyncio.Semaphore(max_parallel)

    async def gen_one(ex):
        async with semaphore:
            # Student messages (what model sees at inference)
            student_messages = build_messages(
                ex,
                include_other_bucket=True,
                max_candidates=max_candidates,
                include_pressure=include_pressure,
                include_player_history=include_player_history,
                include_tournament_history=include_tournament_history,
                include_player_quality=include_player_quality,
            )
            if teacher_differs:
                # Teacher gets richer context for label generation
                teacher_messages = build_messages(
                    ex,
                    include_other_bucket=True,
                    max_candidates=max_candidates,
                    include_pressure=t_pressure,
                    include_player_history=t_history,
                    include_tournament_history=t_tournament,
                    include_player_quality=t_quality,
                )
            else:
                teacher_messages = student_messages
            prompt = renderer.build_generation_prompt(teacher_messages)
            try:
                resp = await sc.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
                text = renderers.get_text_content(renderer.parse_response(resp.sequences[0].tokens)[0])
                return {"messages": student_messages, "completion": text, "example_id": ex.example_id}
            except Exception as exc:
                logger.warning("Failed to generate for %s: %s", ex.example_id, exc)
                return None

    results = await asyncio.gather(*(gen_one(ex) for ex in examples))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    with output_file.open("w") as f:
        for r in results:
            if r and r["completion"].strip():
                f.write(json.dumps(r, sort_keys=True) + "\n")
                valid += 1
    logger.info("Wrote %d valid SFT records to %s", valid, output_path)


async def generate_self_consistency_teacher_labels(
    *,
    teacher_model: str,
    train_examples_path: str,
    output_path: str,
    max_candidates: int,
    n_samples: int = 3,
    sample_temperature: float = 0.3,
    max_tokens: int = 512,
    max_parallel: int = 16,
    include_pressure: bool = False,
    include_player_history: bool = False,
    include_tournament_history: bool = False,
    include_player_quality: bool = False,
) -> None:
    """Self-consistency: generate n_samples at sample_temperature and average probabilities."""
    output_file = Path(output_path)
    if output_file.exists():
        count = sum(1 for l in output_file.read_text().splitlines() if l.strip())
        logger.info(
            "SFT data already exists at %s (%d records). Skipping.", output_path, count
        )
        return

    renderer_name = model_info.get_recommended_renderer_name(teacher_model)
    tokenizer = get_tokenizer(teacher_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    client = tinker.ServiceClient()
    sc = client.create_sampling_client(base_model=teacher_model)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=sample_temperature,
        stop=renderer.get_stop_sequences(),
    )

    examples = load_examples(train_examples_path)
    logger.info(
        "Generating self-consistency labels (%d samples, T=%.2f) for %d examples...",
        n_samples, sample_temperature, len(examples),
    )

    semaphore = asyncio.Semaphore(max_parallel)

    async def gen_one(ex):
        async with semaphore:
            messages = build_messages(
                ex,
                include_other_bucket=True,
                max_candidates=max_candidates,
                include_pressure=include_pressure,
                include_player_history=include_player_history,
                include_tournament_history=include_tournament_history,
                include_player_quality=include_player_quality,
            )
            prompt = renderer.build_generation_prompt(messages)

            if max_candidates > 0 and len(ex.players) > max_candidates:
                top_names = [p.name for p in ex.players[:max_candidates]]
            else:
                top_names = ex.candidate_names
            allowed = [*top_names, "other"]

            accumulated: dict[str, float] = {label: 0.0 for label in allowed}
            valid_count = 0

            try:
                resp = await sc.sample_async(
                    prompt=prompt, num_samples=n_samples, sampling_params=params
                )
                for seq in resp.sequences:
                    try:
                        text = renderers.get_text_content(renderer.parse_response(seq.tokens)[0])
                        forecast, _ = parse_forecast_response(
                            text, allowed_labels=allowed, prob_floor=0.0
                        )
                        for label, prob in forecast.items():
                            accumulated[label] += prob
                        valid_count += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("Self-consistency failed for %s: %s", ex.example_id, exc)
                return None

            if valid_count == 0:
                return None

            averaged = {label: prob / valid_count for label, prob in accumulated.items()}
            total = sum(averaged.values())
            if total <= 0:
                return None
            normalized = {label: round(prob / total, 4) for label, prob in averaged.items()}
            completion = json.dumps({"winner_probs": normalized})
            return {"messages": messages, "completion": completion, "example_id": ex.example_id}

    results = await asyncio.gather(*(gen_one(ex) for ex in examples))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    with output_file.open("w") as f:
        for r in results:
            if r:
                f.write(json.dumps(r, sort_keys=True) + "\n")
                valid += 1
    logger.info("Wrote %d self-consistency SFT records to %s", valid, output_path)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


@chz.chz
class ExpConfig:
    exp_name: str = "exp52_scorecard_sft"
    teacher_model: str = "deepseek-ai/DeepSeek-V3.1"
    student_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_manifest_path: str = "tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json"
    anchor_jsonl_path: str = "tinker_cookbook/recipes/golf_forecasting/anchor_eval_heldout.jsonl"
    sft_data_path: str | None = None  # auto-derived if None
    max_candidates: int = 20
    n_epochs: int = 5
    learning_rate: float = 2e-5
    lora_rank: int = 32
    batch_size: int = 16
    max_tokens_generate: int = 512
    max_tokens_train: int = 2048
    results_base: str = "tinker_cookbook/recipes/golf_forecasting/results"
    base_url: str | None = None
    include_pressure: bool = False
    include_player_history: bool = False
    include_tournament_history: bool = False
    include_player_quality: bool = False
    n_consistency_samples: int = 1  # 1 = greedy, >1 = self-consistency averaging
    sample_temperature: float = 0.3  # temperature for self-consistency samples
    # Teacher-specific feature overrides (None = same as student include_* flags)
    # Set these to True while keeping student flags False for "rich teacher → plain student" distillation
    teacher_include_pressure: bool | None = None
    teacher_include_player_quality: bool | None = None


async def run(config: ExpConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sft_path = config.sft_data_path or f"/tmp/golf_sft_{config.exp_name}_{config.max_candidates}cand.jsonl"

    # 1. Generate teacher labels
    manifest = load_dataset_manifest(config.dataset_manifest_path)
    if config.n_consistency_samples > 1:
        await generate_self_consistency_teacher_labels(
            teacher_model=config.teacher_model,
            train_examples_path=manifest.train_path,
            output_path=sft_path,
            max_candidates=config.max_candidates,
            n_samples=config.n_consistency_samples,
            sample_temperature=config.sample_temperature,
            max_tokens=config.max_tokens_generate,
            include_pressure=config.include_pressure,
            include_player_history=config.include_player_history,
            include_tournament_history=config.include_tournament_history,
            include_player_quality=config.include_player_quality,
        )
    else:
        await generate_teacher_labels(
            teacher_model=config.teacher_model,
            train_examples_path=manifest.train_path,
            output_path=sft_path,
            max_candidates=config.max_candidates,
            max_tokens=config.max_tokens_generate,
            include_pressure=config.include_pressure,
            include_player_history=config.include_player_history,
            include_tournament_history=config.include_tournament_history,
            include_player_quality=config.include_player_quality,
            teacher_include_pressure=config.teacher_include_pressure,
            teacher_include_player_quality=config.teacher_include_player_quality,
        )

    # 2. Build dataset and train
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=config.student_model,
        explicit_renderer_name=None,
        load_checkpoint_path=None,
        base_url=config.base_url,
    )
    dataset_builder = GolfSFTDatasetBuilder(
        sft_data_path=sft_path,
        model_name_for_tokenizer=config.student_model,
        renderer_name=renderer_name,
        batch_size=config.batch_size,
        max_length=config.max_tokens_train,
    )

    log_path = f"/tmp/tinker-examples/golf_sft/{config.exp_name}-{timestamp}"
    run_name = f"{config.exp_name}-{timestamp}"

    train_config = supervised_train.Config(
        model_name=config.student_model,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        num_epochs=config.n_epochs,
        lora_rank=config.lora_rank,
        log_path=log_path,
        renderer_name=renderer_name,
        eval_every=50,
        save_every=50,
        base_url=config.base_url,
    )
    checkpoint_url = await supervised_train.main(train_config)
    logger.info("Training complete. Checkpoint: %s", checkpoint_url)

    if not checkpoint_url:
        logger.error("No checkpoint returned — training may have failed.")
        return

    # 3. Evaluate on anchor and research
    for split_name, jsonl_path in [
        ("anchor", config.anchor_jsonl_path),
        ("research", manifest.val_path),
    ]:
        out_dir = f"{config.results_base}/{config.exp_name}_{split_name}"
        eval_config = GolfForecastEvalConfig(
            model_name=config.student_model,
            checkpoint_url=checkpoint_url,
            heldout_jsonl_path=jsonl_path,
            output_path=out_dir,
            temperature=0.0,
            max_tokens=384,
            max_parallel_tasks=16,
            include_other_bucket=True,
            max_candidates=config.max_candidates,
            base_url=config.base_url,
            include_pressure=config.include_pressure,
            include_player_history=config.include_player_history,
            include_tournament_history=config.include_tournament_history,
            include_player_quality=config.include_player_quality,
        )
        metrics = await run_eval(eval_config)
        logger.info(
            "[%s] ll=%.4f brier=%.4f top1=%.3f",
            split_name,
            metrics["eval/log_loss"],
            metrics["eval/brier"],
            metrics["eval/top1_accuracy"],
        )

    # 4. Update results.tsv
    anchor_metrics_path = Path(config.results_base) / f"{config.exp_name}_anchor"
    research_metrics_path = Path(config.results_base) / f"{config.exp_name}_research"

    def _latest_metrics(results_dir: Path) -> dict:
        subdirs = sorted(results_dir.glob("*/metrics.json"))
        if not subdirs:
            return {}
        return json.loads(subdirs[-1].read_text())

    a = _latest_metrics(anchor_metrics_path)
    r = _latest_metrics(research_metrics_path)

    tsv_path = Path("tinker_cookbook/recipes/golf_forecasting/results.tsv")
    line = (
        f"{config.exp_name}\t"
        f"{a.get('eval/log_loss', 'N/A'):.4f}\t"
        f"{a.get('eval/brier', 'N/A'):.4f}\t"
        f"{r.get('eval/log_loss', 'N/A') if r else 'N/A'}\t"
        f"pending\t"
        f"data+training\t"
        f"Hole-by-hole + round history + max_candidates={config.max_candidates} SFT"
    )
    with tsv_path.open("a") as f:
        f.write(line + "\n")
    logger.info("Appended to results.tsv: %s", line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = chz.entrypoint(ExpConfig)
    asyncio.run(run(config))
