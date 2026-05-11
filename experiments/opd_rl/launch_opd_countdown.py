"""Launch on-policy distillation on the Countdown env.

Thin wrapper around tinker_cookbook.distillation.train_on_policy that swaps the
default DeepMath/Tulu3 dataset for our CountdownDatasetBuilder. Models default
to the pair chosen in program.md (Qwen3-1.7B instruct student, Qwen3-8B teacher).

Example:
    python -m experiments.opd_rl.launch_opd_countdown \
        wandb_project=opd_rl \
        wandb_name=opd-countdown-may11 \
        groups_per_batch=64
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig, TeacherConfig

from experiments.opd_rl.countdown_env import CountdownDatasetBuilder  # noqa: E402

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Models
    model_name: str = "Qwen/Qwen3-1.7B"
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None
    load_checkpoint_path: str | None = None

    # LoRA + renderer
    lora_rank: int = 32
    renderer_name: str | None = None

    # Env
    n_sources: int = 4
    group_size: int = 8
    groups_per_batch: int = 64
    n_batches: int = 1000

    # Training
    learning_rate: float | None = None  # Defaults to hyperparam_utils.get_lr(model_name).
    max_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 20
    save_every: int = 20
    compute_post_kl: bool = False
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None
    base_url: str | None = None


async def cli_main(cli: CLIConfig) -> None:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    learning_rate = cli.learning_rate if cli.learning_rate is not None else hyperparam_utils.get_lr(cli.model_name, is_lora=True)
    if cli.log_path is None:
        cli = chz.replace(
            cli,
            log_path=f"/tmp/dylan/opd-rl/{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )
    Path(cli.log_path).mkdir(parents=True, exist_ok=True)
    cli_utils.maybe_warn_about_log_dir(cli.log_path, cli.behavior_if_log_dir_exists)

    dataset_builder = CountdownDatasetBuilder(
        batch_size=cli.groups_per_batch,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        n_batches=cli.n_batches,
        group_size=cli.group_size,
        n_sources=cli.n_sources,
    )

    dataset_configs = [
        DistillationDatasetConfig(
            dataset_builder=dataset_builder,
            teacher_config=TeacherConfig(
                base_model=cli.teacher_model,
                load_checkpoint_path=cli.teacher_checkpoint,
            ),
            groups_per_batch=cli.groups_per_batch,
        )
    ]

    config = train_on_policy.Config(
        learning_rate=learning_rate,
        dataset_configs=dataset_configs,
        model_name=cli.model_name,
        renderer_name=renderer_name,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        lora_rank=cli.lora_rank,
        kl_penalty_coef=cli.kl_penalty_coef,
        kl_discount_factor=cli.kl_discount_factor,
        loss_fn=cli.loss_fn,
        loss_fn_config=cli.loss_fn_config,
        num_substeps=cli.num_substeps,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name,
        log_path=cli.log_path,
        base_url=cli.base_url,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        compute_post_kl=cli.compute_post_kl,
        load_checkpoint_path=cli.load_checkpoint_path,
        max_steps=cli.max_steps,
    )
    await train_on_policy.main(config)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    chz.entrypoint(lambda cli: asyncio.run(cli_main(cli)), CLIConfig)


if __name__ == "__main__":
    main()
