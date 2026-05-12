"""Launch RL-from-scratch on the Countdown env (no teacher).

The Claim-A comparison for OPD: same student, same env, same hyperparams as
the OPD launcher — but no teacher. Pure GRPO-style RL with group-relative
advantages from the env's correctness/format reward.

Example:
    uv run python -m experiments.opd_rl.launch_rl_countdown \
        wandb_project=opd_rl wandb_name=iter05-rl-from-scratch \
        groups_per_batch=16 group_size=4 max_steps=30
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
from tinker_cookbook.rl.train import Config, main

from experiments.opd_rl.countdown_env import CountdownDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 8
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Env
    n_sources: int = 4
    group_size: int = 4
    groups_per_batch: int = 16
    n_batches: int = 1000
    seed: int = 0
    max_source: int = 25
    max_target: int = 100
    require_division: bool = False

    # Training
    learning_rate: float | None = None  # defaults to hyperparam_utils.get_lr
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 999
    save_every: int = 999
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
            log_path=f"/tmp/dylan/opd-rl/rl-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        )
    cli_utils.check_log_dir(cli.log_path, cli.behavior_if_log_dir_exists)
    Path(cli.log_path).mkdir(parents=True, exist_ok=True)

    dataset_builder = CountdownDatasetBuilder(
        batch_size=cli.groups_per_batch,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        n_batches=cli.n_batches,
        group_size=cli.group_size,
        n_sources=cli.n_sources,
        seed=cli.seed,
        max_source=cli.max_source,
        max_target=cli.max_target,
        require_division=cli.require_division,
    )

    config = Config(
        learning_rate=learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli.model_name,
        renderer_name=renderer_name,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        kl_penalty_coef=cli.kl_penalty_coef,
        num_substeps=cli.num_substeps,
        loss_fn=cli.loss_fn,
        loss_fn_config=cli.loss_fn_config,
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
    await main(config)


def entry() -> None:
    logging.basicConfig(level=logging.INFO)
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    entry()
