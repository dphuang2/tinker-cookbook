"""SFT the student on teacher rollouts as the off-policy distillation
counterpart to OPD. Reads a JSONL produced by `gen_teacher_data.py`.

Example:
    uv run python -m experiments.opd_rl.launch_sft_countdown \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        file_path=experiments/opd_rl/data/teacher_data_v2.jsonl \
        max_steps=30 batch_size=16 lora_rank=8 learning_rate=1e-4 \
        wandb_project=opd_rl wandb_name=iter18-sft-v2 \
        log_path=/tmp/dylan/opd-rl/iter18-sft-v2
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    file_path: str = "experiments/opd_rl/data/teacher_data_v2.jsonl"
    lora_rank: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 4
    max_length: int = 2048
    max_steps: int | None = None
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 0
    save_every: int = 0
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli: CLIConfig) -> None:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    if cli.log_path is None:
        cli = chz.replace(cli, log_path=f"/tmp/dylan/opd-rl/sft-{datetime.now().strftime('%Y-%m-%d-%H-%M')}")
    cli_utils.check_log_dir(cli.log_path, cli.behavior_if_log_dir_exists)
    Path(cli.log_path).mkdir(parents=True, exist_ok=True)

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        max_length=cli.max_length,
        batch_size=cli.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common,
        file_path=cli.file_path,
    )

    config = train.Config(
        log_path=cli.log_path,
        model_name=cli.model_name,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        num_epochs=cli.num_epochs,
        lora_rank=cli.lora_rank,
        wandb_project=cli.wandb_project,
        wandb_name=cli.wandb_name,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        max_steps=cli.max_steps,
    )
    await train.main(config)


def entry() -> None:
    logging.basicConfig(level=logging.INFO)
    cli = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli))


if __name__ == "__main__":
    entry()
