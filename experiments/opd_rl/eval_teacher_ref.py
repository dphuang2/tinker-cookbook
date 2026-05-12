"""Sample a (teacher) model on Countdown and report reward.

Used to fill the `teacher_ref` row of results.tsv so all student rows get a
meaningful `vs_teacher_gap`. No training, just rollouts.

Example:
    uv run python -m experiments.opd_rl.eval_teacher_ref \
        model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        n_problems=128 group_size=1 max_tokens=512
"""
from __future__ import annotations

import asyncio
import logging
import json
from pathlib import Path
from typing import Any

import chz
import numpy as np
import tinker

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from experiments.opd_rl.countdown_env import CountdownDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    n_problems: int = 64
    group_size: int = 1
    n_sources: int = 4
    max_source: int = 25
    max_target: int = 100
    require_division: bool = False
    max_tokens: int = 512
    temperature: float = 1.0
    out_json: str | None = None  # if set, write a json summary here


async def main(cli: CLIConfig) -> dict[str, Any]:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=cli.model_name)
    completer = TinkerTokenCompleter(sampling_client=sampling_client, max_tokens=cli.max_tokens, temperature=cli.temperature)

    builder = CountdownDatasetBuilder(
        batch_size=cli.n_problems,
        model_name_for_tokenizer=cli.model_name,
        renderer_name=renderer_name,
        n_batches=1,
        group_size=cli.group_size,
        n_sources=cli.n_sources,
        max_source=cli.max_source,
        max_target=cli.max_target,
        require_division=cli.require_division,
    )
    dataset, _ = await builder()
    env_group_builders = dataset.get_batch(0)

    rewards: list[float] = []
    corrects: list[float] = []
    formats: list[float] = []
    tasks = [do_group_rollout(gb, completer) for gb in env_group_builders]
    groups = await asyncio.gather(*tasks)
    for tg in groups:
        if tg is None:
            continue
        for r in tg.get_total_rewards():
            r = float(r)
            rewards.append(r)
            # ProblemEnv rewards: correct -> 1 + format_coef, format-only -> format_coef,
            # neither -> -format_coef. With format_coef=0.1: correct≈1.1, format≈0.1, fail≈-0.1.
            corrects.append(1.0 if r > 0.5 else 0.0)
            formats.append(1.0 if r > 0.0 else 0.0)

    summary = {
        "model": cli.model_name,
        "n_problems": cli.n_problems,
        "group_size": cli.group_size,
        "n_sources": cli.n_sources,
        "max_tokens": cli.max_tokens,
        "temperature": cli.temperature,
        "renderer": renderer_name,
        "n_rollouts": len(rewards),
        "reward_mean": float(np.mean(rewards)) if rewards else float("nan"),
        "reward_std": float(np.std(rewards)) if rewards else float("nan"),
        "correct_mean": float(np.mean(corrects)) if corrects else float("nan"),
        "format_mean": float(np.mean(formats)) if formats else float("nan"),
    }
    print("===TEACHER_REF_SUMMARY===")
    print(json.dumps(summary, indent=2))
    if cli.out_json:
        Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out_json).write_text(json.dumps(summary, indent=2))
    return summary


def entry() -> None:
    logging.basicConfig(level=logging.INFO)
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))


if __name__ == "__main__":
    entry()
