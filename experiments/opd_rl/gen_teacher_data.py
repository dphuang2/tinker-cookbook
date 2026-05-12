"""Generate teacher (off-policy) rollouts on Countdown for SFT.

Samples the teacher model on N Countdown problems, keeps only trajectories
where the answer is correct, and writes them as a JSONL of {messages: [...]}
suitable for SFT. Used as the SFT counterpart to OPD for cold-start comparison.

Skips the full RL rollout machinery and instead drives a MessageCompleter
directly so we have prompt+response strings.

Example:
    uv run python -m experiments.opd_rl.gen_teacher_data \
        model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        n_problems=400 n_sources=6 max_source=50 max_target=1000 max_tokens=1024 \
        out_jsonl=experiments/opd_rl/data/teacher_data_v2.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import chz
import numpy as np
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

from experiments.opd_rl.countdown_env import CountdownEnv, _sample_problem

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    n_problems: int = 400
    n_sources: int = 6
    max_source: int = 50
    max_target: int = 1000
    require_division: bool = False
    n_samples_per_problem: int = 4
    max_tokens: int = 1024
    temperature: float = 1.0
    seed: int = 1234
    out_jsonl: str = "experiments/opd_rl/data/teacher_data_v2.jsonl"


async def _sample_one(completer, prompt_text):
    msgs = [{"role": "user", "content": prompt_text}]
    msg = await completer(msgs)
    return msg.get("content", "") if isinstance(msg, dict) else str(msg)


async def main(cli: CLIConfig) -> None:
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    tokenizer = get_tokenizer(cli.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    sc = tinker.ServiceClient()
    sampling_client = await sc.create_sampling_client_async(base_model=cli.model_name)
    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
    )

    # Build problems
    rng = np.random.RandomState(cli.seed)
    problems = []
    for _ in range(cli.n_problems):
        target, srcs = _sample_problem(
            rng,
            n_sources=cli.n_sources,
            max_source=cli.max_source,
            max_target=cli.max_target,
            require_division=cli.require_division,
        )
        problems.append((target, srcs))

    out = Path(cli.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build env-style prompts and graders by instantiating CountdownEnv
    n_total = 0
    n_correct = 0
    written = 0
    with out.open("w") as fout:
        # Process in concurrent waves; cap concurrency to be polite to the service.
        WAVE = 32
        for i in range(0, len(problems), WAVE):
            wave = problems[i:i + WAVE]
            # n_samples_per_problem samples per problem in parallel
            tasks = []
            prompts: list[str] = []
            envs: list[CountdownEnv] = []
            for target, srcs in wave:
                env = CountdownEnv(target=target, sources=srcs, renderer=renderer)
                prompt = env.get_question()
                for _ in range(cli.n_samples_per_problem):
                    tasks.append(_sample_one(completer, prompt))
                    prompts.append(prompt)
                    envs.append(env)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for prompt, env, resp in zip(prompts, envs, responses):
                n_total += 1
                if isinstance(resp, Exception):
                    continue
                if env.check_answer(resp):
                    n_correct += 1
                    rec = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp},
                        ],
                        "target": env.target,
                        "sources": list(env.sources),
                    }
                    fout.write(json.dumps(rec) + "\n")
                    written += 1
            logger.info(f"after wave {i + len(wave)}/{len(problems)}: {n_correct}/{n_total} correct, {written} written")

    print(f"DONE: {n_correct}/{n_total} correct ({n_correct / max(1, n_total):.1%}), {written} examples → {out}")


def entry() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cli = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli))


if __name__ == "__main__":
    entry()
