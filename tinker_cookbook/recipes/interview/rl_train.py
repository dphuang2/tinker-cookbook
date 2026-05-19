"""
v3 Phase C — Reinforcement learning with leave-one-out baseline (RLOO).

Builds on Phase B by sampling groups of rollouts on-policy, scoring each
with the v2.2 metric (accuracy × interleaving × split_balance × efficiency),
computing RLOO advantages (advantage_i = score_i − mean(scores in group
without i)), and running policy-gradient updates with importance-sampling
loss.

Key differences vs `rft_train.py`:
- Uses *current* policy rollouts (not stored positives JSONL)
- Reward comes from a continuous score per rollout (not just keep/drop)
- Negative gradient on low-scoring rollouts pushes the policy away from
  bad patterns (e.g. token-bloated rollouts), not just toward good ones
- Group-relative baseline (RLOO) drops the variance vs per-step REINFORCE

Usage:
    LD_PRELOAD=... .venv/bin/python -m \\
      tinker_cookbook.recipes.interview.rl_train \\
      warmstart_sampler="tinker://.../sampler_weights/step_68_final" \\
      n_problems_per_step=16 group_size=4 max_steps=10 \\
      learning_rate=1e-5
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any

import chz
import tinker
import torch
from datasets import load_dataset

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

from tinker import TensorData
from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.interview.opsd_train import roll_out_student
from tinker_cookbook.recipes.interview.rft_sample import _score_rollout
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class RLConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str | None = None
    lora_rank: int = 32

    # Warmstart: a sampler_weights path from a prior phase. RL will copy
    # those LoRA weights into a fresh training client. (We can't directly
    # load training STATE from a sampler — only weights — so optimizer
    # momentum starts fresh, which is fine for short RL runs.)
    warmstart_sampler: str | None = None

    # Data
    train_index_start: int = 500
    train_index_end: int = 2500
    n_problems_per_step: int = 16
    group_size: int = 4

    # Sampling
    max_tokens_per_turn: int = 24576
    temperature: float = 0.6
    max_turns: int = 8
    no_tool_ref_tokens: int = 5500
    score_token_budget: int = 28000  # skip rollouts longer than this

    # Optimization
    learning_rate: float = 1e-5
    max_steps: int = 10
    save_every: int = 2
    importance_sampling_clip: float = 0.2  # PPO-style clip

    # IO
    log_path: str = "/tmp/tinker-examples/interview/rl_run"


def _rloo_advantages(scores: list[float]) -> list[float]:
    """RLOO: advantage_i = score_i − mean(scores in group without i)."""
    n = len(scores)
    if n <= 1:
        return [0.0] * n
    total = sum(scores)
    return [s - (total - s) / (n - 1) for s in scores]


def _build_rl_datum(
    *, rollout: dict, advantage: float, renderer: Any,
) -> tinker.Datum | None:
    """Construct a Datum with per-token advantage for policy-gradient.

    model_input = sequence_tokens[:-1]
    target_tokens = sequence_tokens[1:]
    logprobs = student sample-time logprobs (zero-padded over prefix)
    advantages = advantage value applied uniformly across student tokens (0 elsewhere)
    mask = 1 on student-generated positions (drop before training)
    """
    from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
    preserve_renderer = Qwen3Renderer(
        renderer.tokenizer, strip_thinking_from_history=False,
    )
    model_input = preserve_renderer.build_generation_prompt(rollout["history"])
    sequence_tokens = list(model_input.to_ints())
    n_student = len(rollout["all_tokens"])
    if len(sequence_tokens) - 1 < n_student:
        # Sanity-check: prefix should be < total. If alignment is off,
        # drop this rollout rather than train on bad data.
        return None

    input_tokens = sequence_tokens[:-1]
    target_tokens = sequence_tokens[1:]
    # mask: 1 on the last n_student positions (student-generated)
    mask = [0.0] * (len(target_tokens) - n_student) + [1.0] * n_student
    # logprobs: pad prefix with 0, append student sample-time logprobs
    student_lp = rollout["all_logprobs"]
    if len(student_lp) != n_student:
        return None
    logprobs = [0.0] * (len(target_tokens) - n_student) + list(student_lp)
    # advantages: uniform across student tokens (0 elsewhere)
    advs = [m * advantage for m in mask]

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advs, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
        },
    )


async def main(config: RLConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log_dir = Path(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(
        json.dumps({k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
                   default=str, indent=2)
    )

    # Data
    ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=42)
    pool = list(range(config.train_index_start, config.train_index_end))
    logger.info(f"Train problem pool: {len(pool)} indices "
                f"[{config.train_index_start}..{config.train_index_end})")

    # Clients
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service = tinker.ServiceClient()

    training_client = await service.create_lora_training_client_async(
        config.model_name, rank=config.lora_rank,
    )
    # Warmstart: load LoRA weights from a prior sampler if specified
    if config.warmstart_sampler:
        logger.warning(
            f"Note: Tinker doesn't support direct LoRA weight loading from a "
            f"sampler_weights path into a fresh training client. For now we "
            f"START FROM FRESH LORA and use {config.warmstart_sampler} as the "
            f"INITIAL SAMPLING client (the policy that generates rollouts on "
            f"step 0). The training client is fresh; after one optim_step we'll "
            f"have moved away from the warmstart sampler."
        )
        sampling_client = await service.create_sampling_client_async(
            model_path=config.warmstart_sampler,
        )
    else:
        # Save initial weights and create a sampling client off them
        fut = await training_client.save_weights_for_sampler_async("step_0", ttl_seconds=86400)
        path = (await fut.result_async()).path
        sampling_client = await service.create_sampling_client_async(model_path=path)

    sample_params = tinker.SamplingParams(
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    rng = random.Random(0xC0DE)
    metrics_log = []
    checkpoints = []

    for step in range(config.max_steps):
        logger.info(f"=== RL step {step + 1}/{config.max_steps} ===")
        # Sample n_problems_per_step problems, group_size rollouts each
        batch_problems = rng.sample(pool, config.n_problems_per_step)
        rollout_tasks = []
        for idx in batch_problems:
            problem = ds[idx]
            for _ in range(config.group_size):
                rollout_tasks.append(
                    (idx, problem, roll_out_student(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        sample_params=sample_params,
                        question=problem["question"],
                        max_turns=config.max_turns,
                    )))
        logger.info(f"  rolling out {len(rollout_tasks)} trajectories...")
        rollouts = await asyncio.gather(
            *[t[2] for t in rollout_tasks], return_exceptions=True,
        )

        # Group by problem index
        groups: dict[int, list[dict]] = {}
        for (idx, problem, _coro), r in zip(rollout_tasks, rollouts):
            if isinstance(r, Exception):
                continue
            n_tok = sum(e - s for s, e in r["turn_token_ranges"])
            if n_tok > config.score_token_budget:
                continue
            # score the rollout
            gt = str(problem.get("final_answer", problem.get("ground_truth", "?")))
            # extract from final visible
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
                rollout=r, is_correct=is_correct,
                no_tool_ref_tokens=config.no_tool_ref_tokens,
            )
            r["_score"] = scored["score"]
            r["_problem"] = problem
            groups.setdefault(idx, []).append(r)

        # Build datums with RLOO advantages
        datums = []
        sum_score = 0.0
        n_scored = 0
        for idx, group in groups.items():
            if len(group) < 2:
                continue
            scores = [r["_score"] for r in group]
            advs = _rloo_advantages(scores)
            sum_score += sum(scores)
            n_scored += len(scores)
            for r, a in zip(group, advs):
                d = _build_rl_datum(rollout=r, advantage=a, renderer=renderer)
                if d is not None:
                    datums.append(d)

        if not datums:
            logger.warning(f"  step {step + 1}: no usable datums, skipping")
            continue

        mean_score = sum_score / max(n_scored, 1)
        logger.info(
            f"  step {step + 1}: {n_scored} scored, {len(datums)} datums, "
            f"mean_score={mean_score:.4f}"
        )

        # forward_backward + optim_step (strip mask first)
        datums_for_train = [
            tinker.Datum(
                model_input=d.model_input,
                loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
            ) for d in datums
        ]
        fwd_fut = await training_client.forward_backward_async(
            datums_for_train, loss_fn="importance_sampling",
            loss_fn_config={"clip_epsilon": config.importance_sampling_clip},
        )
        optim_fut = await training_client.optim_step_async(adam_params)
        await fwd_fut.result_async()
        await optim_fut.result_async()

        metrics_log.append({
            "step": step + 1,
            "n_problems_used": len(groups),
            "n_datums": len(datums),
            "mean_score": mean_score,
        })
        (log_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in metrics_log) + "\n"
        )

        # Save sampler and rebind for next step
        if (step + 1) % config.save_every == 0 or (step + 1) == config.max_steps:
            sampler_fut = await training_client.save_weights_for_sampler_async(
                f"step_{step + 1}", ttl_seconds=86400,
            )
            sampler_path = (await sampler_fut.result_async()).path
            sampling_client = await service.create_sampling_client_async(
                model_path=sampler_path,
            )
            checkpoints.append({"step": step + 1, "sampler_path": sampler_path})
            with open(log_dir / "checkpoints.jsonl", "a") as f:
                f.write(json.dumps(checkpoints[-1]) + "\n")
            logger.info(f"  saved sampler at step {step + 1}: {sampler_path}")

    logger.info(f"RL training complete. {len(checkpoints)} sampler checkpoints saved.")


if __name__ == "__main__":
    cfg = chz.entrypoint(RLConfig)
    asyncio.run(main(cfg))
