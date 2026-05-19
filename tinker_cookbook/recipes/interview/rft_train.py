"""
v3 Phase B — Rejection-Sampling Fine-Tuning (RFT): training step.

Reads the positives JSONL written by rft_sample.py, builds SFT datums
via the renderer's build_supervised_example (cross-entropy on assistant
tokens), runs forward_backward + optim_step on minibatches, saves
sampler weights periodically.

Usage:
    LD_PRELOAD=... .venv/bin/python -m \\
      tinker_cookbook.recipes.interview.rft_train \\
      positives_path=/tmp/.../rft_positives.jsonl \\
      load_checkpoint_path="tinker://...sampler_weights/step_2" \\
      n_epochs=1 batch_size=4 learning_rate=5e-5
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import chz
import tinker

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class RFTTrainConfig:
    positives_path: str  # JSONL from rft_sample.py
    load_checkpoint_path: str | None = None  # state path to resume from
    model_name: str = "Qwen/Qwen3-30B-A3B"
    renderer_name: str | None = None
    lora_rank: int = 32

    # Optimization
    learning_rate: float = 5e-5  # lower than OPSD; SFT signal is direct
    batch_size: int = 4
    n_epochs: int = 1
    max_length: int = 32768

    # IO
    log_path: str = "/tmp/tinker-examples/interview/rft_run"
    save_every: int = 5


def _build_datum_from_record(rec: dict, renderer) -> tinker.Datum:
    """Reconstruct an SFT datum from an RFT positive record.

    `history_json` is a list of Messages with tool_calls serialized as
    JSON-safe dicts (id/type/function.name/function.arguments). Rebuild
    proper ToolCall objects so the renderer's `build_supervised_example`
    emits the same chat-template format the model sees at sample-time.
    """
    from tinker_cookbook.renderers import ToolCall

    raw_history = json.loads(rec["history_json"])
    messages = []
    for msg in raw_history:
        role = msg.get("role")
        if role is None:
            continue
        new_msg = {"role": role, "content": msg.get("content", "")}
        if role == "assistant" and msg.get("tool_calls"):
            tcs = []
            for tc_dict in msg["tool_calls"]:
                fn = tc_dict.get("function", {})
                tcs.append(ToolCall(
                    id=tc_dict.get("id"),
                    function=ToolCall.FunctionBody(
                        name=fn.get("name", "checkpoint"),
                        arguments=fn.get("arguments", "{}"),
                    ),
                ))
            new_msg["tool_calls"] = tcs
        if role == "tool" and msg.get("tool_call_id"):
            new_msg["tool_call_id"] = msg["tool_call_id"]
        messages.append(new_msg)

    model_input, weights = renderer.build_supervised_example(messages)
    return datum_from_model_input_weights(
        model_input, weights, max_length=32768,
    )


async def main(config: RFTTrainConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    log_dir = Path(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load positives
    positives = []
    with open(config.positives_path) as f:
        for line in f:
            if line.strip():
                positives.append(json.loads(line))
    logger.info(f"Loaded {len(positives)} positives from {config.positives_path}")

    # Clients
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service = tinker.ServiceClient()

    if config.load_checkpoint_path:
        # NOTE: load_checkpoint_path expects a *state* path, not a sampler path.
        # If only a sampler is available, start fresh LoRA (warm-init unavailable).
        logger.warning(f"Attempting to load from {config.load_checkpoint_path} — "
                       f"requires a *state* path (not a sampler-weights path)")
        try:
            training_client = await service.create_training_client_from_state_async(
                config.load_checkpoint_path,
            )
        except Exception as e:
            logger.warning(f"Could not load state ({e}); falling back to fresh LoRA")
            training_client = await service.create_lora_training_client_async(
                config.model_name, rank=config.lora_rank,
            )
    else:
        training_client = await service.create_lora_training_client_async(
            config.model_name, rank=config.lora_rank,
        )

    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # Build datums
    datums = []
    for rec in positives:
        try:
            d = _build_datum_from_record(rec, renderer)
            datums.append(d)
        except Exception as e:
            logger.warning(f"datum build failed for idx={rec.get('index')}: {e}")
    logger.info(f"Built {len(datums)} SFT datums")
    if not datums:
        logger.error("no datums to train on, exiting")
        return

    # Train n_epochs over batches of size batch_size
    checkpoints = []
    metrics_log = []
    step = 0
    for epoch in range(config.n_epochs):
        # shuffle within epoch
        import random
        rng = random.Random(0xBEEF + epoch)
        order = list(range(len(datums)))
        rng.shuffle(order)
        for i in range(0, len(order), config.batch_size):
            batch_idx = order[i:i + config.batch_size]
            batch = [datums[j] for j in batch_idx]
            step += 1
            # SFT uses cross_entropy; mask is preserved by datum_from_...
            fwd_bwd_fut = await training_client.forward_backward_async(
                batch, loss_fn="cross_entropy", loss_fn_config=None,
            )
            optim_fut = await training_client.optim_step_async(adam_params)
            fwd_result = await fwd_bwd_fut.result_async()
            await optim_fut.result_async()
            # log mean per-batch loss-output if present
            try:
                out_lps = [o["logprobs"].to_torch() for o in fwd_result.loss_fn_outputs]
                mean_logprob = sum(lp.float().mean().item() for lp in out_lps) / len(out_lps)
            except Exception:
                mean_logprob = float("nan")
            metrics_log.append({"step": step, "epoch": epoch + 1, "batch_size": len(batch),
                                "mean_logprob": mean_logprob})
            logger.info(f"step {step} (epoch {epoch + 1}) batch={len(batch)} mean_lp={mean_logprob:.4f}")
            # checkpoint
            if step % config.save_every == 0:
                fut = await training_client.save_weights_for_sampler_async(
                    f"step_{step}", ttl_seconds=86400,
                )
                sampler_path = (await fut.result_async()).path
                checkpoints.append({"step": step, "sampler_path": sampler_path})
                with open(log_dir / "checkpoints.jsonl", "a") as f:
                    f.write(json.dumps(checkpoints[-1]) + "\n")
                logger.info(f"Saved sampler at step {step}: {sampler_path}")

    # final save
    fut = await training_client.save_weights_for_sampler_async(
        f"step_{step}_final", ttl_seconds=86400,
    )
    sampler_path = (await fut.result_async()).path
    checkpoints.append({"step": step, "sampler_path": sampler_path, "final": True})
    with open(log_dir / "checkpoints.jsonl", "a") as f:
        f.write(json.dumps(checkpoints[-1]) + "\n")
    (log_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics_log) + "\n"
    )
    logger.info(f"RFT training complete. Final sampler: {sampler_path}")


if __name__ == "__main__":
    cfg = chz.entrypoint(RFTTrainConfig)
    asyncio.run(main(cfg))
