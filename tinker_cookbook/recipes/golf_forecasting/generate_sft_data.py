"""Generate SFT training data by running DeepSeek-V3.1 on training examples.

Creates a JSONL file of (prompt, completion) pairs for supervised fine-tuning.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import tinker
from tinker import types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.golf_forecasting.data import (
    load_examples,
    load_dataset_manifest,
)
from tinker_cookbook.recipes.golf_forecasting.env import build_messages

logger = logging.getLogger(__name__)


async def generate_sft_data(
    teacher_model: str = "deepseek-ai/DeepSeek-V3.1",
    dataset_manifest_path: str = "tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json",
    output_path: str = "/tmp/golf_sft_data.jsonl",
    max_candidates: int = 3,
    max_tokens: int = 512,
    max_parallel: int = 16,
) -> None:
    renderer_name = model_info.get_recommended_renderer_name(teacher_model)
    tokenizer = get_tokenizer(teacher_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    client = tinker.ServiceClient()
    sampling_client = client.create_sampling_client(base_model=teacher_model)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )

    manifest = load_dataset_manifest(dataset_manifest_path)
    examples = load_examples(manifest.train_path)
    logger.info("Generating SFT data for %d training examples", len(examples))

    semaphore = asyncio.Semaphore(max_parallel)

    async def gen_one(ex):
        async with semaphore:
            messages = build_messages(ex, include_other_bucket=True, max_candidates=max_candidates)
            prompt = renderer.build_generation_prompt(messages)
            response = await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=params
            )
            text = renderers.get_text_content(
                renderer.parse_response(response.sequences[0].tokens)[0]
            )
            return {"messages": messages, "completion": text, "example_id": ex.example_id}

    results = await asyncio.gather(*(gen_one(ex) for ex in examples))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    with output.open("w") as f:
        for r in results:
            if r["completion"].strip():
                f.write(json.dumps(r, sort_keys=True) + "\n")
                valid += 1

    logger.info("Wrote %d SFT examples to %s (%d total)", valid, output_path, len(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(generate_sft_data())
