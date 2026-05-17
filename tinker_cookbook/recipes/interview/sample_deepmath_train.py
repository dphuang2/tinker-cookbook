"""
Sample Qwen3-30B-A3B thinking traces for a slice of DeepMath used as teacher
input for the progress-update SFT.

Held-out eval is indices 0-499 (already evaluated). This script samples
indices 500-2999 (training + dev) and writes one JSON file with all traces.

Submits every request concurrently with a single asyncio.gather — no
semaphore. Tinker is expected to handle the load.

Usage:
    python -m tinker_cookbook.recipes.interview.sample_deepmath_train
"""

import asyncio
import json
import logging
from pathlib import Path

import tinker
from datasets import load_dataset
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
START_INDEX = 500
END_INDEX = 5500  # exclusive (0017: extended from 3000 to ~5000 records)
MAX_TOKENS = 16384
TEMPERATURE = 0.6
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_train_traces.json")


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading DeepMath-103K dataset...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)
    problems = [ds[i] for i in range(START_INDEX, END_INDEX)]
    n = len(problems)
    logger.info(f"Sampling {n} problems (indices {START_INDEX}..{END_INDEX - 1})")

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    sample_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_sequences,
    )

    prompts = []
    for problem in problems:
        messages: list[renderers.Message] = [
            {
                "role": "user",
                "content": problem["question"]
                + " Write your answer in \\boxed{} format. Don't think for too long unnecessarily, especially when you have a reasonable degree of confidence.",
            },
        ]
        prompts.append(renderer.build_generation_prompt(messages))

    logger.info(f"Submitting {n} sample requests concurrently...")
    sample_results = await asyncio.gather(
        *[
            sampling_client.sample_async(
                prompt=p,
                num_samples=1,
                sampling_params=sample_params,
            )
            for p in prompts
        ]
    )

    results = []
    num_clean = 0
    for i, (problem, sample_result) in enumerate(zip(problems, sample_results)):
        response_tokens = sample_result.sequences[0].tokens
        parsed_message, parse_termination = renderer.parse_response(response_tokens)

        content = parsed_message["content"]
        thinking = ""
        visible = ""
        if isinstance(content, list):
            for part in content:
                if part["type"] == "thinking":
                    thinking = part["thinking"]
                elif part["type"] == "text":
                    visible += part["text"]
        else:
            visible = content

        if parse_termination.is_clean:
            num_clean += 1

        results.append(
            {
                "dataset_index": START_INDEX + i,
                "question": problem["question"],
                "ground_truth": problem["final_answer"],
                "thinking": thinking,
                "response": visible,
                "num_tokens": len(response_tokens),
                "parse_termination": parse_termination.value,
            }
        )
        if (i + 1) % 100 == 0 or (i + 1) == n:
            logger.info(f"Processed {i + 1}/{n} (clean: {num_clean})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} traces to {OUTPUT_PATH} (clean: {num_clean}/{n})")


if __name__ == "__main__":
    asyncio.run(main())
