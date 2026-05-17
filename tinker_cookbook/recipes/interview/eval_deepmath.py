"""
Baseline eval of Qwen3-30B-A3B (thinking mode) on DeepMath problems.

Samples all problems concurrently via asyncio.gather, extracts \\boxed{...},
grades against ground truth using math_rl grading, and writes per-problem
results plus an aggregate accuracy.

Usage:
    python -m tinker_cookbook.recipes.interview.eval_deepmath
"""

import asyncio
import json
import logging
from pathlib import Path

import tinker
from datasets import load_dataset
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
NUM_PROBLEMS = 500
MAX_TOKENS = 16384
TEMPERATURE = 0.6
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_eval.json")


async def sample_one(sampling_client, prompt, sample_params):
    return await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sample_params,
    )


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading DeepMath-103K dataset...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)
    problems = [ds[i] for i in range(NUM_PROBLEMS)]

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    logger.info(f"Using renderer: {renderer_name}")
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    logger.info(f"Creating sampling client for {MODEL_NAME}...")
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

    logger.info(f"Submitting {NUM_PROBLEMS} sample requests in parallel...")
    sample_results = await asyncio.gather(
        *[sample_one(sampling_client, p, sample_params) for p in prompts]
    )

    results = []
    num_correct = 0
    num_clean = 0
    num_extract_failed = 0
    for i, (problem, sample_result) in enumerate(zip(problems, sample_results)):
        response_tokens = sample_result.sequences[0].tokens
        parsed_message, parse_termination = renderer.parse_response(response_tokens)

        content = parsed_message["content"]
        visible = ""
        if isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    visible += part["text"]
        else:
            visible = content

        try:
            predicted = extract_boxed(visible)
            extract_ok = True
        except ValueError:
            predicted = None
            extract_ok = False
            num_extract_failed += 1

        ground_truth = problem["final_answer"]
        is_correct = bool(predicted is not None and grade_answer(predicted, ground_truth))
        if is_correct:
            num_correct += 1
        if parse_termination.is_clean:
            num_clean += 1

        results.append(
            {
                "index": i,
                "question": problem["question"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "extract_ok": extract_ok,
                "parse_termination": parse_termination.value,
                "num_tokens": len(response_tokens),
            }
        )
        logger.info(
            f"[{i + 1}/{NUM_PROBLEMS}] correct={is_correct} "
            f"pred={predicted!r} gt={ground_truth!r} "
            f"tokens={len(response_tokens)} term={parse_termination.value}"
        )

    accuracy = num_correct / NUM_PROBLEMS
    summary = {
        "model": MODEL_NAME,
        "num_problems": NUM_PROBLEMS,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "num_clean_termination": num_clean,
        "num_extract_failed": num_extract_failed,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    logger.info(
        f"Accuracy: {num_correct}/{NUM_PROBLEMS} = {accuracy:.3f} "
        f"(clean termination: {num_clean}/{NUM_PROBLEMS}, "
        f"extract failed: {num_extract_failed})"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"Saved eval to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
