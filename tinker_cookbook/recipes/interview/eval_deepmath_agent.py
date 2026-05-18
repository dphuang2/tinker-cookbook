"""
Agent-loop eval on the held-out DeepMath slice for the progress-update SFT.

Loads a trained sampler checkpoint (default: the final sampler weights from
the most recent SFT run in /tmp/tinker-examples/interview/sft_run) and runs
an agent loop over the held-out problems (indices 0-499). Each step:

    1. Build prompt from current message history + tool spec.
    2. Sample assistant turn.
    3. If parsed message has tool_calls -> append assistant + tool ack, loop.
       Else -> extract \\boxed{} from visible text, grade, done.

Caps each problem at MAX_TURNS to prevent runaway tool-call loops.

Usage:
    python -m tinker_cookbook.recipes.interview.eval_deepmath_agent
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import tinker
from datasets import load_dataset
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.interview.sft_train import (
    PROGRESS_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_SUFFIX,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SFT_LOG_DIR = Path("/tmp/tinker-examples/interview/sft_run")
NUM_PROBLEMS = 500
HELDOUT_START = 0
MAX_TOKENS_PER_TURN = 24576  # 0062 sweet spot
MAX_TURNS = 8
TEMPERATURE = 0.6
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_agent_eval.json")


def find_final_sampler_path(log_dir: Path) -> str:
    ckpts = log_dir / "checkpoints.jsonl"
    last = None
    with open(ckpts) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("sampler_path"):
                last = rec["sampler_path"]
    if not last:
        raise RuntimeError(f"No sampler_path entries in {ckpts}")
    return last


async def run_agent(
    sampling_client,
    renderer,
    tokenizer,
    sample_params,
    problem,
) -> dict:
    tools = [] if os.environ.get("NO_TOOL") else [PROGRESS_TOOL_SPEC]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tools, system_prompt=SYSTEM_PROMPT
    )
    history: list[Message] = list(prefix)
    history.append(
        {"role": "user", "content": problem["question"] + USER_INSTRUCTION_SUFFIX}
    )

    progress_updates: list[str] = []
    total_tokens = 0
    final_visible = ""
    final_termination = "no_final"
    for turn_idx in range(MAX_TURNS):
        prompt = renderer.build_generation_prompt(history)
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sample_params,
        )
        tokens = result.sequences[0].tokens
        total_tokens += len(tokens)
        parsed, termination = renderer.parse_response(tokens)
        final_termination = termination.value
        history.append(parsed)

        tool_calls = parsed.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    progress_updates.append(args.get("summary", args.get("message", "")))
                except (json.JSONDecodeError, AttributeError):
                    progress_updates.append("<unparseable>")
                history.append(
                    {
                        "role": "tool",
                        "content": f"ok (checkpoint {turn_idx + 1} of 8)",
                        "tool_call_id": tc.id or f"call_{turn_idx}",
                    }
                )
            continue

        content = parsed.get("content", "")
        if isinstance(content, list):
            final_visible = "".join(
                p["text"] for p in content if p.get("type") == "text"
            )
        else:
            final_visible = content
        break

    try:
        predicted = extract_boxed(final_visible)
        extract_ok = True
    except ValueError:
        predicted = None
        extract_ok = False

    return {
        "predicted": predicted,
        "extract_ok": extract_ok,
        "final_visible": final_visible,
        "progress_updates": progress_updates,
        "num_turns": min(turn_idx + 1, MAX_TURNS),
        "num_tool_calls": len(progress_updates),
        "total_tokens": total_tokens,
        "last_termination": final_termination,
    }


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    env_sampler = os.environ.get("SAMPLER_PATH")
    if env_sampler == "base":
        sampler_path = None
        logger.info("Using BASE model (no SFT checkpoint)")
    else:
        sampler_path = env_sampler or find_final_sampler_path(SFT_LOG_DIR)
        logger.info(f"Using SFT checkpoint: {sampler_path}")

    logger.info("Loading DeepMath-103K dataset...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)
    problems = [ds[i] for i in range(HELDOUT_START, HELDOUT_START + NUM_PROBLEMS)]

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    service_client = tinker.ServiceClient()
    if sampler_path is None:
        sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=MODEL_NAME,
            model_path=sampler_path,
        )
    sample_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS_PER_TURN,
        temperature=TEMPERATURE,
        stop=stop_sequences,
    )

    logger.info(f"Running agent loop on {NUM_PROBLEMS} problems concurrently...")
    results_raw = await asyncio.gather(
        *[
            run_agent(sampling_client, renderer, tokenizer, sample_params, p)
            for p in problems
        ]
    )

    results = []
    num_correct = 0
    cadence_hist: dict[int, int] = {}
    for i, (problem, r) in enumerate(zip(problems, results_raw)):
        gt = problem["final_answer"]
        is_correct = bool(
            r["predicted"] is not None and grade_answer(r["predicted"], gt)
        )
        if is_correct:
            num_correct += 1
        cadence_hist[r["num_tool_calls"]] = (
            cadence_hist.get(r["num_tool_calls"], 0) + 1
        )
        results.append(
            {
                "index": HELDOUT_START + i,
                "question": problem["question"],
                "ground_truth": gt,
                "is_correct": is_correct,
                **r,
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == NUM_PROBLEMS:
            logger.info(
                f"[{i + 1}/{NUM_PROBLEMS}] correct so far: {num_correct} "
                f"(acc={num_correct / (i + 1):.3f})"
            )

    accuracy = num_correct / NUM_PROBLEMS
    summary = {
        "model": MODEL_NAME,
        "sampler_path": sampler_path,
        "num_problems": NUM_PROBLEMS,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "tool_call_cadence": sorted(cadence_hist.items()),
        "temperature": TEMPERATURE,
        "max_tokens_per_turn": MAX_TOKENS_PER_TURN,
        "max_turns": MAX_TURNS,
    }
    logger.info(f"Accuracy: {num_correct}/{NUM_PROBLEMS} = {accuracy:.3f}")
    logger.info(f"Tool-call cadence: {sorted(cadence_hist.items())}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    logger.info(f"Saved eval to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
