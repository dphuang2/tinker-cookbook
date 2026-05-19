"""Run a batch of DeepMath indices concurrently and dump raw rollouts.

Usage:
    INDICES=54,67,123,228,230,261,314,358,431,489 SAMPLES=4 \
      python -m tinker_cookbook.recipes.interview.dump_raw_rollouts_batch

Saves per-(index, sample) JSON+MD under raw_rollouts/, plus a summary
classifying which runs were truly interleaved (tool calls span ≥2 turns).
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")

import tinker
from datasets import load_dataset

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(): pass

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.interview.sft_train import (
    PROGRESS_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_SUFFIX,
)
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
INDICES = [int(x) for x in os.environ.get("INDICES", "54,67,123,228,230,261,314,358,431,489").split(",")]
SAMPLES_PER_INDEX = int(os.environ.get("SAMPLES", "4"))
MAX_TOKENS_PER_TURN = 24576
MAX_TURNS = 8
TEMPERATURE = 0.6
OUT_DIR = Path(__file__).parent / "raw_rollouts"
OUT_DIR.mkdir(exist_ok=True)


async def run_one(sc, renderer, params, problem, idx, sample_id):
    t_start = time.monotonic()
    log.info(f"START idx={idx:>3} s={sample_id}")
    tools = [PROGRESS_TOOL_SPEC]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tools, system_prompt=SYSTEM_PROMPT
    )
    history: list[Message] = list(prefix)
    history.append({"role": "user", "content": problem["question"] + USER_INSTRUCTION_SUFFIX})

    turns_raw = []
    for turn_idx in range(MAX_TURNS):
        prompt = renderer.build_generation_prompt(history)
        result = await sc.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        tokens = result.sequences[0].tokens
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        tokenizer_local = get_tokenizer(MODEL_NAME)
        decoded = tokenizer_local.decode(tokens)
        parsed, termination = renderer.parse_response(tokens)
        history.append(parsed)
        n_tool_calls = len(parsed.get("tool_calls") or [])
        log.info(f"  ...idx={idx:>3} s={sample_id} turn={turn_idx} "
                 f"tok={len(tokens)} calls={n_tool_calls} term={termination.value}")
        turns_raw.append({
            "turn": turn_idx,
            "decoded": decoded,
            "termination": termination.value,
            "n_tokens": len(tokens),
            "n_tool_calls": n_tool_calls,
        })
        tool_calls = parsed.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                history.append({
                    "role": "tool",
                    "content": "ok",
                    "tool_call_id": tc.id or f"call_{turn_idx}",
                })
            continue
        break

    total_calls = sum(t["n_tool_calls"] for t in turns_raw)
    turns_with_calls = sum(1 for t in turns_raw if t["n_tool_calls"] > 0)
    interleaved = turns_with_calls >= 2  # tool calls spread across ≥2 turns
    elapsed = time.monotonic() - t_start
    tag = "INTERLEAVED" if interleaved else "batched"
    log.info(f"DONE  idx={idx:>3} s={sample_id} | "
             f"turns={len(turns_raw)} calls={total_calls} "
             f"tool_turns={turns_with_calls} | {tag} | {elapsed:.1f}s")

    return {
        "index": idx,
        "sample": sample_id,
        "question": problem["question"],
        "ground_truth": problem.get("final_answer", problem.get("ground_truth", "?")),
        "n_turns": len(turns_raw),
        "n_tool_calls": total_calls,
        "turns_with_tool_calls": turns_with_calls,
        "interleaved": interleaved,
        "turns": turns_raw,
    }


async def main():
    load_dotenv()
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service = tinker.ServiceClient()
    sc = await service.create_sampling_client_async(base_model=MODEL_NAME)
    params = tinker.types.SamplingParams(
        max_tokens=MAX_TOKENS_PER_TURN,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    # build all (idx, sample) tasks
    tasks = []
    for idx in INDICES:
        problem = ds[idx]
        for s in range(SAMPLES_PER_INDEX):
            tasks.append(run_one(sc, renderer, params, problem, idx, s))

    log.info(f"Launching {len(tasks)} rollouts ({len(INDICES)} indices × {SAMPLES_PER_INDEX} samples) concurrently...")
    t_batch_start = time.monotonic()
    summary = []
    done_count = 0
    interleaved_so_far = 0

    # use as_completed so we get incremental progress
    for fut in asyncio.as_completed(tasks):
        try:
            r = await fut
        except Exception as e:
            log.error(f"task failed: {e}")
            done_count += 1
            continue
        done_count += 1
        path = OUT_DIR / f"idx{r['index']:03d}_s{r['sample']}.json"
        path.write_text(json.dumps(r, indent=2))
        summary.append({
            "index": r["index"], "sample": r["sample"],
            "n_turns": r["n_turns"], "n_tool_calls": r["n_tool_calls"],
            "turns_with_calls": r["turns_with_tool_calls"],
            "interleaved": r["interleaved"],
            "path": str(path.name),
        })
        if r["interleaved"]:
            interleaved_so_far += 1
        log.info(f"PROGRESS [{done_count}/{len(tasks)}] interleaved={interleaved_so_far} "
                 f"elapsed={time.monotonic()-t_batch_start:.0f}s")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown for the interleaved ones only
    interleaved = [s for s in summary if s["interleaved"]]
    log.info(f"FINAL: {len(interleaved)} / {len(summary)} interleaved")

    md = [f"# Interleaved rollouts — {len(interleaved)} / {len(summary)} ran with multi-turn tool calls\n"]
    for s in interleaved:
        r = json.loads((OUT_DIR / s["path"]).read_text())
        md.append(f"## idx {r['index']} · sample {r['sample']} — turns={r['n_turns']}, calls={r['n_tool_calls']}\n")
        md.append(f"**Q:** {r['question']}\n")
        md.append(f"**Ground truth:** `{r['ground_truth']}`\n")
        for t in r["turns"]:
            md.append(f"### Turn {t['turn']} ({t['n_tokens']} tok, {t['n_tool_calls']} tool calls, term={t['termination']})\n")
            md.append("```")
            md.append(t["decoded"])
            md.append("```\n")
        md.append("---\n")
    (OUT_DIR / "interleaved.md").write_text("\n".join(md))
    log.info(f"wrote {OUT_DIR / 'interleaved.md'}")


if __name__ == "__main__":
    asyncio.run(main())
