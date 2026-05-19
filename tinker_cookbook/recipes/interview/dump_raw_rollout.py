"""Re-run a single DeepMath problem and dump the raw decoded rollout."""

import asyncio
import json
import os
from pathlib import Path

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
TARGET_INDEX = int(os.environ.get("TARGET_INDEX", "1"))
NO_TOOL = bool(os.environ.get("NO_TOOL"))
MAX_TOKENS_PER_TURN = 24576
MAX_TURNS = 8
TEMPERATURE = 0.6
SUFFIX = "_notool" if NO_TOOL else ""
OUT_PATH = Path(__file__).parent / f"raw_rollout_idx{TARGET_INDEX}{SUFFIX}.json"
OUT_MD = Path(__file__).parent / f"raw_rollout_idx{TARGET_INDEX}{SUFFIX}.md"


async def main():
    load_dotenv()
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)
    problem = ds[TARGET_INDEX]

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

    # Mirror eval_deepmath_agent.py exactly: NO_TOOL drops only the tool spec.
    tools = [] if NO_TOOL else [PROGRESS_TOOL_SPEC]
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
        decoded = tokenizer.decode(tokens)
        parsed, termination = renderer.parse_response(tokens)
        history.append(parsed)
        turns_raw.append({
            "turn": turn_idx,
            "decoded": decoded,
            "termination": termination.value,
            "n_tokens": len(tokens),
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

    out = {
        "index": TARGET_INDEX,
        "question": problem["question"],
        "ground_truth": problem.get("final_answer", problem.get("ground_truth", "?")),
        "turns": turns_raw,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_PATH}")

    # also dump a markdown-friendly transcript
    lines = [f"# Raw rollout — DeepMath idx {TARGET_INDEX}", "", f"**Q:** {problem['question']}", "",
             f"**Ground truth:** `{out['ground_truth']}`", "", "---", ""]
    for t in turns_raw:
        lines.append(f"## Turn {t['turn']} ({t['n_tokens']} tokens · term={t['termination']})")
        lines.append("")
        lines.append("```")
        lines.append(t["decoded"])
        lines.append("```")
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
