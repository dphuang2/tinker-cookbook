"""
Teacher-rewrite Qwen3 thinking traces into multi-turn agent transcripts that
interleave `progress_update` tool calls.

Reads `/tmp/tinker-examples/interview/deepmath_train_traces.json` (output of
sample_deepmath_train.py), uses Kimi-K2.6 on Tinker as the teacher, and emits
SFT-ready records to `/tmp/tinker-examples/interview/sft_dataset.json`.

Each output record is a list of "turns" describing the desired multi-turn
assistant behavior:

    {
      "dataset_index": int,
      "question": str,
      "ground_truth": str,
      "turns": [
        {"role": "assistant", "thinking": "...", "progress_update": "..."},
        ...
        {"role": "assistant", "thinking": "...", "final": "...<boxed answer>"},
      ]
    }

Confident/short traces produce 1 turn (no tool call). Long traces produce up
to ~3 turns with one `progress_update` between each, target cadence ~1 update
per 1000 thinking tokens.

Usage:
    python -m tinker_cookbook.recipes.interview.teacher_rewrite
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import tinker
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

TEACHER_MODEL = "moonshotai/Kimi-K2.6"
RENDERER_NAME = "kimi_k26_disable_thinking"
INPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_train_traces.json")
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/sft_dataset.json")
CONTEXT_WINDOW = 32768
MAX_TOKENS = 2048  # only emit split metadata; output is short regardless of trace length
TEMPERATURE = 0.3
PROMPT_TOKEN_LIMIT = CONTEXT_WINDOW - MAX_TOKENS

INSTRUCTION_TEMPLATE = """\
You will plan where to insert brief progress updates inside an existing
math-problem thinking trace. You will NOT rewrite the thinking - you only
emit metadata describing where to split it and what progress message to
emit at each split point.

The thinking trace below is annotated with character offsets every 200
characters in the form [@<offset>] inserted between sentences (these
markers are NOT part of the thinking - they're just landmarks you can
reference). Pick split offsets that fall at natural reasoning boundaries.

Cadence target: one `progress_update` per ~4000 thinking characters
(~1000 tokens). Short / confident traces (under ~6000 chars) should
get 0 split points (no tool calls at all). Long traces get up to 3
split points. Aim for fewer rather than more.

Each progress_update message must be one short first-person sentence
describing the reasoning state at that point, e.g. "Tried u-substitution
but the cross term didn't cancel - switching to partial fractions."

Output schema (single JSON object, no prose before or after):
{{
  "splits": [
    {{"split_after_offset": <int>, "progress_update": "<short status>"}},
    ...
  ]
}}

- `splits` may be empty (no tool calls needed).
- `split_after_offset` is the character offset in the ORIGINAL thinking
  (ignoring the [@N] markers) immediately AFTER which the split happens.
  Choose values that match the [@N] markers you saw.
- Offsets must be strictly increasing and within (0, total_length).

Problem:
{question}

Thinking length: {total_chars} characters.

Annotated thinking:
<<<THINKING
{annotated_thinking}
THINKING

Final visible answer (kept verbatim, not your concern):
<<<ANSWER
{answer}
ANSWER

Emit only the JSON object."""


def annotate_with_offsets(text: str, every: int = 200) -> str:
    """Insert [@N] markers at offset boundaries to help the teacher pick splits."""
    parts = []
    for i in range(0, len(text), every):
        if i > 0:
            parts.append(f"[@{i}]")
        parts.append(text[i : i + every])
    return "".join(parts)


def build_prompt(renderer, question: str, thinking: str, answer: str):
    instruction = INSTRUCTION_TEMPLATE.format(
        question=question,
        annotated_thinking=annotate_with_offsets(thinking),
        total_chars=len(thinking),
        answer=answer,
    )
    messages: list[renderers.Message] = [{"role": "user", "content": instruction}]
    return renderer.build_generation_prompt(messages)


def extract_json(text: str) -> dict | None:
    """Pull the first balanced top-level JSON object out of the response."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def splice_turns(
    parsed: dict, thinking: str, response: str
) -> list[dict] | None:
    """Use teacher-emitted split offsets to build the multi-turn structure."""
    splits = parsed.get("splits")
    if not isinstance(splits, list):
        return None
    total = len(thinking)
    last_off = 0
    cleaned = []
    for sp in splits:
        if not isinstance(sp, dict):
            return None
        off = sp.get("split_after_offset")
        msg = sp.get("progress_update")
        if not isinstance(off, int) or not isinstance(msg, str) or not msg.strip():
            return None
        if off <= last_off or off >= total:
            return None
        cleaned.append((off, msg.strip()))
        last_off = off

    turns = []
    prev = 0
    for off, msg in cleaned:
        turns.append(
            {
                "role": "assistant",
                "thinking": thinking[prev:off],
                "progress_update": msg,
            }
        )
        prev = off
    turns.append(
        {
            "role": "assistant",
            "thinking": thinking[prev:],
            "final": response,
        }
    )
    return turns


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading traces from {INPUT_PATH}")
    with open(INPUT_PATH) as f:
        traces = json.load(f)
    # Only rewrite clean terminations (others may be malformed/truncated).
    clean = [t for t in traces if t.get("parse_termination") == "stop_sequence"]
    logger.info(f"Filtered {len(clean)}/{len(traces)} clean traces for rewriting")
    limit = int(os.environ.get("NUM_TRACES", "0"))
    if limit > 0:
        clean = clean[:limit]
        logger.info(f"NUM_TRACES={limit}, smoke-testing on first {len(clean)} traces")

    tokenizer = get_tokenizer(TEACHER_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=TEACHER_MODEL)
    sample_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_sequences,
    )

    prompts = []
    kept_traces = []
    num_oversize = 0
    for t in clean:
        p = build_prompt(renderer, t["question"], t["thinking"], t["response"])
        if p.length > PROMPT_TOKEN_LIMIT:
            num_oversize += 1
            continue
        prompts.append(p)
        kept_traces.append(t)
    logger.info(
        f"Built {len(prompts)} prompts (skipped {num_oversize} oversize "
        f">{PROMPT_TOKEN_LIMIT} tokens)"
    )
    logger.info(f"Submitting {len(prompts)} teacher rewrites concurrently...")
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
    clean = kept_traces

    sft_records = []
    num_parse_fail = 0
    num_validate_fail = 0
    num_clean_term = 0
    for i, (trace, sample_result) in enumerate(zip(clean, sample_results)):
        response_tokens = sample_result.sequences[0].tokens
        parsed_message, parse_termination = renderer.parse_response(response_tokens)
        if parse_termination.is_clean:
            num_clean_term += 1

        content = parsed_message["content"]
        visible = ""
        if isinstance(content, list):
            for part in content:
                if part["type"] == "text":
                    visible += part["text"]
        else:
            visible = content

        parsed_json = extract_json(visible)
        if parsed_json is None:
            num_parse_fail += 1
            continue
        turns = splice_turns(parsed_json, trace["thinking"], trace["response"])
        if turns is None:
            num_validate_fail += 1
            continue
        sft_records.append(
            {
                "dataset_index": trace["dataset_index"],
                "question": trace["question"],
                "ground_truth": trace["ground_truth"],
                "turns": turns,
                "num_tool_calls": len(turns) - 1,
            }
        )
        if (i + 1) % 100 == 0 or (i + 1) == len(clean):
            logger.info(
                f"Processed {i + 1}/{len(clean)} "
                f"(records: {len(sft_records)}, "
                f"parse_fail: {num_parse_fail}, "
                f"validate_fail: {num_validate_fail})"
            )

    cadence_hist: dict[int, int] = {}
    for r in sft_records:
        cadence_hist[r["num_tool_calls"]] = cadence_hist.get(r["num_tool_calls"], 0) + 1
    logger.info(f"Tool-call cadence distribution: {sorted(cadence_hist.items())}")
    logger.info(
        f"Teacher clean termination: {num_clean_term}/{len(clean)}, "
        f"parse_fail: {num_parse_fail}, validate_fail: {num_validate_fail}, "
        f"final records: {len(sft_records)}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sft_records, f, indent=2)
    logger.info(f"Saved {len(sft_records)} SFT records to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
