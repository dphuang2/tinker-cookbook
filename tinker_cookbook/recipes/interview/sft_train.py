"""
SFT recipe: teach Qwen3-30B-A3B to interleave `progress_update` tool calls
into its thinking on DeepMath problems.

Reads /tmp/tinker-examples/interview/sft_dataset.json (output of
teacher_rewrite.py) and trains a LoRA adapter on multi-turn conversations
where each assistant turn ends with either a `progress_update` tool call
or a final answer.

Usage:
    python -m tinker_cookbook.recipes.interview.sft_train
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import chz
import tinker

from tinker_cookbook import cli_utils, hyperparam_utils, model_info, renderers
from tinker_cookbook.renderers import Message, Renderer, ToolSpec, TrainOnWhat
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

import datasets

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SFT_DATASET_PATH = "/tmp/tinker-examples/interview/sft_dataset.json"
PURE_MATH_PATH = "/tmp/tinker-examples/interview/deepmath_train_traces.json"
PURE_MATH_COUNT = 0  # 0004 found mixing collapsed cadence; default off
LOG_PATH = "/tmp/tinker-examples/interview/sft_run"
MAX_LENGTH = 32768
BATCH_SIZE = 16
NUM_EPOCHS = 1
LORA_RANK = 32
MIN_TOTAL_THINKING_CHARS = 0  # 0014 found filter hurts (data not fungible)
MAX_TOOL_RECORDS = 0  # 0147 confirmed even 20 records kills cadence; disabled

PROGRESS_TOOL_SPEC: ToolSpec = {
    "name": "checkpoint",
    "description": (
        "Pause your thinking to record a checkpoint summarizing where you "
        "are in your reasoning. This is for YOUR OWN bookkeeping while you "
        "work through the problem -- use it whenever you finish a logical "
        "subtask, switch approach, or want to consolidate progress. Call it "
        "freely; the user will read the summaries to follow along."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": (
                    "One short first-person sentence describing the current "
                    "reasoning state, e.g. 'Tried u-substitution but the "
                    "cross term didn't cancel - switching to partial fractions.'"
                ),
            },
        },
        "required": ["message"],
    },
}

USER_INSTRUCTION_SUFFIX = (
    " Think step by step, then write your final answer in \\boxed{} format. "
    "Use the checkpoint tool *between* reasoning steps -- pause, summarize "
    "where you are in one sentence, then keep thinking. About three "
    "checkpoints spread through your work is typical. Write the boxed "
    "answer as soon as you have it."
)

SYSTEM_PROMPT = (
    "You are solving competition math problems. You have access to a "
    "checkpoint tool for tracking progress."
)  # 0159 variance rerun #27 of 0105

# 0018: mask loss on <think> block tokens to preserve base reasoning capability.
# Qwen3 token IDs for the thinking-block boundaries.
THINK_OPEN_TOKEN = 151667  # <think>
THINK_CLOSE_TOKEN = 151668  # </think>
MASK_THINKING_LOSS = False  # 0018 showed masking thinking caused cadence runaway


def _user_message(question: str) -> Message:
    return {"role": "user", "content": question + USER_INSTRUCTION_SUFFIX}


def _assistant_turn_with_update(
    thinking: str, summary: str, call_id: str
) -> Message:
    # 0002: revert to v1-style `message`-only tool args (no `reasoning`
    # duplication). Prior thinking will not survive into the next turn
    # via tool_call args; that's the v1 baseline behavior.
    tool_call = ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(
            name="progress_update",
            arguments=json.dumps({"message": summary}),
        ),
    )
    return {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": thinking}],
        "tool_calls": [tool_call],
    }


def _assistant_turn_final(thinking: str, final_text: str) -> Message:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": thinking},
            {"type": "text", "text": final_text},
        ],
    }


def _tool_ack(call_id: str) -> Message:
    return {"role": "tool", "content": "ok", "tool_call_id": call_id}


def _mask_thinking_weights(datum: tinker.Datum) -> tinker.Datum:
    """Zero loss weights for tokens within <think>...</think> spans.

    The supervised loss is computed at each target token position. With
    LAST_ASSISTANT_MESSAGE, weights are non-zero only on the last assistant
    message tokens. We further zero out positions whose corresponding INPUT
    token is the thinking content (after <think> and before </think>).
    This trains the model to emit the tool call / final answer but does not
    train it to reproduce specific thinking tokens, preserving base reasoning.
    """
    if not MASK_THINKING_LOSS:
        return datum
    input_tokens = datum.model_input.to_ints()
    weights = list(datum.loss_fn_inputs["weights"].data)
    # Weights are aligned to targets: weight[i] is the loss for predicting
    # token at position i+1. So a thinking-span boundary at token position k
    # in the input means: weights[k-1] predicts that token, and weights[k]
    # predicts what follows it. We mask weights[k] where input_tokens[k+1]
    # is inside a thinking span (so we don't train to predict thinking tokens).
    # Simpler: walk the input, track whether we're inside <think>...</think>,
    # and zero weights[i-1] (the prediction of input_tokens[i]) when
    # input_tokens[i] is between <think> and </think> exclusive of the tags.
    inside = False
    n = len(weights)
    for i, tok in enumerate(input_tokens):
        if tok == THINK_OPEN_TOKEN:
            inside = True
            continue
        if tok == THINK_CLOSE_TOKEN:
            inside = False
            continue
        if inside and i > 0 and i - 1 < n:
            weights[i - 1] = 0.0
    import numpy as np

    new_weights = tinker.types.TensorData.from_numpy(
        np.array(weights, dtype=np.float32)
    )
    new_loss_fn_inputs = dict(datum.loss_fn_inputs)
    new_loss_fn_inputs["weights"] = new_weights
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs=new_loss_fn_inputs,
    )


def pure_math_record_to_datum(
    record: dict, renderer: Renderer, max_length: int
) -> tinker.Datum:
    """Build one Datum for a plain Qwen3 thinking trace (no tool calls).

    Mixing these into training preserves the model's "just answer when
    confident" behavior and reduces erosion of base reasoning capability.
    """
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[PROGRESS_TOOL_SPEC], system_prompt=SYSTEM_PROMPT
    )
    messages: list[Message] = list(prefix)
    messages.append(_user_message(record["question"]))
    messages.append(_assistant_turn_final(record["thinking"], record["response"]))
    return conversation_to_datum(
        messages, renderer, max_length, TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )


def record_to_datums(
    record: dict, renderer: Renderer, max_length: int
) -> list[tinker.Datum]:
    """Build K Datums per record, one per assistant turn.

    Qwen3's native renderer strips <think> from non-last assistant messages.
    To train each turn on its true inference-time prefix, we emit one
    LAST_ASSISTANT_MESSAGE datum per turn. Earlier turns appear as
    stripped-thinking history -- and the previous turn's reasoning survives
    via the tool_call's `reasoning` argument (which is preserved by the
    renderer), so the model can resume from the prior reasoning state.
    """
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[PROGRESS_TOOL_SPEC], system_prompt=SYSTEM_PROMPT
    )
    history: list[Message] = list(prefix)
    history.append(_user_message(record["question"]))
    datums: list[tinker.Datum] = []
    for i, turn in enumerate(record["turns"]):
        if "progress_update" in turn:
            call_id = f"call_{i}"
            asst = _assistant_turn_with_update(
                turn["thinking"], turn["progress_update"], call_id
            )
            datums.append(
                _mask_thinking_weights(
                    conversation_to_datum(
                        history + [asst],
                        renderer,
                        max_length,
                        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                )
            )
            history.append(asst)
            history.append(_tool_ack(call_id))
        else:
            asst = _assistant_turn_final(turn["thinking"], turn["final"])
            datums.append(
                _mask_thinking_weights(
                    conversation_to_datum(
                        history + [asst],
                        renderer,
                        max_length,
                        TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                )
            )
    return datums


@chz.chz
class InterviewSFTBuilder(ChatDatasetBuilder):
    """Builds the progress-update SFT dataset from the teacher-rewrite JSON."""

    file_path: str = SFT_DATASET_PATH
    pure_math_path: str = PURE_MATH_PATH
    pure_math_count: int = PURE_MATH_COUNT
    min_total_thinking_chars: int = MIN_TOTAL_THINKING_CHARS
    max_tool_records: int = MAX_TOOL_RECORDS
    test_size: int = 100
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        with open(self.file_path) as f:
            tool_records_all = json.load(f)
        if self.min_total_thinking_chars > 0:
            tool_records = []
            for r in tool_records_all:
                total = sum(len(t.get("thinking", "")) for t in r["turns"])
                if total >= self.min_total_thinking_chars:
                    tool_records.append(r)
            logger.info(
                f"Filtered tool-call records by thinking>={self.min_total_thinking_chars}: "
                f"{len(tool_records)}/{len(tool_records_all)} kept"
            )
        else:
            tool_records = tool_records_all
        if self.max_tool_records > 0 and len(tool_records) > self.max_tool_records:
            # Deterministic subsample using shuffle_seed.
            import random

            rng = random.Random(self.shuffle_seed)
            rng.shuffle(tool_records)
            tool_records = tool_records[: self.max_tool_records]
            logger.info(
                f"Subsampled tool-call records to {self.max_tool_records} "
                f"(seed={self.shuffle_seed})"
            )
        for r in tool_records:
            r["_kind"] = "tool"
        logger.info(f"Loaded {len(tool_records)} tool-call records from {self.file_path}")

        pure_records: list[dict] = []
        if self.pure_math_count > 0:
            with open(self.pure_math_path) as f:
                raw = json.load(f)
            clean = [r for r in raw if r.get("parse_termination") == "stop_sequence"]
            pure_records = clean[: self.pure_math_count]
            for r in pure_records:
                r["_kind"] = "pure"
            logger.info(
                f"Loaded {len(pure_records)} pure-math records (from {len(clean)} clean)"
            )

        all_records = tool_records + pure_records
        ds = datasets.Dataset.from_list([{"record": json.dumps(r)} for r in all_records])
        ds = ds.shuffle(seed=self.shuffle_seed)

        if self.test_size > 0 and len(ds) > self.test_size:
            test_ds = ds.take(self.test_size)
            train_ds = ds.skip(self.test_size)
        else:
            train_ds = ds
            test_ds = None
        logger.info(
            f"Split: train={len(train_ds)} test={len(test_ds) if test_ds is not None else 0}"
        )

        max_length = self.common_config.max_length
        renderer = self.renderer

        def flatmap_fn(row: dict) -> list[tinker.Datum]:
            rec = json.loads(row["record"])
            if rec.get("_kind") == "pure":
                return [pure_math_record_to_datum(rec, renderer, max_length)]
            return record_to_datums(rec, renderer, max_length)

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, flatmap_fn=flatmap_fn
        )
        test_dataset = (
            SupervisedDatasetFromHFDataset(test_ds, batch_size=len(test_ds), flatmap_fn=flatmap_fn)
            if test_ds is not None
            else None
        )
        return train_dataset, test_dataset


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = InterviewSFTBuilder(common_config=common_config)

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": LOG_PATH,
            "model_name": MODEL_NAME,
            "renderer_name": renderer_name,
            "dataset_builder": dataset,
            "learning_rate": hyperparam_utils.get_lr(MODEL_NAME, is_lora=True),
            "lora_rank": LORA_RANK,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "eval_every": 20,
        }
    )


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="overwrite")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
