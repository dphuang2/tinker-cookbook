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
LOG_PATH = "/tmp/tinker-examples/interview/sft_run"
MAX_LENGTH = 32768
BATCH_SIZE = 16
NUM_EPOCHS = 1
LORA_RANK = 32

PROGRESS_TOOL_SPEC: ToolSpec = {
    "name": "progress_update",
    "description": (
        "Pause your reasoning to record a checkpoint, then resume. Use this "
        "between major reasoning steps; do NOT call it once you've reached a "
        "confident final answer (in that case, end your thinking and give the "
        "boxed answer directly). The arguments capture both a user-facing "
        "summary AND the underlying reasoning, so the reasoning persists when "
        "you resume on the next turn."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "One short first-person sentence for the user, e.g. "
                    "'Tried u-substitution but the cross term didn't cancel "
                    "- switching to partial fractions.'"
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "The full reasoning content from the segment of thinking "
                    "you just completed. This is preserved in your context "
                    "across turns so you can pick up exactly where you left "
                    "off after the tool returns."
                ),
            },
        },
        "required": ["summary", "reasoning"],
    },
}

USER_INSTRUCTION_SUFFIX = (
    " Write your answer in \\boxed{} format. Don't think for too long "
    "unnecessarily, especially when you have a reasonable degree of confidence."
)


def _user_message(question: str) -> Message:
    return {"role": "user", "content": question + USER_INSTRUCTION_SUFFIX}


def _assistant_turn_with_update(
    thinking: str, summary: str, call_id: str
) -> Message:
    # The tool_call's `reasoning` argument carries the same content as the
    # <think> block. Qwen3 strips <think> from non-last assistant messages
    # at render time, but tool_call arguments are preserved -- so when this
    # turn becomes history, the reasoning survives via the tool call.
    tool_call = ToolCall(
        id=call_id,
        function=ToolCall.FunctionBody(
            name="progress_update",
            arguments=json.dumps({"summary": summary, "reasoning": thinking}),
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
        tools=[PROGRESS_TOOL_SPEC], system_prompt=""
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
                conversation_to_datum(
                    history + [asst],
                    renderer,
                    max_length,
                    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
            )
            history.append(asst)
            history.append(_tool_ack(call_id))
        else:
            asst = _assistant_turn_final(turn["thinking"], turn["final"])
            datums.append(
                conversation_to_datum(
                    history + [asst],
                    renderer,
                    max_length,
                    TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
            )
    return datums


@chz.chz
class InterviewSFTBuilder(ChatDatasetBuilder):
    """Builds the progress-update SFT dataset from the teacher-rewrite JSON."""

    file_path: str = SFT_DATASET_PATH
    test_size: int = 100
    shuffle_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        with open(self.file_path) as f:
            records = json.load(f)
        logger.info(f"Loaded {len(records)} SFT records from {self.file_path}")

        ds = datasets.Dataset.from_list([{"record": json.dumps(r)} for r in records])
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
            return record_to_datums(json.loads(row["record"]), renderer, max_length)

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
