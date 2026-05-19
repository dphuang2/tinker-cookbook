"""
v3 Phase A — On-Policy Self-Distillation (OPSD) for the progress-update
interleaving task, Self-Distilled Reasoner style (arXiv 2601.18734).

Both teacher and student are Qwen3-30B-A3B with LoRA. The teacher
conditions on privileged info (the ground-truth answer + an explicit
"divide your reasoning evenly with checkpoints" directive); the student
sees only the problem with the standard v2 prompt. Reverse-KL loss on
student rollouts pushes the student's distribution toward the teacher's.

Scaffold status: WIP. The Tinker cookbook's distillation pipeline
(`tinker_cookbook.distillation.train_on_policy`) gives the teacher the
SAME prompt as the student (see `incorporate_kl_penalty` constructing
`full_sequence_inputs_D` from `datum.model_input`). To do the
privileged-info trick we need to override that — either by:
  (a) building a custom training loop here that calls Tinker's lower-
      level training_client / sampling_client APIs directly, or
  (b) constructing student `model_input` whose prompt is the privileged
      teacher prompt (so the teacher sees it natively) and pre-pending
      a "mask out the privileged tail before sampling" stage on the
      student side.

We're going with (a) because it's cleaner and lets us also customize
the agent loop (student's rollouts must do multi-turn tool calls to
match eval, not single-shot completions).

Open TODOs (driven by the autoresearch loop):
  - [ ] Implement multi-turn agent loop for student rollouts so the
        sample distribution matches eval_deepmath_agent.py
        (PROGRESS_TOOL_SPEC exposed, MAX_TURNS=8, ack content from 0170).
  - [ ] Implement teacher logprob scoring on student sequences with the
        privileged prompt.
  - [ ] Compute per-token reverse-KL advantages, train via
        training_client.forward_backward.
  - [ ] Save checkpoints; emit sampler_path on each save for eval reuse.
  - [ ] Add held-out eval hook every N steps using
        eval_deepmath_agent.py against `runs/<NNNN>-*` slot.

Usage (once implemented):
    LD_PRELOAD=... uv run python -m \\
        tinker_cookbook.recipes.interview.opsd_train \\
        learning_rate=1e-4 groups_per_batch=64 lora_rank=32
"""

from __future__ import annotations

import logging

import chz

from tinker_cookbook.recipes.interview.sft_train import (
    PROGRESS_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_SUFFIX,
)

logger = logging.getLogger(__name__)


# Privileged prompt addendum the teacher sees that the student does not.
# Keeps the student's USER_INSTRUCTION_SUFFIX wording, then adds the
# ground-truth answer and an explicit even-split directive.
TEACHER_PRIVILEGED_SUFFIX = (
    " The verified answer is: {answer}. Produce a single coherent "
    "reasoning trace that derives this answer, with three checkpoint "
    "calls placed roughly one-third and two-thirds of the way through "
    "your thinking and one just before the boxed answer. Each "
    "checkpoint should mark a genuine transition in your reasoning, "
    "and the chunks of thinking between checkpoints should be roughly "
    "equal in length."
)


def make_teacher_user_message(question: str, answer: str) -> dict:
    """Construct the privileged-info user message for the teacher."""
    return {
        "role": "user",
        "content": (
            question
            + USER_INSTRUCTION_SUFFIX
            + TEACHER_PRIVILEGED_SUFFIX.format(answer=answer)
        ),
    }


def make_student_user_message(question: str) -> dict:
    """Construct the regular user message for the student (no privileged
    info). This must match exactly what eval_deepmath_agent.py sends so
    train/eval distributions agree."""
    return {
        "role": "user",
        "content": question + USER_INSTRUCTION_SUFFIX,
    }


@chz.chz
class OPSDConfig:
    """Configuration for Phase A OPSD training."""

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Data
    train_index_start: int = 500     # DeepMath indices 0-499 are eval (held-out)
    train_index_end: int = 3000      # 2500 training problems
    group_size: int = 4
    groups_per_batch: int = 32

    # Sampling
    max_tokens_per_turn: int = 24576
    temperature: float = 0.6
    max_turns: int = 8

    # Optimization
    learning_rate: float = 1e-4
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    num_substeps: int = 1
    max_steps: int = 100

    # IO
    log_path: str = "/tmp/tinker-examples/interview/opsd_run"
    save_every: int = 20
    eval_every: int = 20


async def main(config: OPSDConfig) -> None:
    """Run Phase A OPSD training.

    NOT YET IMPLEMENTED — see module docstring for TODO list. The next
    autoresearch tick should pick the first TODO and land it.
    """
    raise NotImplementedError(
        "opsd_train.main is scaffolding only. See module docstring for "
        "the TODO sequence the autoresearch loop should iterate on."
    )


if __name__ == "__main__":
    import asyncio
    cli_config = chz.entrypoint(OPSDConfig)
    asyncio.run(main(cli_config))
