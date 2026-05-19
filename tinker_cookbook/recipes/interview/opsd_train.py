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

import json
import logging
from typing import Any

import chz
import tinker

from tinker_cookbook.recipes.interview.sft_train import (
    PROGRESS_TOOL_SPEC,
    SYSTEM_PROMPT,
    USER_INSTRUCTION_SUFFIX,
)

logger = logging.getLogger(__name__)


# Three teacher modes for Phase A ablation:
#   "answer_plus_placement" (A, default): teacher sees ground-truth +
#     explicit even-split directive. Strongest KL signal but teacher's
#     reasoning may be unrealistic (already knows the answer).
#   "placement_only" (B): teacher sees the same problem as the student
#     plus only the placement directive — no answer. Realistic reasoning
#     but teacher distribution may not differ enough from student to
#     produce useful KL signal.
#   "answer_as_if_discovering" (C): teacher sees the answer and is told
#     to "produce reasoning as if discovering the answer". Hybrid.

TEACHER_SUFFIX_A = (
    " The verified answer is: {answer}. Produce a single coherent "
    "reasoning trace that derives this answer, with three checkpoint "
    "calls placed roughly one-third and two-thirds of the way through "
    "your thinking and one just before the boxed answer. Each "
    "checkpoint should mark a genuine transition in your reasoning, "
    "and the chunks of thinking between checkpoints should be roughly "
    "equal in length."
)  # 0183: reverted concise wording (no effect in 0182); lowering kl_coef instead

TEACHER_SUFFIX_B = (
    " Structure your reasoning into three roughly equal chunks "
    "separated by checkpoint calls: emit the first checkpoint after "
    "you've established the problem setup and approach, the second "
    "after the main derivation, and the third just before writing the "
    "boxed answer. Each checkpoint should mark a genuine transition; "
    "the chunks of thinking between them should be roughly equal in "
    "length."
)

TEACHER_SUFFIX_C = (
    " The verified answer is: {answer}. Produce the reasoning trace "
    "that derives this answer *as if you were discovering it for the "
    "first time* — show natural exploration including any false starts "
    "you would realistically attempt. Place three checkpoint calls at "
    "roughly equal intervals: after problem setup, after the main "
    "derivation, and just before the boxed answer."
)

TEACHER_SUFFIXES = {
    "answer_plus_placement": TEACHER_SUFFIX_A,
    "placement_only": TEACHER_SUFFIX_B,
    "answer_as_if_discovering": TEACHER_SUFFIX_C,
}

# Default for backward compatibility with smoke tests / 0173-0175 notes
TEACHER_PRIVILEGED_SUFFIX = TEACHER_SUFFIX_A


def make_teacher_user_message(
    question: str,
    answer: str,
    mode: str = "answer_plus_placement",
) -> dict:
    """Construct the privileged-info user message for the teacher.

    Args:
        question: the math problem.
        answer: ground-truth answer (ignored in placement_only mode).
        mode: one of TEACHER_SUFFIXES.keys().
    """
    if mode not in TEACHER_SUFFIXES:
        raise ValueError(
            f"Unknown teacher_mode {mode!r}; expected one of "
            f"{sorted(TEACHER_SUFFIXES)}"
        )
    suffix_template = TEACHER_SUFFIXES[mode]
    # placement_only doesn't have {answer} placeholder
    suffix = (
        suffix_template.format(answer=answer)
        if "{answer}" in suffix_template
        else suffix_template
    )
    return {
        "role": "user",
        "content": question + USER_INSTRUCTION_SUFFIX + suffix,
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

    # Teacher mode (A, B, or C — see TEACHER_SUFFIXES)
    teacher_mode: str = "answer_plus_placement"

    # IO
    log_path: str = "/tmp/tinker-examples/interview/opsd_run"
    save_every: int = 20
    eval_every: int = 20


async def roll_out_student(
    *,
    sampling_client: tinker.SamplingClient,
    renderer: Any,
    tokenizer: Any,
    sample_params: tinker.SamplingParams,
    question: str,
    max_turns: int,
) -> dict:
    """One full multi-turn rollout from the student, mirroring exactly the
    behavior of `eval_deepmath_agent.run_agent`:
      - PROGRESS_TOOL_SPEC exposed
      - SYSTEM_PROMPT + USER_INSTRUCTION_SUFFIX (no privileged info)
      - state-aware ack: "noted; continue your reasoning" until the 4th
        tool call, then "you've checkpointed enough; finalize your
        answer now" (matches 0170 best recipe)
      - returns full token sequence, prompt token boundaries, decoded
        text, and per-turn metadata so the OPSD training step can do
        teacher logprob scoring + per-token KL.

    Output dict shape:
        {
            "question": str,
            "history": list[Message],
            "all_tokens": list[int],   # concatenated assistant tokens
            "turn_token_ranges": list[(start, end)],  # within all_tokens
            "decoded": str,            # concat of decoded turns
            "n_tool_calls": int,
            "n_turn_splits": int,
            "in_think_calls": int,
            "tool_call_char_positions": list[int],
        }
    """
    from tinker_cookbook.renderers import Message

    tools = [PROGRESS_TOOL_SPEC]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tools, system_prompt=SYSTEM_PROMPT
    )
    history: list[Message] = list(prefix)
    history.append({"role": "user", "content": question + USER_INSTRUCTION_SUFFIX})

    all_tokens: list[int] = []
    all_logprobs: list[float] = []  # student sample-time logprob per token
    turn_token_ranges: list[tuple[int, int]] = []
    decoded_concat = ""
    n_turns_with_tool_calls = 0
    in_think_calls = 0
    tool_call_char_positions: list[int] = []
    total_chars = 0

    for turn_idx in range(max_turns):
        prompt_input = renderer.build_generation_prompt(history)
        result = await sampling_client.sample_async(
            prompt=prompt_input,
            num_samples=1,
            sampling_params=sample_params,
        )
        seq = result.sequences[0]
        tokens = seq.tokens
        logprobs = list(seq.logprobs)
        start = len(all_tokens)
        all_tokens.extend(tokens)
        all_logprobs.extend(logprobs)
        turn_token_ranges.append((start, len(all_tokens)))

        decoded_turn = tokenizer.decode(tokens)
        # record tool_call char positions within concatenated decoded stream
        search_start = 0
        while True:
            idx = decoded_turn.find("<tool_call>", search_start)
            if idx < 0:
                break
            tool_call_char_positions.append(total_chars + idx)
            search_start = idx + 1
        think_close = decoded_turn.find("</think>")
        if think_close >= 0:
            in_think_calls += decoded_turn.count("<tool_call>", 0, think_close)
        else:
            in_think_calls += decoded_turn.count("<tool_call>")
        total_chars += len(decoded_turn)
        decoded_concat += decoded_turn

        parsed, _termination = renderer.parse_response(tokens)
        history.append(parsed)

        tool_calls = parsed.get("tool_calls") or []
        if tool_calls:
            n_turns_with_tool_calls += 1
            for tc in tool_calls:
                # 0170-style state-aware ack throttle
                ack = (
                    "you've checkpointed enough; finalize your answer now"
                    if (sum(1 for _ in tool_call_char_positions) >= 4)
                    else "noted; continue your reasoning"
                )
                history.append({
                    "role": "tool",
                    "content": ack,
                    "tool_call_id": tc.id or f"call_{turn_idx}",
                })
            continue
        break

    return {
        "question": question,
        "history": history,
        "all_tokens": all_tokens,
        "all_logprobs": all_logprobs,  # student sample-time logprobs
        "turn_token_ranges": turn_token_ranges,
        "decoded": decoded_concat,
        "n_tool_calls": len(tool_call_char_positions),
        "n_turn_splits": n_turns_with_tool_calls,
        "in_think_calls": in_think_calls,
        "tool_call_char_positions": tool_call_char_positions,
    }


async def score_with_teacher(
    *,
    rollout: dict,
    answer: str,
    teacher_sampling_client: tinker.SamplingClient,
    renderer: Any,
    teacher_mode: str = "answer_plus_placement",
) -> dict:
    """Compute the teacher's per-token logprobs over the student's
    generated tokens, with the teacher conditioned on the privileged
    prompt (which contains the ground-truth answer).

    The teacher and student share renderer/tokenizer (same base model).
    We take the student's full conversation history, swap the first
    user message for the privileged version, render it, and ask the
    teacher to score the concatenation.

    Returns:
        {
            "teacher_logprobs": list[float],  # length == len(student tokens)
            "student_token_mask": list[int],  # 1 where token was student-generated, 0 for ack/system
            "n_assistant_tokens": int,
        }

    The reverse-KL training step would then compute:
        kl_per_token = student_logprob - teacher_logprob
        advantage = -kl_per_token * student_token_mask
    """
    history = list(rollout["history"])

    # find the first user message and swap to privileged version
    swapped = False
    for i, msg in enumerate(history):
        if msg.get("role") == "user":
            original = msg["content"]
            # strip out the trailing USER_INSTRUCTION_SUFFIX before re-adding;
            # the question itself is what came before that suffix
            if original.endswith(USER_INSTRUCTION_SUFFIX):
                question = original[: -len(USER_INSTRUCTION_SUFFIX)]
            else:
                question = original
            new = make_teacher_user_message(question, answer, mode=teacher_mode)
            history[i] = new
            swapped = True
            break
    if not swapped:
        raise ValueError("rollout history has no user message to swap")

    # Render with a preserve-thinking renderer so the teacher sees the
    # student's historical <think>...</think> blocks (default Qwen3
    # renderer strips them, matching HF behavior).
    from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
    preserve_renderer = Qwen3Renderer(
        renderer.tokenizer, strip_thinking_from_history=False
    )
    # Build the model input that would generate one more assistant turn.
    # That input contains all prior history including assistant thinking.
    model_input = preserve_renderer.build_generation_prompt(history)
    sequence_tokens = list(model_input.to_ints())

    # Call the teacher to score
    teacher_logprobs = await teacher_sampling_client.compute_logprobs_async(
        tinker.ModelInput.from_ints(sequence_tokens)
    )

    # student_token_mask: rough heuristic — mark the *last len(all_tokens)*
    # positions as student-generated. Will be refined to account for tool
    # ack tokens being interleaved.
    n_student = len(rollout["all_tokens"])
    mask = [0] * (len(sequence_tokens) - n_student) + [1] * n_student
    if len(mask) != len(sequence_tokens):
        # adjust if lengths drifted; pad/clip to match
        mask = mask[: len(sequence_tokens)] + [0] * max(
            0, len(sequence_tokens) - len(mask)
        )

    return {
        "teacher_logprobs": list(teacher_logprobs),
        "student_token_mask": mask,
        "n_assistant_tokens": n_student,
        "sequence_len": len(sequence_tokens),
        "sequence_tokens": sequence_tokens,
    }


def assemble_opsd_datum(
    *,
    rollout: dict,
    scored: dict,
    kl_penalty_coef: float = 1.0,
) -> "tinker.Datum":
    """Construct a single tinker.Datum for one student rollout.

    Per-token advantage = -kl_penalty_coef * (student_logprob - teacher_logprob)
    on student-generated positions only; 0 elsewhere.

    The Datum is constructed in the rightshifted/leftshifted convention
    that Tinker's training loss functions expect:
        model_input  = sequence_tokens[:-1]
        target_tokens = sequence_tokens[1:]
        logprobs / advantages / mask = aligned with target_tokens

    Args:
        rollout: output of roll_out_student (must contain all_tokens
            and all_logprobs)
        scored: output of score_with_teacher (must contain
            teacher_logprobs, student_token_mask, sequence_len)
        kl_penalty_coef: scale on the KL advantage term
    """
    import torch
    from tinker import TensorData

    # The teacher-scored sequence is what we built in score_with_teacher:
    # teacher-privileged prefix + student tokens. We re-build the same
    # token sequence here for self-consistency.
    # NOTE: scored["sequence_len"] is the total scored length; the last
    # n_assistant_tokens positions are the student-generated tokens.
    seq_len = scored["sequence_len"]
    n_student = scored["n_assistant_tokens"]
    teacher_logprobs = scored["teacher_logprobs"]
    mask_full = scored["student_token_mask"]  # length == seq_len
    student_logprobs = rollout["all_logprobs"]
    assert len(student_logprobs) == n_student, (
        f"student logprobs ({len(student_logprobs)}) != n_student ({n_student})"
    )

    sequence_tokens = scored["sequence_tokens"]
    assert len(sequence_tokens) == seq_len, (
        f"sequence_tokens len ({len(sequence_tokens)}) != seq_len ({seq_len})"
    )

    # Rightshifted model_input = sequence_tokens[:-1]
    # Leftshifted targets    = sequence_tokens[1:]
    input_tokens = sequence_tokens[:-1]
    target_tokens = sequence_tokens[1:]

    # Per-position: student_logprob (only on student tokens, else 0),
    # teacher_logprob (from scored, len == seq_len), mask (1 on student
    # positions only). We align to target_tokens (length seq_len-1).
    # Student logprobs cover the last n_student target positions.
    student_lp_padded = [0.0] * (len(target_tokens) - n_student) + list(student_logprobs)
    teacher_lp_aligned = list(teacher_logprobs)[1:]  # drop first to align with target_tokens
    mask_aligned = list(mask_full)[1:]
    assert len(student_lp_padded) == len(teacher_lp_aligned) == len(mask_aligned) == len(target_tokens)

    # Replace any None in teacher_logprobs with 0.0; the mask will zero
    # them out for non-student positions anyway.
    teacher_lp_clean = [lp if lp is not None else 0.0 for lp in teacher_lp_aligned]

    # reverse_kl = student_lp - teacher_lp; advantage = -kl_coef * mask * reverse_kl
    advantages = [
        -kl_penalty_coef * float(m) * (float(s) - float(t))
        for s, t, m in zip(student_lp_padded, teacher_lp_clean, mask_aligned)
    ]

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(student_lp_padded, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask_aligned, dtype=torch.float32)),
        },
    )


async def main(config: OPSDConfig) -> None:
    """Run Phase A OPSD training.

    Outer loop: every step samples `groups_per_batch` problems from
    DeepMath train indices, rolls them out under the student prompt,
    scores them with the privileged-info teacher prompt, computes
    reverse-KL advantages, and runs one forward_backward + optim_step.

    Sampler weights are saved every `save_every` steps; the sampling
    client is recreated after each save so subsequent rollouts use the
    updated policy.
    """
    import asyncio
    import json
    import random
    from pathlib import Path

    import tinker
    from datasets import load_dataset

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    log_dir = Path(config.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(
        {k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
        default=str, indent=2))

    logger.info(f"OPSD config: {config}")

    # Dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=42)
    train_problems = [
        ds[i] for i in range(config.train_index_start, config.train_index_end)
    ]
    logger.info(f"Loaded {len(train_problems)} training problems")

    # Renderer / tokenizer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Clients
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        config.model_name, rank=config.lora_rank,
    )
    sample_params = tinker.SamplingParams(
        max_tokens=config.max_tokens_per_turn,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # Initial sampler: save current (random LoRA) weights and create client
    save_fut = await training_client.save_weights_for_sampler_async(
        f"step_0", ttl_seconds=86400,
    )
    sampler_path = (await save_fut.result_async()).path
    sampling_client = await service.create_sampling_client_async(model_path=sampler_path)
    logger.info(f"Initial sampler at {sampler_path}")

    # Same sampling client used as teacher (same base + LoRA — the
    # privileged prompt is what makes teacher distribution differ).
    teacher_client = sampling_client

    checkpoints = []
    metrics_log = []
    rng = random.Random(0xC0FFEE)
    SCORE_TOKEN_BUDGET = 28000

    for step in range(config.max_steps):
        logger.info(f"=== step {step + 1}/{config.max_steps} ===")
        # Sample a batch of problems uniformly
        batch_problems = rng.sample(train_problems, config.groups_per_batch)

        # Roll out student concurrently
        rollout_tasks = [
            roll_out_student(
                sampling_client=sampling_client,
                renderer=renderer,
                tokenizer=tokenizer,
                sample_params=sample_params,
                question=p["question"],
                max_turns=config.max_turns,
            ) for p in batch_problems
        ]
        rollouts = await asyncio.gather(*rollout_tasks, return_exceptions=True)

        # Filter overlong / failed rollouts
        keep_pairs: list[tuple[dict, dict]] = []
        skipped = {"error": 0, "overlong": 0}
        for problem, r in zip(batch_problems, rollouts):
            if isinstance(r, Exception):
                skipped["error"] += 1
                continue
            n_tok = sum(e - s for s, e in r["turn_token_ranges"])
            if n_tok > SCORE_TOKEN_BUDGET:
                skipped["overlong"] += 1
                continue
            keep_pairs.append((problem, r))
        logger.info(
            f"step {step + 1}: rollouts kept={len(keep_pairs)} "
            f"skipped_error={skipped['error']} skipped_overlong={skipped['overlong']}"
        )
        if not keep_pairs:
            logger.warning(f"step {step + 1}: no usable rollouts, skipping")
            continue

        # Score each with teacher; assemble datums
        score_tasks = [
            score_with_teacher(
                rollout=r,
                answer=str(p.get("final_answer", p.get("ground_truth", "?"))),
                teacher_sampling_client=teacher_client,
                renderer=renderer,
                teacher_mode=config.teacher_mode,
            ) for p, r in keep_pairs
        ]
        scored_list = await asyncio.gather(*score_tasks, return_exceptions=True)

        datums = []
        for (p, r), scored in zip(keep_pairs, scored_list):
            if isinstance(scored, Exception):
                logger.warning(f"  teacher scoring failed: {scored}")
                continue
            try:
                d = assemble_opsd_datum(
                    rollout=r, scored=scored, kl_penalty_coef=config.kl_penalty_coef,
                )
                datums.append(d)
            except Exception as e:
                logger.warning(f"  datum assembly failed: {e}")
                continue

        if not datums:
            logger.warning(f"step {step + 1}: no usable datums after scoring, skipping")
            continue

        # forward_backward (strip mask) + optim_step
        datums_for_train = [
            tinker.Datum(
                model_input=d.model_input,
                loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
            )
            for d in datums
        ]
        fwd_bwd_fut = await training_client.forward_backward_async(
            datums_for_train, loss_fn="importance_sampling", loss_fn_config=None,
        )
        optim_fut = await training_client.optim_step_async(adam_params)
        fwd_result = await fwd_bwd_fut.result_async()
        await optim_fut.result_async()

        # Quick metric: mean training logprob (proxy for loss)
        out_lps = [out["logprobs"].to_torch() for out in fwd_result.loss_fn_outputs]
        mean_logprob = sum(lp.float().mean().item() for lp in out_lps) / len(out_lps)
        logger.info(
            f"step {step + 1}: trained on {len(datums)} datums, "
            f"mean_training_logprob={mean_logprob:.4f}"
        )
        metrics_log.append({"step": step + 1, "n_datums": len(datums),
                            "mean_training_logprob": mean_logprob,
                            **skipped})
        (log_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in metrics_log) + "\n"
        )

        # Save sampler periodically and rebind sampling_client
        if (step + 1) % config.save_every == 0 or (step + 1) == config.max_steps:
            save_fut = await training_client.save_weights_for_sampler_async(
                f"step_{step + 1}", ttl_seconds=86400,
            )
            sampler_path = (await save_fut.result_async()).path
            sampling_client = await service.create_sampling_client_async(
                model_path=sampler_path,
            )
            teacher_client = sampling_client
            checkpoints.append({"step": step + 1, "sampler_path": sampler_path})
            with open(log_dir / "checkpoints.jsonl", "a") as f:
                f.write(json.dumps(checkpoints[-1]) + "\n")
            logger.info(f"Saved sampler at step {step + 1}: {sampler_path}")

    logger.info(f"OPSD training complete. {len(checkpoints)} sampler checkpoints saved.")


if __name__ == "__main__":
    import asyncio
    cli_config = chz.entrypoint(OPSDConfig)
    asyncio.run(main(cli_config))
