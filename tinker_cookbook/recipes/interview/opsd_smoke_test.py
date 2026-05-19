"""Smoke test for opsd_train.roll_out_student.

Runs the multi-turn student rollout on N DeepMath train problems and
dumps the structured output to JSON. Used to verify the rollout shape
matches eval_deepmath_agent.py before wiring teacher logprob scoring.

Usage:
    LD_PRELOAD=.../libnccl.so.2 .venv/bin/python \\
      -m tinker_cookbook.recipes.interview.opsd_smoke_test
"""

from __future__ import annotations

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
from tinker_cookbook.recipes.interview.opsd_train import (
    roll_out_student, score_with_teacher, assemble_opsd_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
N_PROBLEMS = int(os.environ.get("N", "3"))
TRAIN_INDEX_START = 500
MAX_TOKENS_PER_TURN = 24576
TEMPERATURE = 0.6
MAX_TURNS = 8
OUT_DIR = Path(__file__).parent / "opsd_smoke_out"


async def main():
    load_dotenv()
    OUT_DIR.mkdir(exist_ok=True)

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

    tasks = []
    for i in range(N_PROBLEMS):
        idx = TRAIN_INDEX_START + i
        problem = ds[idx]
        tasks.append(roll_out_student(
            sampling_client=sc,
            renderer=renderer,
            tokenizer=tokenizer,
            sample_params=params,
            question=problem["question"],
            max_turns=MAX_TURNS,
        ))

    print(f"Rolling out {N_PROBLEMS} student trajectories concurrently...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary = []
    for i, r in enumerate(results):
        idx = TRAIN_INDEX_START + i
        if isinstance(r, Exception):
            print(f"  idx={idx}: FAILED {r!r}")
            continue
        # write per-problem file
        out_path = OUT_DIR / f"idx{idx:04d}.json"
        # don't dump all_tokens (large); dump shape summary
        summary_record = {
            "index": idx,
            "question": r["question"][:200],
            "n_turns": len(r["turn_token_ranges"]),
            "n_tokens_total": sum(e - s for s, e in r["turn_token_ranges"]),
            "n_tool_calls": r["n_tool_calls"],
            "n_turn_splits": r["n_turn_splits"],
            "in_think_calls": r["in_think_calls"],
            "tool_call_char_positions": r["tool_call_char_positions"],
            "decoded_head": r["decoded"][:800],
            "decoded_tail": r["decoded"][-400:],
        }
        out_path.write_text(json.dumps(summary_record, indent=2))
        summary.append({
            "idx": idx,
            "turns": summary_record["n_turns"],
            "calls": summary_record["n_tool_calls"],
            "splits": summary_record["n_turn_splits"],
            "in_think": summary_record["in_think_calls"],
            "tokens": summary_record["n_tokens_total"],
        })
        print(f"  idx={idx} turns={summary_record['n_turns']} "
              f"calls={summary_record['n_tool_calls']} "
              f"splits={summary_record['n_turn_splits']} "
              f"tokens={summary_record['n_tokens_total']}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT_DIR}/summary.json")

    # also smoke-test score_with_teacher on the FIRST successful rollout
    # using the SAME sampling client as the "teacher" (since v3 Phase A
    # uses the same base model for both teacher and student; the
    # privileged prompt is what makes the teacher distribution differ).
    first_ok_idx = None
    for i, r in enumerate(results):
        if not isinstance(r, Exception):
            first_ok_idx = i
            break
    if first_ok_idx is not None:
        idx = TRAIN_INDEX_START + first_ok_idx
        problem = ds[idx]
        answer = problem.get("final_answer", problem.get("ground_truth", "?"))
        print(f"Smoke-testing score_with_teacher on idx={idx} (answer={answer!r})...")
        scored = await score_with_teacher(
            rollout=results[first_ok_idx],
            answer=str(answer),
            teacher_sampling_client=sc,
            renderer=renderer,
        )
        print(f"  teacher_logprobs len={len(scored['teacher_logprobs'])} "
              f"n_assistant_tokens={scored['n_assistant_tokens']} "
              f"sequence_len={scored['sequence_len']} "
              f"mask_sum={sum(scored['student_token_mask'])}")
        # log mean teacher logprob on student tokens (sanity: should be
        # a reasonable per-token log-prob, e.g. -1 to -3)
        import statistics
        student_lps = [
            lp for lp, m in zip(scored["teacher_logprobs"], scored["student_token_mask"])
            if m == 1 and lp is not None
        ]
        if student_lps:
            print(f"  mean teacher logprob on student tokens: "
                  f"{statistics.mean(student_lps):.3f} "
                  f"(over {len(student_lps)} non-None tokens)")
        (OUT_DIR / f"teacher_score_idx{idx:04d}.json").write_text(
            json.dumps({
                "index": idx,
                "answer": str(answer),
                "n_assistant_tokens": scored["n_assistant_tokens"],
                "sequence_len": scored["sequence_len"],
                "mean_student_logprob": (statistics.mean(student_lps)
                                         if student_lps else None),
            }, indent=2)
        )

        # Smoke-test assemble_opsd_datum
        print("Smoke-testing assemble_opsd_datum...")
        datum = assemble_opsd_datum(
            rollout=results[first_ok_idx], scored=scored, kl_penalty_coef=1.0,
        )
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()
        n_masked = int(mask.sum().item())
        nonzero_adv = adv[mask.bool()]
        print(f"  Datum built. model_input.length={datum.model_input.length} "
              f"n_masked_positions={n_masked} "
              f"adv_mean(masked)={nonzero_adv.mean().item():.4f} "
              f"adv_std(masked)={nonzero_adv.std().item():.4f} "
              f"adv_min={nonzero_adv.min().item():.4f} "
              f"adv_max={nonzero_adv.max().item():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
