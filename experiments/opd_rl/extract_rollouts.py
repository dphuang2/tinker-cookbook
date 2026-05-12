"""Walk a run's iteration_*/train_logtree.json files and extract compact
rollout records: (iter, problem_prompt, response, reward). Writes a single
`rollouts.jsonl` per run that fits in <5MB and lets reviewers see what the
student actually said at every step.

Usage:
    uv run python -m experiments.opd_rl.extract_rollouts \
        --src /tmp/dylan/opd-rl/iter07-... \
        --dst experiments/opd_rl/data/iter07/rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _walk_content(node, out: list[str]):
    """Collect every string under any 'content' key. Order = walk order."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "content" and isinstance(v, str):
                out.append(v)
            else:
                _walk_content(v, out)
    elif isinstance(node, list):
        for x in node:
            _walk_content(x, out)


def _find_rewards(node, out: list[float]):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("final_reward", "total_reward") and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                _find_rewards(v, out)
    elif isinstance(node, list):
        for x in node:
            _find_rewards(x, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="run log_path directory")
    ap.add_argument("--dst", required=True, help="output rollouts.jsonl")
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    n_traj = 0
    with dst.open("w") as fout:
        for iter_dir in sorted(src.glob("iteration_*")):
            lt = iter_dir / "train_logtree.json"
            if not lt.exists():
                continue
            iter_idx = int(re.search(r"\d+", iter_dir.name).group(0))
            data = json.loads(lt.read_text())
            # The structure alternates user/assistant content blocks per trajectory.
            # Pair them up by walk order, and try to align with rewards from the
            # rollout_summaries.jsonl sibling for canonical reward values.
            summaries_path = iter_dir / "train_rollout_summaries.jsonl"
            rewards = []
            if summaries_path.exists():
                for line in summaries_path.open():
                    rec = json.loads(line)
                    rewards.append(rec.get("final_reward"))
            contents: list[str] = []
            _walk_content(data, contents)
            # Heuristic: walk picks up file path + each user/assistant content.
            # Strip the path entry and any short headers.
            contents = [c for c in contents if len(c) > 30 and not c.endswith(".html")]
            # Group consecutive (user, assistant) pairs.
            i = 0
            traj_idx = 0
            while i + 1 < len(contents):
                prompt = contents[i]
                response = contents[i + 1]
                # Sanity: prompt should mention "Using each of the numbers" (our env).
                if "Using each of the numbers" in prompt:
                    rec = {
                        "iter": iter_idx,
                        "traj_idx": traj_idx,
                        "prompt": prompt,
                        "response": response,
                        "reward": rewards[traj_idx] if traj_idx < len(rewards) else None,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_traj += 1
                    traj_idx += 1
                    i += 2
                else:
                    i += 1
    print(f"Wrote {n_traj} trajectories → {dst} ({dst.stat().st_size / 1024:.1f}KB)")


if __name__ == "__main__":
    main()
