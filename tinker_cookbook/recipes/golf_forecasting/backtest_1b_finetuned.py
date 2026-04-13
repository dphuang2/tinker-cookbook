"""Backtest exp80 fine-tuned 1B and DeepSeek teacher on Masters 2026 using the binary+margin prompt."""

from __future__ import annotations

import asyncio
import json
import os

import tinker
from tinker import types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.golf_forecasting.env import parse_forecast_response, score_forecast
from tinker_cookbook.tokenizer_utils import get_tokenizer


# --- Step 1: Build backtest data from ESPN ---

async def fetch_and_build_examples() -> list[dict]:
    """Fetch Masters 2026 from ESPN and build 3 round snapshots."""
    import urllib.request

    url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard/401811941"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    # API returns the event directly (not nested under "events")
    competitors = data["competitions"][0]["competitors"]

    # Winner is order=1
    winner_name = None
    for c in competitors:
        if c.get("order") == 1:
            winner_name = c["athlete"]["displayName"]
            break
    if not winner_name:
        winner_name = competitors[0]["athlete"]["displayName"]

    examples = []
    for round_num in [1, 2, 3]:
        players = []
        for c in competitors:
            name = c["athlete"]["displayName"]
            linescores = c.get("linescores", [])
            if len(linescores) < round_num:
                continue
            # Skip players with 0-value rounds (missed the cut / didn't play)
            round_scores = linescores[:round_num]
            if any(ls.get("value", 0) < 50 for ls in round_scores):
                continue
            # Cumulative score to par through this round
            cumulative = sum(ls.get("value", 0) - 72 for ls in round_scores)
            players.append({
                "name": name,
                "score_to_par": cumulative,
            })

        # Sort by score (lowest first)
        players.sort(key=lambda p: p["score_to_par"])
        leader_score = players[0]["score_to_par"]
        for p in players:
            p["strokes_behind"] = p["score_to_par"] - leader_score
            p["position"] = ""  # Will be assigned below

        # Assign positions
        for i, p in enumerate(players):
            p["position"] = str(i + 1)
            p["holes_remaining"] = (4 - round_num) * 18

        examples.append({
            "example_id": f"masters2026-r{round_num}",
            "tournament_name": "Masters Tournament",
            "round_number": round_num,
            "players": players,
            "target_winner": winner_name,
            "snapshot_timestamp": f"2026-04-{8 + round_num}T19:00:00Z",
        })

    return examples


# --- Step 2 & 3: Run models ---

def build_binary_margin_prompt(example: dict) -> list[dict[str, str]]:
    """The exact prompt format exp79/80 was trained on."""
    players = example["players"]
    leader = players[0]
    leader_score = leader["score_to_par"]
    margin = players[1]["strokes_behind"] if len(players) >= 2 else 0
    within_3 = sum(1 for p in players if p["strokes_behind"] <= 3)
    total = len(players)

    system = "You are a calibrated golf forecaster. Return JSON only."
    user = (
        f"Golf: {example['tournament_name']}, after round {example['round_number']} of 4.\n"
        f"Leader: {leader['name']} at {leader_score:+.0f}\n"
        f"Lead margin: {margin:.0f} stroke(s) over 2nd place\n"
        f"Players within 3 strokes: {within_3}\n"
        f"Total field: {total}\n"
        f"\n"
        f"What probability does the leader win the tournament?\n"
        f'Return JSON: {{"winner_probs": {{"{leader["name"]}": probability, "other": 1-probability}}}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def run_model(examples, model_name, checkpoint_url=None, renderer_name=None, max_tokens=256, label=""):
    """Run a model on examples and return results."""
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    client = tinker.ServiceClient()
    if checkpoint_url:
        sampling_client = client.create_sampling_client(
            model_path=checkpoint_url, base_model=model_name,
        )
    else:
        sampling_client = client.create_sampling_client(base_model=model_name)

    params = types.SamplingParams(
        max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences(),
    )

    results = []
    for ex in examples:
        messages = build_binary_margin_prompt(ex)
        prompt = renderer.build_generation_prompt(messages)
        response = await sampling_client.sample_async(
            prompt=prompt, num_samples=1, sampling_params=params,
        )
        text = renderers.get_text_content(renderer.parse_response(response.sequences[0].tokens)[0])

        leader_name = ex["players"][0]["name"]
        allowed = [leader_name, "other"]

        # Map target to "other" if winner isn't the leader
        target_winner = ex["target_winner"]
        effective_target = leader_name if target_winner.lower() == leader_name.lower() else "other"

        try:
            forecast, diag = parse_forecast_response(text, allowed_labels=allowed)
            scores = score_forecast(forecast, target_label=effective_target)
            result = {
                "example_id": ex["example_id"],
                "round": ex["round_number"],
                "leader": leader_name,
                "target_winner": target_winner,
                "effective_target": effective_target,
                "forecast": forecast,
                "log_loss": scores["log_loss"],
                "brier": scores["brier"],
                "target_prob": scores["target_prob"],
                "top1_correct": scores["top1_correct"],
                "format_valid": 1.0,
            }
        except Exception as e:
            print(f"  PARSE FAILED for {ex['example_id']}: {e}")
            print(f"  Raw: {text[:300]}")
            result = {
                "example_id": ex["example_id"],
                "round": ex["round_number"],
                "leader": leader_name,
                "target_winner": target_winner,
                "effective_target": effective_target,
                "forecast": {leader_name: 0.0, "other": 0.0},
                "log_loss": 13.8,
                "brier": 2.0,
                "target_prob": 0.0,
                "top1_correct": 0.0,
                "format_valid": 0.0,
            }

        results.append(result)

    return results


# --- Step 4: Print results ---

def print_results(results, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"{'Round':>6s}  {'Leader':>20s}  {'Leader%':>8s}  {'Other%':>8s}  {'Winner':>20s}  {'LL':>6s}  {'Brier':>6s}")
    print("-" * 65)

    for r in results:
        fc = r["forecast"]
        leader_name = r["leader"]
        leader_prob = fc.get(leader_name, 0.0)
        other_prob = fc.get("other", 0.0)
        print(
            f"  R{r['round']}   {leader_name:>20s}  {leader_prob:>7.1%}  {other_prob:>7.1%}  "
            f"{r['target_winner']:>20s}  {r['log_loss']:>5.2f}  {r['brier']:>5.3f}"
        )

    n = len(results)
    avg_ll = sum(r["log_loss"] for r in results) / n
    avg_brier = sum(r["brier"] for r in results) / n
    avg_fv = sum(r["format_valid"] for r in results) / n
    print("-" * 65)
    print(f"  Avg  {'':>20s}  {'':>8s}  {'':>8s}  {'':>20s}  {avg_ll:>5.2f}  {avg_brier:>5.3f}   fv={avg_fv:.0%}")


async def main():
    print("Fetching Masters 2026 data from ESPN...")
    examples = await fetch_and_build_examples()

    print(f"Built {len(examples)} round snapshots. Winner: {examples[0]['target_winner']}")
    for ex in examples:
        leader = ex["players"][0]
        margin = ex["players"][1]["strokes_behind"] if len(ex["players"]) >= 2 else 0
        print(f"  R{ex['round_number']}: Leader={leader['name']} ({leader['score_to_par']:+.0f}), margin={margin:.0f}")

    print("\nRunning fine-tuned 1B (exp80)...")
    results_1b = await run_model(
        examples,
        model_name="meta-llama/Llama-3.2-1B",
        checkpoint_url="tinker://6873b020-9c55-5178-9951-5f57a1edc117:train:0/sampler_weights/final",
        renderer_name="role_colon",
        max_tokens=256,
        label="1B fine-tuned",
    )

    print("\nRunning DeepSeek-V3.1 (teacher, zero-shot)...")
    results_ds = await run_model(
        examples,
        model_name="deepseek-ai/DeepSeek-V3.1",
        max_tokens=512,
        label="DeepSeek teacher",
    )

    print_results(results_1b, "Fine-tuned 1B (exp80 checkpoint)")
    print_results(results_ds, "DeepSeek-V3.1 (teacher, zero-shot)")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results", "masters2026_backtest_1b_vs_deepseek")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"1b_finetuned": results_1b, "deepseek_teacher": results_ds}, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    asyncio.run(main())
