"""Fetch PGA Tour leaderboard data from the public ESPN API and produce JSONL
records in the format expected by `build_dataset.normalize_records`.

Usage:
    python -m tinker_cookbook.recipes.golf_forecasting.fetch_espn_data \
        --output /tmp/golf_espn_raw.jsonl \
        --seasons 2024,2025

Each output record represents a mid-tournament leaderboard snapshot (after R2 or
R3) with the eventual winner stored in `target_winner`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"

# Skip non-standard events (team events, match play, Q-school, etc.)
SKIP_KEYWORDS = {
    "zurich classic",       # team event
    "presidents cup",
    "ryder cup",
    "q-school",
    "showdown",
}


def _get_calendar(season_range: str) -> list[dict]:
    """Return list of {id, label, startDate} for a season."""
    url = f"{ESPN_BASE}/scoreboard?dates={season_range}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    leagues = data.get("leagues", [])
    if not leagues:
        return []
    return leagues[0].get("calendar", [])


def _get_event(event_id: str) -> dict | None:
    """Fetch full scoreboard for a single event."""
    url = f"{ESPN_BASE}/scoreboard/{event_id}"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _parse_score_to_par(score_str: str) -> int | None:
    """Parse ESPN score string like '-35', '+2', 'E' to integer."""
    if not score_str or score_str in ("E", "-", "--", "WD", "DQ", "CUT"):
        return 0 if score_str == "E" else None
    try:
        return int(score_str)
    except ValueError:
        return None


def _build_snapshot_after_round(
    event_data: dict,
    snapshot_round: int,
) -> dict | None:
    """Build a leaderboard snapshot using cumulative scores through `snapshot_round`.

    Returns a record dict in the format expected by normalize_records, or None
    if the data is insufficient.
    """
    event_name = event_data["name"]
    event_id = event_data["id"]
    event_date = event_data.get("date", "")
    comp = event_data.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])

    if not competitors:
        return None

    # Determine the winner (order=1 in final standings)
    winner = None
    for c in competitors:
        if c.get("order") == 1:
            winner = c["athlete"]["displayName"]
            break
    if winner is None:
        # fallback: first competitor
        winner = competitors[0]["athlete"]["displayName"]

    # Build cumulative scores through snapshot_round
    player_records = []
    for c in competitors:
        linescores = c.get("linescores", [])
        # linescores are round-level objects; each has a displayValue (score relative to par)
        if len(linescores) < snapshot_round:
            continue  # player didn't make it to this round

        cumulative_score = 0
        round_scores = []
        skip_player = False
        for r_idx in range(snapshot_round):
            ls = linescores[r_idx]
            round_display = ls.get("displayValue", "E")
            round_val = _parse_score_to_par(round_display)
            if round_val is None:
                skip_player = True
                break
            cumulative_score += round_val
            round_scores.append(round_val)
        if skip_player:
            continue

        player_records.append({
            "name": c["athlete"]["displayName"],
            "cumulative_to_par": cumulative_score,
            "round_scores": round_scores,
            "last_round_score": round_scores[-1] if round_scores else 0,
        })

    if len(player_records) < 5:
        return None

    # Sort by cumulative score (lowest = best)
    player_records.sort(key=lambda p: p["cumulative_to_par"])
    leader_score = player_records[0]["cumulative_to_par"]

    players = []
    for pos_idx, p in enumerate(player_records):
        behind = p["cumulative_to_par"] - leader_score
        total_rounds = 4  # standard PGA event
        holes_per_round = 18
        holes_completed = snapshot_round * holes_per_round
        holes_remaining = (total_rounds - snapshot_round) * holes_per_round

        players.append({
            "name": p["name"],
            "position": str(pos_idx + 1),
            "score_to_par": p["cumulative_to_par"],
            "strokes_behind": behind,
            "holes_completed": holes_completed,
            "holes_remaining": holes_remaining,
            "round_score": p["last_round_score"],
        })

    day_map = {2: "Saturday", 3: "Sunday"}
    snapshot_ts = event_date  # approximate

    record = {
        "example_id": f"{event_id}-r{snapshot_round}",
        "tournament_id": str(event_id),
        "tournament_name": event_name,
        "round_number": snapshot_round,
        "event_day": day_map.get(snapshot_round, f"Round {snapshot_round}"),
        "snapshot_timestamp": snapshot_ts,
        "players": players,
        "target_winner": winner,
        "source_urls": [f"{ESPN_BASE}/scoreboard/{event_id}"],
    }
    return record


def _should_skip(label: str) -> bool:
    lower = label.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


def fetch_season(season_year: int) -> list[dict]:
    """Fetch all completed events for a season and create snapshot records."""
    date_range = f"{season_year}0101-{season_year}1231"
    calendar = _get_calendar(date_range)
    logger.info("Found %d calendar entries for %d", len(calendar), season_year)

    records = []
    for entry in calendar:
        event_id = entry.get("id", "")
        label = entry.get("label", "")
        start_date = entry.get("startDate", "")

        if _should_skip(label):
            logger.info("Skipping %s (%s)", label, event_id)
            continue

        # Only fetch events that have already ended
        try:
            end_date = entry.get("endDate", start_date)
            event_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if event_end > datetime.now(timezone.utc):
                logger.info("Skipping future/in-progress event: %s", label)
                continue
        except (ValueError, TypeError):
            pass

        logger.info("Fetching %s (%s)...", label, event_id)
        event_data = _get_event(event_id)
        if event_data is None:
            logger.warning("Could not fetch event %s", event_id)
            continue

        # Create snapshots after round 2 and round 3
        for snap_round in [2, 3]:
            record = _build_snapshot_after_round(event_data, snap_round)
            if record is not None:
                records.append(record)
                logger.info(
                    "  Created R%d snapshot: %d players, winner=%s",
                    snap_round,
                    len(record["players"]),
                    record["target_winner"],
                )

        # Rate limit: be nice to ESPN's API
        time.sleep(0.5)

    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch ESPN PGA Tour data")
    parser.add_argument("--output", type=str, default="/tmp/golf_espn_raw.jsonl")
    parser.add_argument("--seasons", type=str, default="2024,2025")
    args = parser.parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    all_records: list[dict] = []
    for season in seasons:
        records = fetch_season(season)
        all_records.extend(records)
        logger.info("Season %d: %d snapshot records", season, len(records))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in all_records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")

    logger.info("Wrote %d records to %s", len(all_records), output_path)


if __name__ == "__main__":
    main()
