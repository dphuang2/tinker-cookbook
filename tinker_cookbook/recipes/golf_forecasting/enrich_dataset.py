"""Enrich existing JSONL dataset files with per-player round-by-round scores and
hole-by-hole data fetched from the ESPN tournament detail endpoint.

Reads every unique tournament_id from the existing JSONL files, fetches the
detailed ESPN scoreboard for each PGA-tour numeric ID, and patches the records
in-place with:
  - player.holes          : list of {hole, score, to_par}
  - player.scorecard_compact : compact scorecard string
  - player.round_history  : {R1: -3, R2: -5, ...} for previous rounds

Non-PGA IDs (prefixed with 'eur-' or 'lpga-') are skipped for now (those use
different ESPN league slugs and have fewer missing-data issues).

Usage:
    python -m tinker_cookbook.recipes.golf_forecasting.enrich_dataset \
        [--data-dir tinker_cookbook/example_data/golf_forecasting]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ESPN_PGA_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"

CACHE_DIR = Path("tinker_cookbook/example_data/golf_forecasting/raw/espn_detail")


def _fetch_tournament(tournament_id: str) -> dict | None:
    """Fetch detailed ESPN scoreboard for a single numeric PGA tournament ID."""
    cache_path = CACHE_DIR / f"{tournament_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    url = f"{ESPN_PGA_BASE}/{tournament_id}"
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code in (404, 500):
            logger.warning("  ESPN 404/500 for %s", tournament_id)
            return None
        resp.raise_for_status()
        data = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, sort_keys=True))
        return data
    except Exception as exc:
        logger.warning("  Failed to fetch %s: %s", tournament_id, exc)
        return None


def _parse_score_to_par(s: str) -> int | None:
    if not s or s in ("E", "-", "--", "WD", "DQ", "CUT"):
        return 0 if s == "E" else None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_holes(linescore: dict) -> list[dict]:
    """Extract sorted hole scores from a round linescore."""
    holes = []
    for h in linescore.get("linescores", []):
        period = h.get("period")
        if period is None:
            continue
        p = int(period)
        if not 1 <= p <= 18:
            continue
        raw = h.get("value")
        if raw is None:
            continue
        st = h.get("scoreType", {}).get("displayValue", "E")
        to_par = _parse_score_to_par(st)
        if to_par is None:
            to_par = 0
        holes.append({"hole": p, "score": int(raw), "to_par": to_par})
    return sorted(holes, key=lambda x: x["hole"])


def _compact(holes: list[dict]) -> str:
    parts = []
    for h in sorted(holes, key=lambda x: x["hole"]):
        tp = h["to_par"]
        parts.append("E" if tp == 0 else (f"+{tp}" if tp > 0 else str(tp)))
    return " ".join(parts)


def _build_player_detail_map(event_data: dict) -> dict[str, dict]:
    """Return {normalized_name: {R1: score, R2: score, ..., holes_by_round: {1: [...]}}}."""
    comp = (event_data.get("competitions") or [{}])[0]
    result: dict[str, dict] = {}
    for c in comp.get("competitors", []):
        name = (c.get("athlete") or {}).get("displayName", "")
        if not name:
            continue
        key = name.lower().strip()
        round_scores: dict[str, int] = {}
        holes_by_round: dict[int, list[dict]] = {}
        for ls in c.get("linescores", []):
            period = ls.get("period")
            if period is None:
                continue
            rnum = int(period)
            if not 1 <= rnum <= 5:
                continue
            disp = ls.get("displayValue", "E")
            val = _parse_score_to_par(disp)
            if val is not None:
                round_scores[f"R{rnum}"] = val
                holes = _parse_holes(ls)
                if holes:
                    holes_by_round[rnum] = holes
        if round_scores:
            result[key] = {
                "round_scores": round_scores,
                "holes_by_round": holes_by_round,
            }
    return result


def _enrich_record(record: dict, detail_map: dict[str, dict]) -> dict:
    """Patch one JSONL record with round-history and hole-by-hole data."""
    snap_round = int(record.get("round_number", 1))
    players_out = []
    for p in record.get("players", []):
        key = p["name"].lower().strip()
        detail = detail_map.get(key)
        enriched = dict(p)
        if detail:
            rs = detail["round_scores"]
            # Round history: previous completed rounds
            history = {k: v for k, v in rs.items() if int(k[1:]) < snap_round}
            if history:
                enriched["round_history"] = history
            # Hole-by-hole for the snap round
            holes = detail["holes_by_round"].get(snap_round, [])
            if holes:
                enriched["holes"] = holes
                enriched["scorecard_compact"] = _compact(holes)
        players_out.append(enriched)
    record = dict(record)
    record["players"] = players_out
    return record


def _is_pga_id(tid: str) -> bool:
    return tid.isdigit()


def enrich_jsonl(path: Path, detail_cache: dict[str, dict]) -> int:
    """Enrich all records in a JSONL file. Returns count of enriched records."""
    lines = path.read_text().splitlines()
    enriched_count = 0
    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        tid = str(record.get("tournament_id", ""))
        detail_map = detail_cache.get(tid)
        if detail_map:
            record = _enrich_record(record, detail_map)
            enriched_count += 1
        out_lines.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(out_lines) + "\n")
    return enriched_count


def main(data_dir: str = "tinker_cookbook/example_data/golf_forecasting") -> None:
    data_path = Path(data_dir)
    target_files = [
        data_path / "train.jsonl",
        data_path / "val.jsonl",
        data_path / "heldout.jsonl",
        Path("tinker_cookbook/recipes/golf_forecasting/anchor_eval_heldout.jsonl"),
    ]

    # Collect all unique PGA tournament IDs across all files
    all_tids: set[str] = set()
    for fp in target_files:
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tid = str(rec.get("tournament_id", ""))
            if _is_pga_id(tid):
                all_tids.add(tid)

    logger.info("Found %d unique PGA tournament IDs", len(all_tids))

    # Fetch + cache ESPN detail for each
    detail_cache: dict[str, dict] = {}
    for i, tid in enumerate(sorted(all_tids)):
        logger.info("[%d/%d] Fetching detail for tournament %s", i + 1, len(all_tids), tid)
        data = _fetch_tournament(tid)
        if data:
            detail_cache[tid] = _build_player_detail_map(data)
            logger.info("  → %d players with detail data", len(detail_cache[tid]))
        # Rate limit
        time.sleep(0.2)

    logger.info("Fetched detail for %d tournaments", len(detail_cache))

    # Enrich each JSONL file
    for fp in target_files:
        if not fp.exists():
            logger.warning("File not found, skipping: %s", fp)
            continue
        count = enrich_jsonl(fp, detail_cache)
        logger.info("Enriched %s: %d records updated", fp, count)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="tinker_cookbook/example_data/golf_forecasting",
    )
    args = parser.parse_args()
    main(args.data_dir)
