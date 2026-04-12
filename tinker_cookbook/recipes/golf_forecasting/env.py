from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import tinker
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.recipes.golf_forecasting.data import (
    GolfForecastExample,
    candidate_labels,
    format_compact_scorecard,
    leaderboard_table,
    normalize_player_name,
    scorecard_momentum,
)

# ---------------------------------------------------------------------------
# Pressure profile: cached player lead-hold statistics.
# Loaded lazily from artifacts/pressure_profiles.json relative to this file.
# ---------------------------------------------------------------------------
_PRESSURE_PROFILES: dict[str, dict] | None = None
_PRESSURE_PROFILES_NORMALIZED: dict[str, dict] | None = None
_TOURNAMENT_HISTORY: dict[str, list] | None = None
_PLAYER_COURSE_HISTORY: dict[str, dict] | None = None
_PLAYER_QUALITY: dict[str, dict] | None = None
_PLAYER_RECENT_FORM: dict[str, list] | None = None
_TRAINING_EXAMPLES_R3: list | None = None


def _load_pressure_profiles() -> dict[str, dict]:
    global _PRESSURE_PROFILES, _PRESSURE_PROFILES_NORMALIZED
    if _PRESSURE_PROFILES is not None:
        return _PRESSURE_PROFILES_NORMALIZED  # type: ignore[return-value]
    artifacts_dir = Path(__file__).parent / "artifacts"
    profiles_path = artifacts_dir / "pressure_profiles.json"
    if not profiles_path.exists():
        _PRESSURE_PROFILES = {}
        _PRESSURE_PROFILES_NORMALIZED = {}
        return {}
    raw = json.loads(profiles_path.read_text())
    # Handle both formats: flat dict or nested {_metadata, profiles}
    if "_metadata" in raw:
        profiles_dict = raw.get("profiles", {})
    else:
        profiles_dict = raw
    _PRESSURE_PROFILES = profiles_dict
    # Build a normalized-name -> profile lookup for fuzzy matching
    _PRESSURE_PROFILES_NORMALIZED = {
        normalize_player_name(name): profile for name, profile in profiles_dict.items()
    }
    return _PRESSURE_PROFILES_NORMALIZED
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)

FORECAST_SYSTEM_PROMPT = (
    "You are a calibrated golf forecasting assistant. "
    "Read the live leaderboard snapshot and produce a probability distribution over likely winners. "
    "Return JSON only."
)

# Tournaments that use Modified Stableford scoring (higher score = better position).
# Standard golf is stroke play (lower score = better); these are the exceptions.
_STABLEFORD_TOURNAMENTS: frozenset[str] = frozenset(
    {
        "Barracuda Championship",
    }
)


class WinnerForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner_probs: dict[str, float]

    @field_validator("winner_probs")
    @classmethod
    def validate_probs(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("winner_probs must not be empty")
        if any(prob < 0.0 for prob in value.values()):
            raise ValueError("winner_probs cannot contain negative probabilities")
        return value


def _load_player_quality() -> dict[str, dict]:
    """Load player quality scores from artifacts/player_quality.json."""
    global _PLAYER_QUALITY
    if _PLAYER_QUALITY is not None:
        return _PLAYER_QUALITY
    path = Path(__file__).parent / "artifacts" / "player_quality.json"
    if not path.exists():
        _PLAYER_QUALITY = {}
        return {}
    raw = json.loads(path.read_text())
    _PLAYER_QUALITY = raw.get("players", {})
    return _PLAYER_QUALITY


def _load_training_examples_r3() -> list:
    """Load R3 training examples lazily for few-shot calibration.

    Only R3 examples are useful for few-shot: R3 margin→winner outcomes are the
    most direct precedent for the prediction task.
    """
    global _TRAINING_EXAMPLES_R3
    if _TRAINING_EXAMPLES_R3 is not None:
        return _TRAINING_EXAMPLES_R3
    try:
        from tinker_cookbook.recipes.golf_forecasting.data import load_examples

        train_path = Path(__file__).parent.parent.parent / "example_data" / "golf_forecasting" / "train.jsonl"
        if not train_path.exists():
            _TRAINING_EXAMPLES_R3 = []
            return []
        all_examples = load_examples(str(train_path))
        _TRAINING_EXAMPLES_R3 = [e for e in all_examples if e.round_number == 3]
    except Exception:
        _TRAINING_EXAMPLES_R3 = []
    return _TRAINING_EXAMPLES_R3


def _build_few_shot_section(
    example,
    *,
    n_examples: int = 2,
) -> str:
    """Build a few-shot calibration section from similar historical R3 scenarios.

    Finds training examples with the same R3 lead margin (or within ±1 stroke),
    from tournaments before the current year, and formats them as concrete
    historical precedents to anchor the model's probability estimates.
    """
    if example.round_number != 3:
        return ""

    scores = [p.score_to_par for p in example.players]
    if len(scores) < 2:
        return ""
    margin = int(scores[1] - scores[0])

    current_year = example.snapshot_timestamp[:4] if example.snapshot_timestamp else None
    current_tid = example.tournament_id

    training = _load_training_examples_r3()
    if not training:
        return ""

    # Collect candidates: same margin ±0 first, then ±1, to find n_examples
    candidates: list = []
    for delta in (0, 1):
        for ex in training:
            if ex.tournament_id == current_tid:
                continue
            if current_year and ex.snapshot_timestamp and ex.snapshot_timestamp[:4] >= current_year:
                continue
            ex_scores = [p.score_to_par for p in ex.players]
            if len(ex_scores) < 2:
                continue
            ex_margin = int(ex_scores[1] - ex_scores[0])
            if abs(ex_margin - margin) <= delta and ex not in candidates:
                candidates.append(ex)
        if len(candidates) >= n_examples:
            break

    if not candidates:
        return ""

    selected = candidates[:n_examples]
    lines = []
    for ex in selected:
        ex_scores = [p.score_to_par for p in ex.players]
        ex_margin = int(ex_scores[1] - ex_scores[0])
        year = ex.snapshot_timestamp[:4] if ex.snapshot_timestamp else "?"
        leader_name = ex.players[0].name if ex.players else "?"
        winner = ex.target_winner or "?"
        # Determine winner's starting position
        winner_pos = None
        for i, p in enumerate(ex.players):
            if p.name == winner:
                winner_pos = i + 1
                break
        if winner == leader_name:
            outcome = f"leader {winner} won (held lead)"
        elif winner_pos:
            outcome = f"{winner} won from position {winner_pos}"
        else:
            outcome = f"{winner} won"
        lines.append(
            f"  {ex.tournament_name} {year}: leader +{ex_margin} ahead → {outcome}"
        )

    margin_desc = f"+{margin}" if margin > 0 else "tied"
    header = f"Historical R3 precedents (leader {margin_desc}, similar margin):"
    return header + "\n" + "\n".join(lines)


def _build_player_quality_section(top_names: list[str]) -> str:
    """Build a section showing each candidate's career quality context."""
    quality = _load_player_quality()
    if not quality:
        return ""
    lines = []
    for name in top_names:
        norm = normalize_player_name(name)
        # Try exact match then normalized match
        data = quality.get(name) or next(
            (v for k, v in quality.items() if normalize_player_name(k) == norm), None
        )
        if data and data.get("wins", 0) > 0:
            wins = data["wins"]
            top5 = data.get("top5", 0)
            win_rate = data.get("win_rate", 0.0)
            lines.append(f"- {name}: {wins} career win{'s' if wins != 1 else ''}, {top5} top-5s, {win_rate:.1%} win rate")
        elif data and data.get("top5", 0) > 0:
            top5 = data.get("top5", 0)
            lines.append(f"- {name}: 0 career wins, {top5} top-5s")
        else:
            lines.append(f"- {name}: limited career history in data")
    if not lines:
        return ""
    return "Career quality context:\n" + "\n".join(lines)


def _load_player_recent_form() -> dict[str, list]:
    """Load player recent form (last 5 tournament results) from artifacts."""
    global _PLAYER_RECENT_FORM
    if _PLAYER_RECENT_FORM is not None:
        return _PLAYER_RECENT_FORM
    path = Path(__file__).parent / "artifacts" / "player_recent_form.json"
    if not path.exists():
        _PLAYER_RECENT_FORM = {}
        return {}
    raw = json.loads(path.read_text())
    _PLAYER_RECENT_FORM = raw.get("form", {})
    return _PLAYER_RECENT_FORM


def _build_player_recent_form_section(
    top_names: list[str],
    *,
    current_year: str | None = None,
    max_results: int = 3,
) -> str:
    """Show each candidate's recent tournament form (last N results before current year)."""
    form_data = _load_player_recent_form()
    if not form_data:
        return ""
    lines = []
    for name in top_names:
        norm = normalize_player_name(name)
        results = form_data.get(name) or next(
            (v for k, v in form_data.items() if normalize_player_name(k) == norm), None
        )
        if not results:
            continue
        # Filter to results before current year
        filtered = [r for r in results if not current_year or r.get("year", "9999") < current_year]
        if not filtered:
            continue
        recent = filtered[:max_results]
        parts = []
        for r in recent:
            pos = r.get("position", 999)
            won = r.get("won", False) or pos == 1
            t = r.get("tournament", "?")[:20]
            if won or pos == 1:
                parts.append(f"Won ({t})")
            elif pos <= 5:
                parts.append(f"T{pos} ({t})")
            elif pos <= 10:
                parts.append(f"T{pos} ({t})")
            else:
                parts.append(f"T{pos} ({t})")
        if parts:
            lines.append(f"  {name}: {', '.join(parts)}")
    if not lines:
        return ""
    return f"Recent form (last {max_results} events, 54-hole position):\n" + "\n".join(lines)


def _load_player_course_history() -> dict[str, dict]:
    """Load per-player historical appearances at each tournament.

    Uses v1 (R3 snapshot positions from training JSONL) because R3 positions
    are the best proxy for "entering-final-round position" — matching the
    prediction scenario. v2 (final R4 positions) is available but hurts because
    it shows post-R4 outcomes which are less relevant as a prior.
    """
    global _PLAYER_COURSE_HISTORY
    if _PLAYER_COURSE_HISTORY is not None:
        return _PLAYER_COURSE_HISTORY
    path = Path(__file__).parent / "artifacts" / "player_course_history.json"
    if not path.exists():
        _PLAYER_COURSE_HISTORY = {}
        return {}
    raw = json.loads(path.read_text())
    _PLAYER_COURSE_HISTORY = raw.get("history", {})
    return _PLAYER_COURSE_HISTORY


# Tournament name aliases: handle rebrands and name changes.
_TOURNAMENT_ALIASES: dict[str, str] = {
    "Rocket Classic": "Rocket Mortgage Classic",
    "Sentry Tournament of Champions": "The Sentry",
    "The Sentry": "Sentry Tournament of Champions",
    "Fortinet Championship": "Safeway Open",
    "Safeway Open": "Fortinet Championship",
    "Shriners Children's Open": "Shriners Hospitals for Children Open",
    "Shriners Hospitals for Children Open": "Shriners Children's Open",
    "Zurich Classic of New Orleans": "Zurich Classic",
    "Zurich Classic": "Zurich Classic of New Orleans",
}


def _build_player_course_history_section(
    tournament_name: str,
    top_names: list[str],
    *,
    current_year: str | None = None,
    max_years: int = 3,
) -> str:
    """Build per-player historical performance at this specific tournament.

    Shows each candidate's R3 positions in past years at this venue,
    filtered to only include years before the current snapshot year.
    """
    history = _load_player_course_history()
    # Try original name first, then aliases
    tournament_data = history.get(tournament_name) or history.get(
        _TOURNAMENT_ALIASES.get(tournament_name, tournament_name), {}
    )
    if not tournament_data:
        return ""

    lines = []
    for name in top_names:
        normalized = normalize_player_name(name)
        # Try exact name first, then normalized
        player_records = tournament_data.get(name) or tournament_data.get(normalized)
        if not player_records:
            # Try fuzzy: match normalized key
            player_records = next(
                (v for k, v in tournament_data.items() if normalize_player_name(k) == normalized),
                None,
            )
        if not player_records:
            continue

        # Filter to years before current
        filtered = [r for r in player_records if not current_year or r.get("year", "9999") < current_year]
        if not filtered:
            continue

        # Take most recent max_years
        recent = sorted(filtered, key=lambda x: x.get("year", ""), reverse=True)[:max_years]

        parts = []
        for r in recent:
            year = r.get("year", "?")
            # v2 uses final_position + final_score; v1 uses r3_position + r3_score_to_par
            pos = r.get("final_position") or r.get("r3_position")
            won = r.get("won", False) or pos == 1
            score = r.get("final_score") if r.get("final_score") is not None else r.get("r3_score_to_par")
            score_str = f"({int(score):+d})" if score is not None else ""
            if won or pos == 1:
                parts.append(f"Won {year}{score_str}")
            elif pos:
                parts.append(f"T{pos} {year}{score_str}")
            else:
                parts.append(f"{year}")

        if parts:
            lines.append(f"  {name}: {', '.join(parts)}")

    if not lines:
        return ""

    return f"Player history at {tournament_name} (recent finishes):\n" + "\n".join(lines)


def _load_tournament_history() -> dict[str, list]:
    """Load historical tournament winners from artifacts/tournament_history.json."""
    global _TOURNAMENT_HISTORY
    if _TOURNAMENT_HISTORY is not None:
        return _TOURNAMENT_HISTORY
    history_path = Path(__file__).parent / "artifacts" / "tournament_history.json"
    if not history_path.exists():
        _TOURNAMENT_HISTORY = {}
        return {}
    raw = json.loads(history_path.read_text())
    _TOURNAMENT_HISTORY = raw.get("tournaments", {})
    return _TOURNAMENT_HISTORY


def _build_tournament_history_section(
    tournament_name: str,
    top_names: list[str],
    *,
    current_year: str | None = None,
    max_past_winners: int = 4,
) -> str:
    """Build a compact tournament history section showing past winners.

    Only shows winners from BEFORE the current snapshot year to avoid
    anachronistic data leakage. Also highlights which of the current
    top candidates have won here before.
    """
    history = _load_tournament_history()
    records = history.get(tournament_name, [])
    if not records:
        return ""

    # Filter to only show past winners from before the current year
    if current_year:
        records = [r for r in records if r.get("year", "9999") < current_year]

    if not records:
        return ""

    # Sort by year (most recent first)
    sorted_records = sorted(records, key=lambda x: x.get("year", ""), reverse=True)
    recent = sorted_records[:max_past_winners]

    past_winners = [r["winner"] for r in recent]
    winner_years = {r["winner"]: r["year"] for r in recent}

    # Check if any current top candidates have won here before
    normalized_candidates = {normalize_player_name(n): n for n in top_names}
    course_specialists = []
    for pw in past_winners:
        norm_pw = normalize_player_name(pw)
        canonical = normalized_candidates.get(norm_pw)
        if canonical:
            year = winner_years.get(pw, "?")
            course_specialists.append(f"{canonical} (won {year})")

    lines = []
    past_winners_str = ", ".join(
        f"{r['winner']} ({r['year']})" for r in recent
    )
    lines.append(f"Recent winners: {past_winners_str}")

    if course_specialists:
        lines.append(f"Current contenders who have won here: {', '.join(course_specialists)}")

    return f"Tournament history ({tournament_name}):\n" + "\n".join(f"  {l}" for l in lines)


def _build_pressure_section(
    top_names: list[str],
    *,
    round_num: int,
    max_players: int = 5,
) -> str:
    """Build a compact pressure profile section for the top players.

    Only shows data for R2/R3 snapshots (where lead-hold stats are relevant).
    Includes both lead-hold rates (for leaders) and within-3 conversion rates
    (for all players within 3 strokes of the lead).
    """
    if round_num not in (2, 3):
        return ""

    profiles = _load_pressure_profiles()
    if not profiles:
        return ""

    lines = []
    for name in top_names[:max_players]:
        normalized = normalize_player_name(name)
        profile = profiles.get(normalized)
        if profile is None:
            continue

        r3_rate = profile.get("r3_lead_hold_rate")
        r3_leads = profile.get("r3_leads", 0)
        r3_blown = profile.get("r3_blown_leads", 0)

        # Only show players with at least 2 R3 leads — minimum for meaningful signal
        if r3_leads < 2 or r3_rate is None:
            continue

        rate_str = f"{r3_rate:.0%}"
        lines.append(f"  {name}: R3 lead→win={rate_str} ({r3_leads} leads, {r3_blown} blown)")

    if not lines:
        return ""

    label = "R3" if round_num == 3 else "R2"
    return f"Historical pressure profiles (entering {label}):\n" + "\n".join(lines)


def _build_scorecard_section(
    example: GolfForecastExample,
    *,
    top_names: list[str],
    max_scorecard_players: int = 5,
    r3_only: bool = True,
) -> str:
    """Build a compact hole-by-hole scorecard section for the top players.

    Only includes players that have per-hole data. Keeps tokens compact by
    using the format: 'E -1 +1 -1 E E -1 E -2' (one token per hole).
    Also shows the last-5-hole momentum summary.

    r3_only: If True, only include scorecard data for R3 snapshots.
    Empirical finding: scorecard data helps for R3 (heading into final round)
    but slightly hurts for R2 (adds noise when 2+ rounds remain).
    """
    # A/B test showed scorecard data hurts R2 predictions — skip if r3_only
    if r3_only and example.round_number != 3:
        return ""

    # Build a name->player lookup
    player_map = {p.name: p for p in example.players}
    lines = []
    for name in top_names[:max_scorecard_players]:
        player = player_map.get(name)
        if player is None or not player.holes:
            continue
        compact = player.scorecard_compact or format_compact_scorecard(player.holes)
        momentum = scorecard_momentum(player.holes, last_n=5)
        lines.append(f"  {name}: {compact}  [{momentum}]")

    if not lines:
        return ""

    # Determine which round the scorecard covers
    round_label = f"Round {example.round_number}"
    header = (
        f"Hole-by-hole scoring (holes 1-18, {round_label}; "
        f"negative=under par, +positive=over par, E=even; "
        f"shows scoring momentum heading into the final round):"
    )
    return header + "\n" + "\n".join(lines)


def build_messages(
    example: GolfForecastExample,
    *,
    include_other_bucket: bool = True,
    max_candidates: int = 20,
    include_scorecard: bool = False,
    include_pressure: bool = True,
    include_tournament_history: bool = False,
    include_player_history: bool = False,
    include_player_quality: bool = False,
    include_recent_form: bool = False,
    include_few_shot: bool = False,
) -> list[renderers.Message]:
    # Detect Stableford tournaments (higher score = better; need descending sort)
    is_stableford = example.tournament_name in _STABLEFORD_TOURNAMENTS
    if is_stableford:
        # Re-sort players by DESCENDING score_to_par: most Stableford points first
        stableford_sorted = sorted(example.players, key=lambda p: p.score_to_par, reverse=True)
        stableford_players_override: list | None = stableford_sorted
    else:
        stableford_sorted = None
        stableford_players_override = None

    # Limit to top N players by position to keep prompt manageable
    if is_stableford and stableford_sorted is not None:
        if max_candidates > 0 and len(stableford_sorted) > max_candidates:
            top_names = [p.name for p in stableford_sorted[:max_candidates]]
        else:
            top_names = [p.name for p in stableford_sorted]
    elif max_candidates > 0 and len(example.players) > max_candidates:
        top_names = [p.name for p in example.players[:max_candidates]]
    else:
        top_names = example.candidate_names

    if include_other_bucket:
        candidates = [*top_names, "other"]
    else:
        candidates = top_names

    weather = example.system_context.get("weather_summary")
    course = example.system_context.get("course_difficulty")
    field_strength = example.system_context.get("field_strength")
    extras = []
    if weather:
        extras.append(f"- Weather: {weather}")
    if course:
        extras.append(f"- Course difficulty: {course}")
    if field_strength:
        extras.append(f"- Field strength: {field_strength}")
    extra_context = "\n".join(extras) if extras else "- No extra context provided."

    n_total = len(example.players)
    n_shown = min(max_candidates, n_total) if max_candidates > 0 else n_total

    # Compute field analysis from leaderboard.
    # For Stableford, use the descending-sorted players (highest score = leader).
    ordered_players = stableford_sorted if (is_stableford and stableford_sorted) else list(example.players)
    if is_stableford:
        # Higher score = better; leader is first, margin = leader_pts - 2nd_pts
        scores = [p.score_to_par for p in ordered_players]
        leader_score = scores[0] if scores else 0
        field_analysis_parts = []
        if len(scores) >= 2:
            gap_to_2nd = leader_score - scores[1]  # positive = leader ahead
            field_analysis_parts.append(f"Leader's margin: {gap_to_2nd:.0f} point(s) over 2nd place")
        if len(scores) >= 5:
            within_3 = sum(1 for s in scores if leader_score - s <= 3)
            field_analysis_parts.append(f"Players within 3 points of lead: {within_3}")
        if len(scores) >= 10:
            within_5 = sum(1 for s in scores if leader_score - s <= 5)
            field_analysis_parts.append(f"Players within 5 points: {within_5}")
    else:
        scores = [p.score_to_par for p in example.players]
        leader_score = scores[0] if scores else 0
        field_analysis_parts = []
        if len(scores) >= 2:
            gap_to_2nd = scores[1] - leader_score
            field_analysis_parts.append(f"Leader's margin: {gap_to_2nd:.0f} stroke(s) over 2nd place")
        if len(scores) >= 5:
            within_3 = sum(1 for s in scores if s - leader_score <= 3)
            field_analysis_parts.append(f"Players within 3 strokes of lead: {within_3}")
        if len(scores) >= 10:
            within_5 = sum(1 for s in scores if s - leader_score <= 5)
            field_analysis_parts.append(f"Players within 5 strokes: {within_5}")
    field_analysis = "\n".join(f"- {p}" for p in field_analysis_parts) if field_analysis_parts else ""

    # Round-specific calibration guidance.
    # For mc=3, hints are tuned via A/B testing (not raw empirical rates) since LLMs over-weight leaders.
    # For mc>3, hints scale "other" down proportionally (empirical: R2 outside top-5=29%, top-7=21%).
    round_num = example.round_number
    nc = n_shown  # number of named candidates in the prompt
    if round_num == 1:
        if nc <= 3:
            calibration_hint = (
                "After round 1, the leader wins only about 15-20% of the time. "
                "The field is very wide open — give significant probability to 'other'."
            )
        elif nc <= 5:
            calibration_hint = (
                "After round 1, the leader wins only about 15-20% of the time. "
                "~49% of winners come from outside the top-5 — assign ~48% to 'other'."
            )
        elif nc <= 7:
            calibration_hint = (
                "After round 1, the leader wins only about 15-20% of the time. "
                "~42% of winners come from outside the top-7 — assign ~40% to 'other'."
            )
        else:
            calibration_hint = (
                "After round 1, the leader wins only about 15-20% of the time. "
                "~37% of winners come from outside the top-10 — assign ~35% to 'other'."
            )
    elif round_num == 2:
        if nc <= 3:
            calibration_hint = (
                "After round 2, ~47% of winners come from OUTSIDE the top-3 leaderboard positions. "
                "The leader wins only 20-30% of the time. "
                "Assign at least 40-50% probability to 'other' — upsets are the norm, not the exception."
            )
        elif nc <= 5:
            calibration_hint = (
                "After round 2, ~29% of winners come from OUTSIDE the top-5 leaderboard positions. "
                "The leader wins only 20-30% of the time. "
                "Assign approximately 27-30% probability to 'other'."
            )
        elif nc <= 7:
            calibration_hint = (
                "After round 2, ~21% of winners come from OUTSIDE the top-7 leaderboard positions. "
                "The leader wins only 20-30% of the time. "
                "Assign approximately 19-22% probability to 'other'."
            )
        else:
            calibration_hint = (
                "After round 2, ~15% of winners come from OUTSIDE the top-10 leaderboard positions. "
                "The leader wins only 20-30% of the time. "
                "Assign approximately 13-16% probability to 'other'."
            )
    else:
        # Compute margin-based calibration hint using correctly-ordered players
        r3_scores = [p.score_to_par for p in ordered_players]
        if len(r3_scores) >= 2:
            if is_stableford:
                margin = int(r3_scores[0] - r3_scores[1])  # points leader is ahead (higher=better)
            else:
                margin = int(r3_scores[1] - r3_scores[0])  # strokes leader is ahead
        else:
            margin = 0
        # R3 "other" percentage scaled by candidate count (empirical: outside top-5=12%, top-7=7%)
        if nc <= 3:
            r3_other = "about 18-20%"
        elif nc <= 5:
            r3_other = "about 10-12%"
        elif nc <= 7:
            r3_other = "about 6-8%"
        else:
            r3_other = "about 4-6%"
        if margin == 0:
            calibration_hint = (
                "After round 3 with players tied for the lead: historically ~66% of the time "
                "one of the co-leaders wins. 2nd place wins 18%, top-3 cover 81%. "
                f"Assign {r3_other} to 'other'; concentrate probability on leaders."
            )
        elif margin == 1:
            calibration_hint = (
                "After round 3 with the leader ahead by 1 stroke: historically the leader "
                "wins only ~42% of the time — leads of just 1 stroke are volatile. "
                f"2nd place wins 18%, outside top-3 wins 19%. Assign {r3_other} to 'other'."
            )
        elif margin <= 2:
            calibration_hint = (
                "After round 3 with the leader ahead by 2 strokes: historically the leader "
                f"wins ~52% of the time. Assign {r3_other} to 'other', rest to top contenders."
            )
        elif margin == 3:
            calibration_hint = (
                "After round 3 with the leader ahead by 3 strokes: historically the leader "
                f"wins ~68% of the time. Top-3 covers 86%. Assign {r3_other} to 'other'."
            )
        else:
            calibration_hint = (
                f"After round 3 with the leader ahead by {margin} strokes: historically "
                "leaders with 4+ stroke leads win 80-85% of the time. "
                f"Give the leader dominant probability (~80%), assign {r3_other} to 'other'."
            )

    # Build hole-by-hole scorecard section (R3 only, when enabled)
    # A/B testing shows scorecard adds noise; disable by default, pressure profiles are more helpful.
    scorecard_section = (
        _build_scorecard_section(example, top_names=top_names[:max_candidates])
        if include_scorecard
        else ""
    )

    # Build pressure profile section for R2/R3 snapshots (enabled by default)
    pressure_section = (
        _build_pressure_section(top_names[:max_candidates], round_num=round_num)
        if include_pressure
        else ""
    )

    # Build tournament history section (disabled by default, opt-in)
    # Filter to only show winners before the current year to avoid anachronistic data
    current_year = example.snapshot_timestamp[:4] if example.snapshot_timestamp else None
    tournament_history_section = (
        _build_tournament_history_section(
            example.tournament_name,
            top_names[:max_candidates],
            current_year=current_year,
        )
        if include_tournament_history
        else ""
    )

    # Build per-player course history section (disabled by default, opt-in)
    # Shows each candidate's R3 positions in past years at this specific venue.
    player_history_section = (
        _build_player_course_history_section(
            example.tournament_name,
            top_names[:max_candidates],
            current_year=current_year,
        )
        if include_player_history
        else ""
    )

    # Build player quality section (disabled by default, opt-in)
    # Shows each candidate's career win count and win rate as a general quality proxy.
    player_quality_section = (
        _build_player_quality_section(top_names[:max_candidates])
        if include_player_quality
        else ""
    )

    # Build player recent form section (disabled by default, opt-in)
    # Shows each candidate's last 3 tournament results (54-hole position).
    recent_form_section = (
        _build_player_recent_form_section(
            top_names[:max_candidates],
            current_year=current_year,
        )
        if include_recent_form
        else ""
    )

    # Build few-shot calibration section (disabled by default, R3 only)
    # Shows 2 historical examples from training set with similar R3 lead margin.
    few_shot_section = (
        _build_few_shot_section(example)
        if include_few_shot
        else ""
    )

    stableford_note = (
        "IMPORTANT: This tournament uses MODIFIED STABLEFORD scoring. "
        "Players accumulate POINTS each hole (eagle=+5, birdie=+2, par=0, bogey=-1, double bogey=-3). "
        "The player with the MOST points wins. Higher 'Total' values are BETTER. "
        "The leaderboard below is sorted by descending points (most points = position 1).\n\n"
        if is_stableford
        else ""
    )
    instructions = (
        f"Tournament: {example.tournament_name}\n"
        f"Course: {example.course_name or 'Unknown'}\n"
        f"Round: {round_num}\n"
        f"Event day: {example.event_day or 'Unknown'}\n"
        f"Snapshot time: {example.snapshot_timestamp}\n\n"
        + stableford_note
        + f"Leaderboard snapshot (top {n_shown} of {n_total} players):\n"
        f"{leaderboard_table(example, max_players=max_candidates, players_override=stableford_players_override, is_stableford=is_stableford)}\n\n"
        + (scorecard_section + "\n\n" if scorecard_section else "")
        + (pressure_section + "\n\n" if pressure_section else "")
        + (tournament_history_section + "\n\n" if tournament_history_section else "")
        + (player_history_section + "\n\n" if player_history_section else "")
        + (player_quality_section + "\n\n" if player_quality_section else "")
        + (recent_form_section + "\n\n" if recent_form_section else "")
        + (few_shot_section + "\n\n" if few_shot_section else "")
        + "Extra context:\n"
        f"{extra_context}\n\n"
        + (f"Field analysis:\n{field_analysis}\n\n" if field_analysis else "")
        + f"Calibration guidance: {calibration_hint}\n\n"
        "Return a JSON object with a single key `winner_probs`. "
        "Each key must be one of the candidate labels below and the probabilities must sum to 1. "
        "Use `other` for all players not listed individually. "
        "Assign non-zero probability to at least 5 candidates.\n"
        f"Candidate labels: {', '.join(candidates)}\n"
        "Do not include explanations, markdown fences, or extra keys.\n"
        'Example: {"winner_probs": {"Player A": 0.30, "Player B": 0.20, "Player C": 0.15, "Player D": 0.10, "Player E": 0.08, "other": 0.17}}'
    )
    return [
        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
        {"role": "user", "content": instructions},
    ]


def _extract_json_blob(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output")
    return text[start : end + 1]


def parse_forecast_response(
    text: str,
    *,
    allowed_labels: Sequence[str],
    prob_floor: float = 0.001,
) -> tuple[dict[str, float], dict[str, float]]:
    raw = WinnerForecast.model_validate_json(_extract_json_blob(text))
    allowed_lookup = {normalize_player_name(label): label for label in allowed_labels}
    output: dict[str, float] = {label: 0.0 for label in allowed_labels}
    unknown_mass = 0.0
    for label, prob in raw.winner_probs.items():
        canonical = allowed_lookup.get(normalize_player_name(label))
        if canonical is None:
            unknown_mass += prob
            continue
        output[canonical] += prob

    total = sum(output.values()) + unknown_mass
    if total <= 0:
        raise ValueError("Total forecast probability must be positive")
    normalized = {label: prob / total for label, prob in output.items()}

    # Apply probability floor to prevent catastrophic log-loss on zero-probability labels.
    if prob_floor > 0:
        n_labels = len(normalized)
        floor_total = prob_floor * n_labels
        if floor_total < 1.0:
            scale = (1.0 - floor_total) / max(1.0 - floor_total, 1e-12)
            floored = {}
            for label, prob in normalized.items():
                floored[label] = prob_floor + prob * scale
            # Re-normalize to ensure sum=1
            ftotal = sum(floored.values())
            normalized = {label: prob / ftotal for label, prob in floored.items()}

    diagnostics = {
        "raw_total_probability": total,
        "unknown_probability_mass": unknown_mass,
    }
    return normalized, diagnostics


def compute_multiclass_brier(
    forecast: dict[str, float],
    *,
    target_label: str,
) -> float:
    total = 0.0
    for label, prob in forecast.items():
        target = 1.0 if label == target_label else 0.0
        total += (prob - target) ** 2
    return total


def compute_log_loss(
    forecast: dict[str, float],
    *,
    target_label: str,
    floor: float = 1e-6,
) -> float:
    target_prob = max(forecast.get(target_label, 0.0), floor)
    return -math.log(target_prob)


def score_forecast(
    forecast: dict[str, float],
    *,
    target_label: str,
) -> dict[str, float]:
    ordered = sorted(forecast.items(), key=lambda item: item[1], reverse=True)
    top_labels = [label for label, _ in ordered[:3]]
    brier = compute_multiclass_brier(forecast, target_label=target_label)
    log_loss = compute_log_loss(forecast, target_label=target_label)
    # Multiclass Brier ranges from 0 to 2, so divide by 2 for a [0, 1] reward.
    brier_reward = 1.0 - min(brier / 2.0, 1.0)
    return {
        "brier": brier,
        "log_loss": log_loss,
        "brier_reward": brier_reward,
        "target_prob": forecast.get(target_label, 0.0),
        "top1_correct": float(ordered[0][0] == target_label),
        "top3_contains_target": float(target_label in top_labels),
    }


class GolfForecastEnv(Env):
    def __init__(
        self,
        *,
        example: GolfForecastExample,
        renderer: renderers.Renderer,
        include_other_bucket: bool = True,
        format_coef: float = 0.1,
        max_candidates: int = 20,
        include_scorecard: bool = False,
        include_pressure: bool = True,
        include_tournament_history: bool = False,
        include_player_history: bool = False,
    ):
        self.example = example
        self.renderer = renderer
        self.include_other_bucket = include_other_bucket
        self.format_coef = format_coef
        self.max_candidates = max_candidates
        self.messages = build_messages(
            example,
            include_other_bucket=include_other_bucket,
            max_candidates=max_candidates,
            include_scorecard=include_scorecard,
            include_pressure=include_pressure,
            include_tournament_history=include_tournament_history,
            include_player_history=include_player_history,
        )
        # Build allowed labels to match what build_messages uses
        if max_candidates > 0 and len(example.players) > max_candidates:
            top_names = [p.name for p in example.players[:max_candidates]]
        else:
            top_names = example.candidate_names
        self.allowed_labels = [*top_names, "other"] if include_other_bucket else top_names

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        base_metrics: dict[str, float] = {"parse_success": float(parse_success)}
        if not parse_success:
            return self._invalid_result(
                message=message,
                metrics=base_metrics,
                reason="renderer_parse_failed",
                content=content,
            )

        try:
            forecast, diagnostics = parse_forecast_response(content, allowed_labels=self.allowed_labels)
        except (ValidationError, ValueError) as exc:
            logger.debug("Forecast parse failed for %s: %s", self.example.example_id, exc)
            return self._invalid_result(
                message=message,
                metrics=base_metrics,
                reason=f"forecast_parse_failed:{exc}",
                content=content,
            )

        # Use effective target: if winner is outside the candidate set, map to "other"
        effective_target = self.example.target_label
        if effective_target not in self.allowed_labels:
            effective_target = "other"
        scores = score_forecast(forecast, target_label=effective_target)
        reward = scores["brier_reward"]
        metrics = {
            **base_metrics,
            **scores,
            **diagnostics,
            "format_valid": 1.0,
        }
        self._log_attempt(message=message, content=content, forecast=forecast, metrics=metrics)
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    def _invalid_result(
        self,
        *,
        message: renderers.Message,
        metrics: dict[str, float],
        reason: str,
        content: str,
    ) -> StepResult:
        invalid_metrics = {
            **metrics,
            "format_valid": 0.0,
            "brier": 2.0,
            "log_loss": compute_log_loss({}, target_label=self.example.target_label),
            "brier_reward": 0.0,
            "target_prob": 0.0,
            "top1_correct": 0.0,
            "top3_contains_target": 0.0,
            "raw_total_probability": 0.0,
            "unknown_probability_mass": 0.0,
        }
        self._log_attempt(
            message=message,
            content=content,
            forecast=None,
            metrics=invalid_metrics,
            error_reason=reason,
        )
        return StepResult(
            reward=-self.format_coef,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=invalid_metrics,
        )

    def _log_attempt(
        self,
        *,
        message: renderers.Message,
        content: str,
        metrics: dict[str, float],
        forecast: dict[str, float] | None,
        error_reason: str | None = None,
    ) -> None:
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=self.messages))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Forecast Reward"):
            summary = {
                "example_id": self.example.example_id,
                "target_winner": self.example.target_winner,
                "target_label": self.example.target_label,
                "format_valid": bool(metrics["format_valid"]),
                "target_prob": f"{metrics['target_prob']:.4f}",
                "brier": f"{metrics['brier']:.4f}",
                "log_loss": f"{metrics['log_loss']:.4f}",
                "reward": f"{metrics['brier_reward']:.4f}",
            }
            if error_reason is not None:
                summary["error_reason"] = error_reason
            logtree.table_from_dict(summary, caption="Forecast summary")
            if forecast is not None:
                logtree.table_from_dict(
                    {label: f"{prob:.4f}" for label, prob in sorted(forecast.items())},
                    caption="Normalized forecast probabilities",
                )
            else:
                logtree.log_text(content[:1000])


@dataclass(frozen=True)
class GolfForecastGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], GolfForecastEnv]
    num_envs: int
    dataset_name: str = "golf_forecasting"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]

