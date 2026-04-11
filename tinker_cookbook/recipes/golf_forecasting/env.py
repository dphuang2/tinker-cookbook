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


def _build_pressure_section(
    top_names: list[str],
    *,
    round_num: int,
    max_players: int = 5,
) -> str:
    """Build a compact pressure profile section for the top players.

    Only shows data for R2/R3 snapshots (where lead-hold stats are relevant).
    Only includes players for whom we have at least 2 R3 leads on record.
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
        if r3_leads < 2:
            continue  # Too little data to be meaningful
        rate_str = f"{r3_rate:.0%}" if r3_rate is not None else "?"
        lines.append(
            f"  {name}: R3 lead→win={rate_str} ({r3_leads} leads, {r3_blown} blown)"
        )

    if not lines:
        return ""

    return "Lead-hold pressure profiles (historical, from R3 leading position):\n" + "\n".join(lines)


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
) -> list[renderers.Message]:
    # Limit to top N players by position to keep prompt manageable
    if max_candidates > 0 and len(example.players) > max_candidates:
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

    # Compute field analysis from leaderboard
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

    # Round-specific calibration guidance
    round_num = example.round_number
    if round_num == 1:
        calibration_hint = (
            "After round 1, the leader wins only about 15-20% of the time. "
            "The field is very wide open — give significant probability to 'other'."
        )
    elif round_num == 2:
        calibration_hint = (
            "After round 2, the leader wins roughly 30-35% of the time. "
            "Comebacks are common — spread probability across multiple contenders and 'other'."
        )
    else:
        calibration_hint = (
            "After round 3, the leader wins roughly 50-60% of the time. "
            "The top 3-5 players cover most outcomes, but upsets still happen."
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

    instructions = (
        f"Tournament: {example.tournament_name}\n"
        f"Course: {example.course_name or 'Unknown'}\n"
        f"Round: {round_num}\n"
        f"Event day: {example.event_day or 'Unknown'}\n"
        f"Snapshot time: {example.snapshot_timestamp}\n\n"
        f"Leaderboard snapshot (top {n_shown} of {n_total} players):\n"
        f"{leaderboard_table(example, max_players=max_candidates)}\n\n"
        + (scorecard_section + "\n\n" if scorecard_section else "")
        + (pressure_section + "\n\n" if pressure_section else "")
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

