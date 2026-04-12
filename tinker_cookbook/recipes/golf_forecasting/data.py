from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import chz

from tinker_cookbook import renderers
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def normalize_player_name(name: str) -> str:
    collapsed = " ".join(name.split()).strip().lower()
    collapsed = re.sub(r"[^\w\s'-]+", "", collapsed)
    return collapsed


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True)
class HoleScore:
    """Per-hole score for a single round."""

    hole: int
    score: int  # raw strokes
    to_par: int  # relative to par (negative = under par)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HoleScore":
        return HoleScore(
            hole=int(data["hole"]),
            score=int(data["score"]),
            to_par=int(data["to_par"]),
        )


@dataclass(frozen=True)
class PlayerSnapshot:
    name: str
    position: str
    score_to_par: float
    strokes_behind: float
    holes_completed: int | None = None
    current_hole: int | None = None
    holes_remaining: int | None = None
    round_score: float | None = None
    tee_time: str | None = None
    prior_win_prob: float | None = None
    recent_form_score: float | None = None
    # Hole-by-hole scoring for the current (latest) round
    holes: tuple[HoleScore, ...] = ()
    # Compact scorecard representation (e.g., "-1 E +1 -1 E E E -1 E")
    scorecard_compact: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PlayerSnapshot":
        holes_raw = data.get("holes", [])
        holes = tuple(HoleScore.from_dict(h) for h in holes_raw) if holes_raw else ()
        return PlayerSnapshot(
            name=str(data["name"]),
            position=str(data.get("position", "")),
            score_to_par=float(data.get("score_to_par", 0.0)),
            strokes_behind=float(data.get("strokes_behind", 0.0)),
            holes_completed=_maybe_int(data.get("holes_completed")),
            current_hole=_maybe_int(data.get("current_hole")),
            holes_remaining=_maybe_int(data.get("holes_remaining")),
            round_score=_maybe_float(data.get("round_score")),
            tee_time=data.get("tee_time"),
            prior_win_prob=_maybe_float(data.get("prior_win_prob")),
            recent_form_score=_maybe_float(data.get("recent_form_score")),
            holes=holes,
            scorecard_compact=data.get("scorecard_compact"),
            metadata=cast(dict[str, Any], data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CourseHole:
    """Per-hole course info (scorecard)."""

    hole: int
    par: int
    yards: int | None = None
    handicap_rank: int | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CourseHole":
        return CourseHole(
            hole=int(data["hole"]),
            par=int(data["par"]),
            yards=_maybe_int(data.get("yards")),
            handicap_rank=_maybe_int(data.get("handicap_rank")),
        )


@dataclass(frozen=True)
class GolfForecastExample:
    example_id: str
    tournament_id: str
    tournament_name: str
    course_name: str | None
    round_number: int
    event_day: str | None
    snapshot_timestamp: str
    players: tuple[PlayerSnapshot, ...]
    target_winner: str
    system_context: dict[str, Any] = field(default_factory=dict)
    source_urls: tuple[str, ...] = ()
    other_field_prior: float | None = None
    # Course scorecard: par/yards per hole
    course_scorecard: tuple[CourseHole, ...] = ()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GolfForecastExample":
        players = tuple(PlayerSnapshot.from_dict(player) for player in data["players"])
        scorecard_raw = data.get("course_scorecard", [])
        course_scorecard = tuple(CourseHole.from_dict(h) for h in scorecard_raw) if scorecard_raw else ()
        return GolfForecastExample(
            example_id=str(data["example_id"]),
            tournament_id=str(data.get("tournament_id") or _slugify(str(data["tournament_name"]))),
            tournament_name=str(data["tournament_name"]),
            course_name=data.get("course_name"),
            round_number=int(data.get("round_number", 1)),
            event_day=data.get("event_day"),
            snapshot_timestamp=str(data["snapshot_timestamp"]),
            players=players,
            target_winner=str(data["target_winner"]),
            system_context=cast(dict[str, Any], data.get("system_context", {})),
            source_urls=tuple(str(url) for url in data.get("source_urls", [])),
            other_field_prior=_maybe_float(data.get("other_field_prior")),
            course_scorecard=course_scorecard,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @property
    def candidate_names(self) -> list[str]:
        seen: set[str] = set()
        ordered_names: list[str] = []
        for player in self.players:
            normalized = normalize_player_name(player.name)
            if normalized in seen:
                continue
            ordered_names.append(player.name)
            seen.add(normalized)
        return ordered_names

    @property
    def target_label(self) -> str:
        target = normalize_player_name(self.target_winner)
        for candidate in self.candidate_names:
            if normalize_player_name(candidate) == target:
                return candidate
        return "other"


@dataclass(frozen=True)
class SourceArtifact:
    name: str
    url: str
    local_path: str
    content_type: str
    fetched_at: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SourceArtifact":
        return SourceArtifact(
            name=str(data["name"]),
            url=str(data["url"]),
            local_path=str(data["local_path"]),
            content_type=str(data.get("content_type", "application/octet-stream")),
            fetched_at=str(data["fetched_at"]),
        )


@dataclass(frozen=True)
class DatasetManifest:
    dataset_version: str
    created_at: str
    output_dir: str
    train_path: str
    val_path: str
    heldout_path: str
    heldout_locked: bool
    split_tournament_ids: dict[str, list[str]]
    source_artifacts: tuple[SourceArtifact, ...] = ()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DatasetManifest":
        return DatasetManifest(
            dataset_version=str(data["dataset_version"]),
            created_at=str(data["created_at"]),
            output_dir=str(data["output_dir"]),
            train_path=str(data["train_path"]),
            val_path=str(data["val_path"]),
            heldout_path=str(data["heldout_path"]),
            heldout_locked=bool(data.get("heldout_locked", True)),
            split_tournament_ids=cast(dict[str, list[str]], data.get("split_tournament_ids", {})),
            source_artifacts=tuple(
                SourceArtifact.from_dict(item) for item in data.get("source_artifacts", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_artifacts"] = [asdict(item) for item in self.source_artifacts]
        return data


def load_examples(path: str) -> list[GolfForecastExample]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Golf forecasting dataset not found: {jsonl_path}")
    examples: list[GolfForecastExample] = []
    with jsonl_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(GolfForecastExample.from_dict(json.loads(stripped)))
    return examples


def load_dataset_manifest(path: str) -> DatasetManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Golf forecasting manifest not found: {manifest_path}")
    return DatasetManifest.from_dict(json.loads(manifest_path.read_text()))


def format_compact_scorecard(holes: tuple[HoleScore, ...]) -> str:
    """Format hole scores as a compact string, e.g., 'E -1 -1 +1 E E -1 E -2'.

    Negative = under par (birdie/eagle), positive = over par (bogey).
    E = even par.
    """
    parts = []
    for h in sorted(holes, key=lambda x: x.hole):
        if h.to_par == 0:
            parts.append("E")
        elif h.to_par > 0:
            parts.append(f"+{h.to_par}")
        else:
            parts.append(str(h.to_par))
    return " ".join(parts)


def scorecard_momentum(holes: tuple[HoleScore, ...], *, last_n: int = 5) -> str:
    """Summarize scoring momentum over the last N holes played."""
    if not holes:
        return ""
    sorted_holes = sorted(holes, key=lambda x: x.hole)
    recent = sorted_holes[-last_n:]
    total = sum(h.to_par for h in recent)
    if total == 0:
        trend = "E"
    elif total > 0:
        trend = f"+{total}"
    else:
        trend = str(total)
    return f"Last {len(recent)} holes: {trend}"


def leaderboard_table(
    example: GolfForecastExample,
    *,
    max_players: int | None = None,
    players_override: list["PlayerSnapshot"] | None = None,
    is_stableford: bool = False,
) -> str:
    """Render the leaderboard as a markdown table.

    For R2+ snapshots shows This Rnd + Prev Rnds columns so the model can see
    per-round scoring trajectory, not just the cumulative total.

    players_override: if provided, use this ordered list instead of example.players
    (useful for re-sorting Stableford tournaments where higher score = better).
    is_stableford: if True, recalculates 'behind' relative to the new leader (max score)
    and shows 'Pts' label instead of 'Total'.
    """
    source = players_override if players_override is not None else example.players
    players = source[:max_players] if max_players is not None else source
    is_r1 = example.round_number == 1

    # For Stableford, compute behind relative to the new sorted leader (highest score)
    stableford_leader_score = players[0].score_to_par if (is_stableford and players) else None

    total_label = "Pts" if is_stableford else "Total"

    if is_r1:
        header = (
            f"| Player | Pos | {total_label} | This Rnd | Behind | Hole | Done | Remaining | Prior | Recent |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
    else:
        header = (
            f"| Player | Pos | {total_label} | This Rnd | Prev Rnds | Behind | Hole | Done | Remaining | Prior | Recent |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )

    rows = []
    for rank, player in enumerate(players, start=1):
        rnd_str = f"{player.round_score:+.0f}" if player.round_score is not None else "-"
        if not is_r1 and player.round_score is not None:
            prev_str = f"{player.score_to_par - player.round_score:+.0f}"
        else:
            prev_str = "-"

        if is_stableford and stableford_leader_score is not None:
            # Use Stableford rank (1, 2, 3...) and points-behind-leader for Behind
            pos_display = str(rank)
            behind_val = stableford_leader_score - player.score_to_par
        else:
            pos_display = player.position or "-"
            behind_val = player.strokes_behind

        # For Stableford, show raw points without the +/- prefix (positive = good)
        total_fmt = f"{player.score_to_par:.0f}" if is_stableford else f"{player.score_to_par:+.0f}"

        if is_r1:
            rows.append(
                "| {name} | {pos} | {total} | {rnd} | {behind:.1f} | {hole} | "
                "{done} | {rem} | {prior} | {recent} |".format(
                    name=player.name,
                    pos=pos_display,
                    total=total_fmt,
                    rnd=rnd_str,
                    behind=behind_val,
                    hole=player.current_hole if player.current_hole is not None else "-",
                    done=player.holes_completed if player.holes_completed is not None else "-",
                    rem=player.holes_remaining if player.holes_remaining is not None else "-",
                    prior=f"{player.prior_win_prob:.3f}" if player.prior_win_prob is not None else "-",
                    recent=f"{player.recent_form_score:.2f}" if player.recent_form_score is not None else "-",
                )
            )
        else:
            rows.append(
                "| {name} | {pos} | {total} | {rnd} | {prev} | {behind:.1f} | {hole} | "
                "{done} | {rem} | {prior} | {recent} |".format(
                    name=player.name,
                    pos=pos_display,
                    total=total_fmt,
                    rnd=rnd_str,
                    prev=prev_str,
                    behind=behind_val,
                    hole=player.current_hole if player.current_hole is not None else "-",
                    done=player.holes_completed if player.holes_completed is not None else "-",
                    rem=player.holes_remaining if player.holes_remaining is not None else "-",
                    prior=f"{player.prior_win_prob:.3f}" if player.prior_win_prob is not None else "-",
                    recent=f"{player.recent_form_score:.2f}" if player.recent_form_score is not None else "-",
                )
            )
    return "\n".join([header, *rows])


def candidate_labels(example: GolfForecastExample) -> list[str]:
    return [*example.candidate_names, "other"]


class GolfForecastDataset(RLDataset):
    def __init__(
        self,
        examples: Sequence[GolfForecastExample],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        split: Literal["train", "val"] = "train",
        include_other_bucket: bool = True,
        include_player_history: bool = False,
        include_tournament_history: bool = False,
    ):
        self.examples = list(examples)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.include_other_bucket = include_other_bucket
        self.include_player_history = include_player_history
        self.include_tournament_history = include_tournament_history

    def __len__(self) -> int:
        return math.ceil(len(self.examples) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        from tinker_cookbook.recipes.golf_forecasting.env import GolfForecastGroupBuilder, GolfForecastEnv

        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.examples))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            GolfForecastGroupBuilder(
                env_thunk=partial(
                    GolfForecastEnv,
                    example=example,
                    renderer=self.renderer,
                    include_other_bucket=self.include_other_bucket,
                    include_player_history=self.include_player_history,
                    include_tournament_history=self.include_tournament_history,
                ),
                num_envs=self.group_size,
                dataset_name="golf_forecasting",
            )
            for example in self.examples[batch_start:batch_end]
        ]


@chz.chz
class GolfForecastDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    dataset_manifest_path: str | None = None
    train_jsonl_path: str | None = None
    val_jsonl_path: str | None = None
    include_other_bucket: bool = True
    include_player_history: bool = False
    include_tournament_history: bool = False

    async def __call__(self) -> tuple[GolfForecastDataset, GolfForecastDataset | None]:
        if self.dataset_manifest_path is not None:
            manifest = load_dataset_manifest(self.dataset_manifest_path)
            train_path = manifest.train_path
            val_path = manifest.val_path
        else:
            if self.train_jsonl_path is None:
                raise ValueError("Either dataset_manifest_path or train_jsonl_path must be provided.")
            train_path = self.train_jsonl_path
            val_path = self.val_jsonl_path

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = GolfForecastDataset(
            examples=load_examples(train_path),
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            split="train",
            include_other_bucket=self.include_other_bucket,
            include_player_history=self.include_player_history,
            include_tournament_history=self.include_tournament_history,
        )
        val_dataset = (
            GolfForecastDataset(
                examples=load_examples(val_path),
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                split="val",
                include_other_bucket=self.include_other_bucket,
                include_player_history=self.include_player_history,
                include_tournament_history=self.include_tournament_history,
            )
            if val_path is not None
            else None
        )
        return train_dataset, val_dataset

