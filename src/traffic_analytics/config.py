"""Typed configuration loaded from YAML, with validation and clear errors."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import parse_timestamp


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


@dataclass
class ModelConfig:
    weights: str = "yolo11n.pt"
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])


@dataclass
class VideoConfig:
    path: str = "data/raw/traffic.mp4"
    frame_stride: int = 2
    max_frames: int | None = None
    output_fps: float | None = None


@dataclass
class LineConfig:
    name: str = "main_line"
    start: tuple[float, float] = (0.15, 0.55)
    end: tuple[float, float] = (0.85, 0.55)
    positive_direction: str = "southbound"
    negative_direction: str = "northbound"


@dataclass
class CountingConfig:
    base_timestamp: str = ""
    min_confidence: float = 0.3
    count_reentries: bool = False
    max_track_age_frames: int = 90


@dataclass
class AnalyticsConfig:
    interval_seconds: int = 300


@dataclass
class OutputConfig:
    dir: str = "output"
    video: str | None = None
    events: str | None = None
    reports_dir: str | None = None


@dataclass
class AppConfig:
    camera_id: str = "camera_01"
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    lines: list[LineConfig] = field(default_factory=lambda: [LineConfig()])
    counting: CountingConfig = field(default_factory=CountingConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if not str(self.camera_id).strip():
            raise ConfigError("camera_id must not be empty")

        model = self.model
        if not model.weights:
            raise ConfigError("model.weights must not be empty")
        if model.imgsz <= 0:
            raise ConfigError("model.imgsz must be a positive integer")
        if not 0.0 <= model.conf <= 1.0:
            raise ConfigError("model.conf must be between 0 and 1")
        if not 0.0 <= model.iou <= 1.0:
            raise ConfigError("model.iou must be between 0 and 1")
        if not model.classes:
            raise ConfigError("model.classes must contain at least one COCO class id")
        if any(not isinstance(class_id, int) or class_id < 0 for class_id in model.classes):
            raise ConfigError("model.classes must be a list of non-negative integers")

        video = self.video
        if not video.path:
            raise ConfigError("video.path must not be empty")
        if video.frame_stride < 1:
            raise ConfigError("video.frame_stride must be >= 1")
        if video.max_frames is not None and video.max_frames < 1:
            raise ConfigError("video.max_frames must be null or >= 1")
        if video.output_fps is not None and video.output_fps <= 0:
            raise ConfigError("video.output_fps must be null or > 0")

        if not self.lines:
            raise ConfigError("At least one counting line is required")
        seen_names: set[str] = set()
        for index, line in enumerate(self.lines):
            label = f"lines[{index}]"
            if not str(line.name).strip():
                raise ConfigError(f"{label}.name must not be empty")
            if line.name in seen_names:
                raise ConfigError(f"{label}.name '{line.name}' is duplicated")
            seen_names.add(line.name)
            for key, point in (("start", line.start), ("end", line.end)):
                if len(point) != 2:
                    raise ConfigError(f"{label}.{key} must contain exactly two values")
                if not all(0.0 <= float(value) <= 1.0 for value in point):
                    raise ConfigError(f"{label}.{key} coordinates must be normalized (0..1)")
            if abs(line.start[0] - line.end[0]) < 1e-9 and abs(line.start[1] - line.end[1]) < 1e-9:
                raise ConfigError(f"{label} start and end must differ")
            if not line.positive_direction or not line.negative_direction:
                raise ConfigError(f"{label} direction labels must not be empty")
            if line.positive_direction == line.negative_direction:
                raise ConfigError(f"{label} direction labels must be different")

        counting = self.counting
        if not 0.0 <= counting.min_confidence <= 1.0:
            raise ConfigError("counting.min_confidence must be between 0 and 1")
        if counting.max_track_age_frames < 1:
            raise ConfigError("counting.max_track_age_frames must be >= 1")
        if counting.base_timestamp:
            try:
                parse_timestamp(counting.base_timestamp)
            except ValueError as exc:
                raise ConfigError(f"counting.base_timestamp: {exc}") from exc

        if self.analytics.interval_seconds <= 0:
            raise ConfigError("analytics.interval_seconds must be > 0")

        if not self.output.dir:
            raise ConfigError("output.dir must not be empty")


def _section(cls: type, data: Any, label: str) -> Any:
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"'{label}' must be a mapping")
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{label}': {', '.join(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{label}' section: {exc}") from exc


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{label} must be a two-element list, e.g. [0.15, 0.55]")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} values must be numbers") from exc
    return x, y


def _lines(data: Any) -> list[LineConfig]:
    if data is None:
        return [LineConfig()]
    if not isinstance(data, list) or not data:
        raise ConfigError("'lines' must be a non-empty list")

    allowed = {item.name for item in dataclasses.fields(LineConfig)}
    lines: list[LineConfig] = []
    for index, item in enumerate(data):
        label = f"lines[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{label} must be a mapping")
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ConfigError(f"Unknown key(s) in {label}: {', '.join(unknown)}")
        if "start" not in item or "end" not in item:
            raise ConfigError(f"{label} requires both 'start' and 'end'")
        kwargs = {key: value for key, value in item.items() if key not in {"start", "end"}}
        lines.append(
            LineConfig(
                start=_point(item["start"], f"{label}.start"),
                end=_point(item["end"], f"{label}.end"),
                **kwargs,
            )
        )
    return lines


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")

    allowed_top = {"camera_id", "model", "video", "lines", "counting", "analytics", "output"}
    unknown = sorted(set(raw) - allowed_top)
    if unknown:
        raise ConfigError(f"Unknown top-level key(s): {', '.join(unknown)}")

    config = AppConfig(
        camera_id=str(raw.get("camera_id", "camera_01")),
        model=_section(ModelConfig, raw.get("model"), "model"),
        video=_section(VideoConfig, raw.get("video"), "video"),
        lines=_lines(raw.get("lines")),
        counting=_section(CountingConfig, raw.get("counting"), "counting"),
        analytics=_section(AnalyticsConfig, raw.get("analytics"), "analytics"),
        output=_section(OutputConfig, raw.get("output"), "output"),
    )
    config.validate()
    return config


def describe_config(config: AppConfig) -> str:
    """Human-readable one-block summary used by the CLI before a run."""
    lines = [
        f"camera_id       : {config.camera_id}",
        f"model           : {config.model.weights} ({config.model.device}, imgsz={config.model.imgsz}, conf={config.model.conf})",
        f"source video    : {config.video.path}",
        f"frame stride    : {config.video.frame_stride}",
        f"counting lines  : {', '.join(line.name for line in config.lines)}",
        f"base timestamp  : {config.counting.base_timestamp or '(run start time)'}",
    ]
    return "\n".join(lines)


def as_plain_dict(config: AppConfig) -> dict[str, Any]:
    """JSON-friendly dump of the effective configuration (for run metadata)."""
    return dataclasses.asdict(config)
