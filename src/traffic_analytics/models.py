"""Domain models shared by detection, counting, annotation, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class TrackedObject:
    """A single vehicle observation for one frame.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    def as_int_bbox(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


@dataclass
class CrossingEvent:
    """Structured event emitted when a tracked vehicle crosses a counting line.

    The first seven fields are the stable contract consumed by downstream
    analytics/data-pipeline projects. The remaining fields add traceability.
    """

    camera_id: str
    timestamp: str
    track_id: int
    object_class: str
    confidence: float
    event_type: str
    direction: str
    line_name: str = ""
    class_id: int = 0
    frame_index: int = 0
    video_time_s: float = 0.0
    schema_version: str = "1.0"
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
