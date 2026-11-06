"""Virtual counting lines and direction-aware crossing logic.

A line is defined by two normalized points. In image coordinates (y grows
downwards) a line drawn left-to-right has its *positive* side below it, so the
default labels are southbound (positive) and northbound (negative).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence

from .models import CrossingEvent, TrackedObject

Point = tuple[float, float]
_EPS = 1e-9


@dataclass(frozen=True)
class CrossingLine:
    """A directed virtual line used to count vehicles.

    ``start`` and ``end`` are normalized ``(x, y)`` points in the frame.
    """

    name: str
    start: Point
    end: Point
    positive_direction: str = "southbound"
    negative_direction: str = "northbound"

    def __post_init__(self) -> None:
        if len(self.start) != 2 or len(self.end) != 2:
            raise ValueError(f"Line '{self.name}' requires 2D start and end points")
        if hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]) < 1e-6:
            raise ValueError(f"Line '{self.name}' has zero length")
        if self.positive_direction == self.negative_direction:
            raise ValueError(f"Line '{self.name}' direction labels must differ")

    @property
    def vector(self) -> Point:
        return (self.end[0] - self.start[0], self.end[1] - self.start[1])

    def pixel_points(self, width: int, height: int) -> tuple[Point, Point]:
        """Convert normalized points to pixel coordinates."""
        return (
            (self.start[0] * width, self.start[1] * height),
            (self.end[0] * width, self.end[1] * height),
        )

    def side(self, point: Point) -> float:
        """Signed side value: sign tells which side of the line the point is on."""
        (sx, sy), (ex, ey) = self.start, self.end
        return (ex - sx) * (point[1] - sy) - (ey - sy) * (point[0] - sx)

    def crosses(self, previous: Point, current: Point) -> bool:
        """True when segment ``previous -> current`` intersects the line segment."""
        rx, ry = current[0] - previous[0], current[1] - previous[1]
        sx, sy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        denominator = rx * sy - ry * sx
        if abs(denominator) < 1e-12:
            return False  # parallel or degenerate movement

        qx = self.start[0] - previous[0]
        qy = self.start[1] - previous[1]
        t = (qx * sy - qy * sx) / denominator
        u = (qx * ry - qy * rx) / denominator
        return -_EPS <= t <= 1.0 + _EPS and -_EPS <= u <= 1.0 + _EPS


class LineCounter:
    """Tracks vehicle centres across frames and emits one event per crossing."""

    def __init__(
        self,
        lines: Sequence[CrossingLine],
        *,
        camera_id: str = "",
        min_confidence: float = 0.3,
        count_reentries: bool = False,
        max_track_age_frames: int = 90,
    ) -> None:
        if not lines:
            raise ValueError("At least one counting line is required")
        self._camera_id = camera_id
        self._lines = list(lines)
        self._min_confidence = min_confidence
        self._count_reentries = count_reentries
        self._max_age = max(1, int(max_track_age_frames))
        self._previous: dict[int, tuple[Point, int]] = {}
        self._counted: set[tuple[str, int]] = set()
        self._last_count_frame: dict[tuple[str, int], int] = {}
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._event_count = 0

    @property
    def lines(self) -> list[CrossingLine]:
        return list(self._lines)

    @property
    def event_count(self) -> int:
        return self._event_count

    def reset(self) -> None:
        self._previous.clear()
        self._counted.clear()
        self._last_count_frame.clear()
        self._counts.clear()
        self._event_count = 0

    def update(
        self,
        objects: Iterable[TrackedObject],
        *,
        frame_index: int,
        timestamp: str,
        video_time_s: float | None = None,
    ) -> list[CrossingEvent]:
        """Update track history and return events produced by this frame."""
        events: list[CrossingEvent] = []
        time_s = float(video_time_s) if video_time_s is not None else 0.0

        for obj in objects:
            if obj.confidence < self._min_confidence:
                continue
            center = obj.center
            previous = self._previous.get(obj.track_id)
            self._previous[obj.track_id] = (center, frame_index)

            if previous is None:
                continue
            previous_center, previous_frame = previous
            if frame_index - previous_frame > self._max_age:
                continue
            if previous_center == center:
                continue

            for line in self._lines:
                if not line.crosses(previous_center, center):
                    continue
                side_before = line.side(previous_center)
                side_after = line.side(center)
                if side_before < -_EPS and side_after >= -_EPS:
                    direction = line.positive_direction
                elif side_before > _EPS and side_after <= _EPS:
                    direction = line.negative_direction
                else:
                    continue

                key = (line.name, obj.track_id)
                if not self._count_reentries:
                    if key in self._counted:
                        continue
                else:
                    last_frame = self._last_count_frame.get(key)
                    if last_frame is not None and frame_index - last_frame < 5:
                        continue

                self._counted.add(key)
                self._last_count_frame[key] = frame_index
                self._counts[(line.name, obj.class_name, direction)] += 1
                self._event_count += 1
                events.append(
                    CrossingEvent(
                        camera_id=self._camera_id,
                        timestamp=timestamp,
                        track_id=int(obj.track_id),
                        object_class=obj.class_name,
                        confidence=round(float(obj.confidence), 4),
                        event_type="line_crossing",
                        direction=direction,
                        line_name=line.name,
                        class_id=int(obj.class_id),
                        frame_index=int(frame_index),
                        video_time_s=round(time_s, 3),
                    )
                )

        if frame_index % 120 == 0:
            self._prune(frame_index)
        return events

    def _prune(self, frame_index: int) -> None:
        stale = [
            track_id
            for track_id, (_, seen_at) in self._previous.items()
            if frame_index - seen_at > self._max_age
        ]
        for track_id in stale:
            del self._previous[track_id]

    def snapshot(self) -> dict[str, object]:
        """Aggregate counts for HUD rendering and console output."""
        by_class: Counter[str] = Counter()
        by_direction: Counter[str] = Counter()
        by_line: Counter[str] = Counter()
        by_line_direction: Counter[str] = Counter()
        for (line_name, class_name, direction), count in self._counts.items():
            by_class[class_name] += count
            by_direction[direction] += count
            by_line[line_name] += count
            by_line_direction[f"{line_name}:{direction}"] += count
        return {
            "total": int(sum(self._counts.values())),
            "by_class": dict(by_class),
            "by_direction": dict(by_direction),
            "by_line": dict(by_line),
            "by_line_direction": dict(by_line_direction),
        }
