"""OpenCV visualization: boxes, track IDs, counting lines, and live counters."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import cv2
import numpy as np

from .counter import CrossingLine
from .models import TrackedObject

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 255),       # amber
    "motorcycle": (255, 140, 0),  # blue
    "bus": (0, 90, 255),        # red-orange
    "truck": (0, 220, 130),     # green
    "bicycle": (220, 180, 0),
}
DEFAULT_COLOR = (200, 200, 200)
POSITIVE_COLOR = (80, 240, 120)
NEGATIVE_COLOR = (60, 170, 255)
LINE_COLOR = (0, 255, 255)


def class_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


class Annotator:
    """Stateful frame renderer (keeps short motion trails per track)."""

    def __init__(
        self,
        lines: Sequence[CrossingLine],
        camera_id: str,
        *,
        trail_length: int = 18,
        show_trails: bool = True,
        backend_label: str = "YOLO + ByteTrack",
    ) -> None:
        self._lines = list(lines)
        self._camera_id = camera_id
        self._backend_label = backend_label
        self._trail_length = max(0, trail_length)
        self._show_trails = show_trails and trail_length > 0
        self._trails: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._track_class: dict[int, str] = {}

    def draw(
        self,
        frame: np.ndarray,
        objects: Iterable[TrackedObject],
        *,
        frame_index: int,
        timestamp: str,
        counts: dict[str, object],
        processing_fps: float | None = None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = max(0.45, min(0.75, width / 1600.0))

        self.draw_lines(frame)

        objects = list(objects)
        if self._show_trails:
            for obj in objects:
                center = (int(obj.center[0]), int(obj.center[1]))
                trail = self._trails.setdefault(obj.track_id, deque(maxlen=self._trail_length))
                trail.append(center)
                self._last_seen[obj.track_id] = frame_index
                self._track_class[obj.track_id] = obj.class_name
            self._prune_trails(frame_index)
            self._draw_trails(frame)

        for obj in objects:
            self._draw_object(frame, obj, scale)

        self._draw_hud(
            frame,
            frame_index=frame_index,
            timestamp=timestamp,
            counts=counts,
            processing_fps=processing_fps,
            scale=scale,
        )
        return frame

    def draw_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw only the configured virtual lines (used by the line preview tool)."""
        for line in self._lines:
            self._draw_line(frame, line)
        return frame

    # ------------------------------------------------------------------ lines
    def _draw_line(self, frame: np.ndarray, line: CrossingLine) -> None:
        height, width = frame.shape[:2]
        first, second = line.pixel_points(width, height)
        p1 = (int(round(first[0])), int(round(first[1])))
        p2 = (int(round(second[0])), int(round(second[1])))
        cv2.line(frame, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)

        dx, dy = second[0] - first[0], second[1] - first[1]
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        normal = (-dy / length, dx / length)
        offset = max(10.0, min(width, height) * 0.02)

        self._draw_direction_arrow(
            frame, first, second, normal, 0.34, +1, line.positive_direction, POSITIVE_COLOR, offset
        )
        self._draw_direction_arrow(
            frame, first, second, normal, 0.68, -1, line.negative_direction, NEGATIVE_COLOR, offset
        )

        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        label = line.name
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            frame,
            (mid[0] - text_w // 2 - 5, mid[1] - text_h - 8),
            (mid[0] + text_w // 2 + 5, mid[1] + 2),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (mid[0] - text_w // 2, mid[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_direction_arrow(
        frame: np.ndarray,
        first: tuple[float, float],
        second: tuple[float, float],
        normal: tuple[float, float],
        fraction: float,
        sign: int,
        label: str,
        color: tuple[int, int, int],
        offset: float,
    ) -> None:
        base_x = first[0] + (second[0] - first[0]) * fraction
        base_y = first[1] + (second[1] - first[1]) * fraction
        start = (
            int(round(base_x + normal[0] * offset * sign * 0.3)),
            int(round(base_y + normal[1] * offset * sign * 0.3)),
        )
        tip = (
            int(round(base_x + normal[0] * offset * sign * 1.8)),
            int(round(base_y + normal[1] * offset * sign * 1.8)),
        )
        cv2.arrowedLine(frame, start, tip, color, 2, cv2.LINE_AA, tipLength=0.4)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        text_x = int(tip[0] - text_size[0] / 2)
        text_y = int(tip[1] + 16 if normal[1] * sign >= 0 else tip[1] - 10)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------------- tracks
    def _draw_trails(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        for track_id, points in self._trails.items():
            if len(points) < 2:
                continue
            color = class_color(self._track_class.get(track_id, ""))
            cv2.polylines(overlay, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    def _prune_trails(self, frame_index: int) -> None:
        stale = [
            track_id
            for track_id in self._trails
            if frame_index - self._last_seen.get(track_id, 0) > 90
        ]
        for track_id in stale:
            self._trails.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            self._track_class.pop(track_id, None)

    def _draw_object(self, frame: np.ndarray, obj: TrackedObject, scale: float) -> None:
        height, width = frame.shape[:2]
        color = class_color(obj.class_name)
        x1, y1, x2, y2 = obj.as_int_bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        corner = max(6, int(min(x2 - x1, y2 - y1) * 0.18))
        thickness = 3
        for start, end in (
            ((x1, y1), (x1 + corner, y1)),
            ((x1, y1), (x1, y1 + corner)),
            ((x2, y1), (x2 - corner, y1)),
            ((x2, y1), (x2, y1 + corner)),
            ((x1, y2), (x1 + corner, y2)),
            ((x1, y2), (x1, y2 - corner)),
            ((x2, y2), (x2 - corner, y2)),
            ((x2, y2), (x2, y2 - corner)),
        ):
            cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

        label = f"{obj.class_name} #{obj.track_id} {obj.confidence:.2f}"
        font_scale = 0.5 * (scale / 0.65)
        thickness_text = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text
        )
        label_y1 = y1 - text_h - baseline - 6
        if label_y1 < 0:
            label_y1 = y1 + 2
        label_x2 = min(width - 1, x1 + text_w + 10)
        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y1 + text_h + baseline + 4), color, -1)
        text_color = (15, 15, 15) if sum(color) > 360 else (245, 245, 245)
        cv2.putText(
            frame,
            label,
            (x1 + 5, label_y1 + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness_text,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------- HUD
    def _draw_hud(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        timestamp: str,
        counts: dict[str, object],
        processing_fps: float | None,
        scale: float,
    ) -> None:
        height, width = frame.shape[:2]
        by_class = dict(counts.get("by_class") or {})
        by_direction = dict(counts.get("by_direction") or {})
        total = int(counts.get("total") or 0)

        rows = sorted(by_class.items(), key=lambda item: (-item[1], item[0]))[:5]
        panel_w = int(min(310, max(210, width * 0.26)))
        panel_h = int(76 + 22 * max(1, len(rows)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)

        cv2.putText(
            frame,
            "TRAFFIC ANALYTICS",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self._camera_id,
            (24, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            timestamp.replace("T", " "),
            (24, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        y = 102
        for class_name, count in rows:
            cv2.circle(frame, (29, y - 5), 5, class_color(class_name), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{class_name:<11}",
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                str(count),
                (40 + int(panel_w * 0.55), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            y += 22

        # Total counter, top right.
        total_w = int(min(200, width * 0.2))
        total_panel_h = 136 if by_direction else 96
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (width - total_w - 12, 12),
            (width - 12, 12 + total_panel_h),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)
        cv2.putText(
            frame,
            "VEHICLES",
            (width - total_w, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        total_font = 1.7 if width >= 1000 else 1.05
        cv2.putText(
            frame,
            f"{total:04d}",
            (width - total_w, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            total_font,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        if by_direction:
            summary = "  ".join(
                f"{name[:1].upper()}:{count}"
                for name, count in sorted(by_direction.items(), key=lambda item: item[0])
            )
            cv2.putText(
                frame,
                summary,
                (width - total_w, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Bottom status bar.
        bar_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - bar_h), (width, height), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        fps_text = f"{processing_fps:.1f} fps" if processing_fps else "fps n/a"
        status = f"frame {frame_index:06d}   |   {fps_text}   |   {self._backend_label}   |   virtual line counting"
        cv2.putText(
            frame,
            status,
            (14, height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
