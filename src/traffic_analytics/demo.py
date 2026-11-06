"""Offline synthetic traffic scene + manifest tracker.

The demo renders a top-down four-lane road with deterministic vehicle motion and
writes a per-frame manifest of ground-truth boxes. ``ManifestDetector`` replays
that manifest through the normal pipeline, so counting, annotation, event
generation, and reporting can be exercised without YOLO, PyTorch, or a video
download. Real analyses always use the YOLO + ByteTrack backend.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, AnalyticsConfig, CountingConfig, LineConfig, OutputConfig, VideoConfig
from .models import TrackedObject
from .pipeline import RunResult, process_video
from .utils import ensure_dir

log = logging.getLogger(__name__)

REFERENCE_WIDTH = 1280.0
ROAD_LEFT_RATIO = 0.156
ROAD_RIGHT_RATIO = 0.844
LINE_Y_RATIO = 0.58
REFERENCE_FPS = 25.0

CLASS_SPECS = {
    "car": {"class_id": 2, "size": (78, 148), "speed": 4.2, "weight": 0.60},
    "motorcycle": {"class_id": 3, "size": (36, 78), "speed": 5.0, "weight": 0.15},
    "truck": {"class_id": 7, "size": (92, 190), "speed": 3.6, "weight": 0.15},
    "bus": {"class_id": 5, "size": (98, 205), "speed": 3.8, "weight": 0.10},
}

PALETTES = {
    "car": [(180, 120, 50), (50, 160, 220), (200, 200, 200), (60, 60, 190), (120, 190, 120)],
    "motorcycle": [(40, 190, 255), (30, 110, 230), (200, 210, 40)],
    "truck": [(150, 110, 60), (60, 120, 170), (90, 90, 95)],
    "bus": [(40, 160, 220), (30, 70, 200), (200, 170, 50)],
}


@dataclass(frozen=True)
class Vehicle:
    track_id: int
    class_id: int
    class_name: str
    lane: int
    x: float
    spawn_y: float
    spawn_frame: int
    width: int
    height: int
    speed: float
    direction: str
    color: tuple[int, int, int]
    confidence: float


@dataclass
class DemoResult:
    run: RunResult
    input_video: str
    manifest_path: str
    preview_path: str | None
    vehicle_count: int


class ManifestDetector:
    """Replays pre-recorded detections; used by the offline demo only."""

    name = "synthetic-manifest"

    def __init__(self, manifest_path: str | Path) -> None:
        self._frames: dict[int, list[TrackedObject]] = {}
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                record = json.loads(line)
                objects = [
                    TrackedObject(
                        track_id=int(item["track_id"]),
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                        confidence=float(item["confidence"]),
                        bbox=tuple(float(value) for value in item["bbox"]),
                    )
                    for item in record.get("objects", [])
                ]
                self._frames[int(record["frame_index"])] = objects

    def update(self, frame: np.ndarray, frame_index: int) -> list[TrackedObject]:
        return self._frames.get(frame_index, [])


def _lane_centers(width: float) -> list[float]:
    left = width * ROAD_LEFT_RATIO
    right = width * ROAD_RIGHT_RATIO
    lane_width = (right - left) / 4.0
    return [left + lane_width * (index + 0.5) for index in range(4)]


def _vehicle_bbox(vehicle: Vehicle, frame_index: int) -> tuple[float, float, float, float]:
    travelled = (frame_index - vehicle.spawn_frame) * vehicle.speed
    center_y = (
        vehicle.spawn_y + travelled
        if vehicle.direction == "southbound"
        else vehicle.spawn_y - travelled
    )
    return (
        vehicle.x - vehicle.width / 2.0,
        center_y - vehicle.height / 2.0,
        vehicle.x + vehicle.width / 2.0,
        center_y + vehicle.height / 2.0,
    )


def _build_vehicles(
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    seed: int,
) -> list[Vehicle]:
    rng = random.Random(seed)
    scale = width / REFERENCE_WIDTH
    lane_centers = _lane_centers(width)
    duration_frames = int(round(duration_s * fps))
    names = list(CLASS_SPECS)
    weights = [CLASS_SPECS[name]["weight"] for name in names]
    vehicles: list[Vehicle] = []
    track_id = 0

    for lane_index, lane_x in enumerate(lane_centers):
        direction = "southbound" if lane_index < 2 else "northbound"
        spawn_time = rng.uniform(-7.0, -2.0)
        while spawn_time < duration_s:
            spawn_frame = int(round(spawn_time * fps))
            if spawn_frame < duration_frames:
                class_name = rng.choices(names, weights=weights, k=1)[0]
                spec = CLASS_SPECS[class_name]
                spec_width, spec_height = spec["size"]
                vehicle_width = max(8, int(round(spec_width * scale)))
                vehicle_height = max(16, int(round(spec_height * scale)))
                base_color = rng.choice(PALETTES[class_name])
                color = tuple(
                    int(max(20, min(245, channel + rng.randint(-18, 18))))
                    for channel in base_color
                )
                speed = spec["speed"] * rng.uniform(0.94, 1.06) * (fps / REFERENCE_FPS) * scale
                spawn_y = (
                    -vehicle_height / 2.0 - 8
                    if direction == "southbound"
                    else height + vehicle_height / 2.0 + 8
                )
                track_id += 1
                vehicles.append(
                    Vehicle(
                        track_id=track_id,
                        class_id=int(spec["class_id"]),
                        class_name=class_name,
                        lane=lane_index,
                        x=lane_x + rng.uniform(-7.0, 7.0) * scale,
                        spawn_y=spawn_y,
                        spawn_frame=spawn_frame,
                        width=vehicle_width,
                        height=vehicle_height,
                        speed=speed,
                        direction=direction,
                        color=color,
                        confidence=round(rng.uniform(0.86, 0.98), 3),
                    )
                )
            spawn_time += rng.uniform(1.9, 3.1)

    return vehicles


def _mix(color: tuple[int, int, int], other: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(
        int(round(channel * (1 - amount) + target * amount))
        for channel, target in zip(color, other)
    )


def _rounded_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for center in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, center, radius, color, -1)
    else:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for center in (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x1 + radius, y2 - radius),
            (x2 - radius, y2 - radius),
        ):
            cv2.circle(image, center, radius, color, thickness)


def _draw_vehicle(image: np.ndarray, vehicle: Vehicle, bbox: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    if width <= 2 or height <= 4:
        return

    radius = max(4, int(min(width, height) * 0.16))
    front_at_bottom = vehicle.direction == "southbound"

    def span(from_front: float, to_front: float) -> tuple[int, int]:
        if front_at_bottom:
            return int(round(y2 - height * from_front)), int(round(y2 - height * to_front))
        return int(round(y1 + height * from_front)), int(round(y1 + height * to_front))

    # Shadow, wheels, body.
    cv2.rectangle(image, (x1 + 4, y1 + 5), (x2 + 4, y2 + 5), (30, 30, 32), -1)
    wheel_w = max(3, int(width * 0.13))
    wheel_h = max(6, int(height * 0.14))
    for wheel_y in (y1 + int(height * 0.16), y2 - int(height * 0.16) - wheel_h):
        cv2.rectangle(image, (x1 - wheel_w // 2, wheel_y), (x1 + wheel_w, wheel_y + wheel_h), (20, 20, 22), -1)
        cv2.rectangle(
            image,
            (x2 - wheel_w, wheel_y),
            (x2 + wheel_w // 2, wheel_y + wheel_h),
            (20, 20, 22),
            -1,
        )
    _rounded_rect(image, (x1, y1), (x2, y2), radius, vehicle.color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (25, 25, 28), 1)

    inset = max(3, int(width * 0.14))
    if vehicle.class_name == "motorcycle":
        rider_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(image, rider_center, max(5, int(width * 0.28)), (35, 35, 40), -1, cv2.LINE_AA)
        cv2.circle(image, rider_center, max(3, int(width * 0.18)), vehicle.color, -1, cv2.LINE_AA)
        seat = span(0.48, 0.7)
        cv2.rectangle(image, (x1 + inset, seat[0]), (x2 - inset, seat[1]), (30, 30, 32), -1)
    elif vehicle.class_name == "bus":
        strip = span(0.18, 0.82)
        cv2.rectangle(
            image,
            (x1 + inset, strip[0]),
            (x2 - inset, strip[1]),
            (105, 125, 140),
            -1,
        )
        for fraction in (0.26, 0.38, 0.50, 0.62, 0.74):
            window = span(fraction, fraction + 0.07)
            cv2.rectangle(
                image,
                (x1 + inset + 2, window[0]),
                (x2 - inset - 2, window[1]),
                (60, 80, 95),
                -1,
            )
    elif vehicle.class_name == "truck":
        cargo = span(0.36, 0.86)
        cv2.rectangle(
            image,
            (x1 + inset, cargo[0]),
            (x2 - inset, cargo[1]),
            _mix(vehicle.color, (235, 235, 235), 0.45),
            -1,
        )
        cv2.rectangle(image, (x1 + inset, cargo[0]), (x2 - inset, cargo[1]), (40, 40, 42), 1)
        windshield = span(0.12, 0.24)
        cv2.rectangle(
            image,
            (x1 + inset + 2, windshield[0]),
            (x2 - inset - 2, windshield[1]),
            (90, 110, 120),
            -1,
        )
    else:
        windshield = span(0.16, 0.30)
        roof = span(0.34, 0.58)
        rear_window = span(0.66, 0.76)
        cv2.rectangle(
            image,
            (x1 + inset + 1, windshield[0]),
            (x2 - inset - 1, windshield[1]),
            (105, 125, 138),
            -1,
        )
        cv2.rectangle(
            image,
            (x1 + inset + 2, roof[0]),
            (x2 - inset - 2, roof[1]),
            _mix(vehicle.color, (245, 245, 245), 0.18),
            -1,
        )
        cv2.rectangle(
            image,
            (x1 + inset + 4, rear_window[0]),
            (x2 - inset - 4, rear_window[1]),
            (80, 100, 112),
            -1,
        )

    # Lights.
    light_radius = max(2, int(width * 0.06))
    light_y = (y1 + int(height * 0.08)) if not front_at_bottom else (y2 - int(height * 0.08))
    light_color = (225, 245, 255)
    for light_x in (x1 + int(width * 0.18), x2 - int(width * 0.18)):
        cv2.circle(image, (light_x, light_y), light_radius, light_color, -1, cv2.LINE_AA)
    tail_y = (y2 - int(height * 0.06)) if not front_at_bottom else (y1 + int(height * 0.06))
    for light_x in (x1 + int(width * 0.18), x2 - int(width * 0.18)):
        cv2.circle(image, (light_x, tail_y), light_radius, (40, 40, 220), -1, cv2.LINE_AA)


def _build_background(width: int, height: int) -> np.ndarray:
    rng = random.Random(20261106)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = (44, 72, 48)

    road_left = int(width * ROAD_LEFT_RATIO)
    road_right = int(width * ROAD_RIGHT_RATIO)

    for _ in range(max(80, int(width * height / 9000))):
        grass_x = rng.randint(0, width - 1)
        grass_y = rng.randint(0, height - 6)
        if road_left - 30 < grass_x < road_right + 30:
            continue
        shade = rng.randint(36, 58)
        cv2.line(
            image,
            (grass_x, grass_y),
            (grass_x, grass_y + rng.randint(2, 6)),
            (shade, shade + 22, shade),
            1,
        )

    sidewalk = max(18, int(width * 0.018))
    cv2.rectangle(image, (road_left - sidewalk, 0), (road_left, height), (112, 112, 118), -1)
    cv2.rectangle(image, (road_right, 0), (road_right + sidewalk, height), (112, 112, 118), -1)
    cv2.line(image, (road_left - sidewalk, 0), (road_left - sidewalk, height), (78, 78, 84), 3)
    cv2.line(image, (road_right + sidewalk, 0), (road_right + sidewalk, height), (78, 78, 84), 3)
    for seam_y in range(0, height, 56):
        cv2.line(image, (road_left - sidewalk, seam_y), (road_left, seam_y), (94, 94, 100), 1)
        cv2.line(image, (road_right, seam_y), (road_right + sidewalk, seam_y), (94, 94, 100), 1)

    cv2.rectangle(image, (road_left, 0), (road_right, height), (58, 58, 62), -1)
    for _ in range(max(120, int(width * height / 3200))):
        dot_x = rng.randint(road_left + 6, road_right - 6)
        dot_y = rng.randint(6, height - 6)
        shade = rng.randint(62, 74)
        cv2.circle(image, (dot_x, dot_y), 1, (shade, shade, shade + 2), -1)

    # Crosswalk near the top edge.
    stripe_width = max(14, int(width * 0.018))
    crosswalk_top = int(height * 0.035)
    crosswalk_bottom = int(height * 0.075)
    for stripe_x in range(road_left + 10, road_right - 10, stripe_width * 2):
        cv2.rectangle(
            image,
            (stripe_x, crosswalk_top),
            (min(stripe_x + stripe_width, road_right - 10), crosswalk_bottom),
            (205, 205, 205),
            -1,
        )

    lane_width = (road_right - road_left) / 4.0
    cv2.line(image, (road_left + 5, 0), (road_left + 5, height), (225, 225, 225), 4)
    cv2.line(image, (road_right - 5, 0), (road_right - 5, height), (225, 225, 225), 4)

    for divider_index in (1, 3):
        divider_x = int(road_left + lane_width * divider_index)
        for dash_y in range(0, height, 82):
            cv2.line(
                image,
                (divider_x, dash_y),
                (divider_x, min(dash_y + 44, height)),
                (225, 225, 225),
                4,
            )

    median_x = int(road_left + lane_width * 2)
    cv2.line(image, (median_x - 5, 0), (median_x - 5, height), (50, 190, 220), 3)
    cv2.line(image, (median_x + 5, 0), (median_x + 5, height), (50, 190, 220), 3)

    for arrow_y in (int(height * 0.20), int(height * 0.84)):
        for lane_index, lane_x in enumerate(_lane_centers(width)):
            direction_down = lane_index < 2
            start = (int(lane_x), arrow_y - (36 if direction_down else -36))
            end = (int(lane_x), arrow_y + (36 if direction_down else -36))
            cv2.arrowedLine(image, start, end, (205, 205, 205), 3, cv2.LINE_AA, tipLength=0.4)

    cv2.putText(
        image,
        "SYNTHETIC",
        (max(8, road_left // 2 - 48), int(height * 0.48)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (165, 175, 165),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "DEMO SCENE",
        (max(4, road_left // 2 - 58), int(height * 0.48) + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (165, 175, 165),
        1,
        cv2.LINE_AA,
    )
    return image


def _render_video(
    video_path: Path,
    manifest_path: Path,
    vehicles: list[Vehicle],
    *,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
) -> int:
    background = _build_background(width, height)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video writer: {video_path}")

    frame_count = int(round(duration_s * fps))
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for frame_index in range(frame_count):
                frame = background.copy()
                objects = []
                for vehicle in vehicles:
                    bbox = _vehicle_bbox(vehicle, frame_index)
                    if bbox[3] < -24 or bbox[1] > height + 24:
                        continue
                    if bbox[2] < 0 or bbox[0] > width:
                        continue
                    _draw_vehicle(frame, vehicle, bbox)
                    if bbox[3] > 0 and bbox[1] < height:
                        objects.append(
                            {
                                "track_id": vehicle.track_id,
                                "class_id": vehicle.class_id,
                                "class_name": vehicle.class_name,
                                "confidence": vehicle.confidence,
                                "bbox": [round(value, 2) for value in bbox],
                            }
                        )
                manifest.write(
                    json.dumps({"frame_index": frame_index, "objects": objects}) + "\n"
                )
                writer.write(frame)
                if frame_index and frame_index % 250 == 0:
                    log.info("Rendered %d/%d demo frames", frame_index, frame_count)
    finally:
        writer.release()
    return frame_count


def _extract_preview(video_path: Path, frame_index: int, output_path: Path) -> str | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = capture.read()
        if not ok:
            return None
        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return str(output_path)
    finally:
        capture.release()


def run_demo(
    output_dir: str | Path = "output/demo",
    *,
    duration: float = 30.0,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
    seed: int = 7,
    frame_stride: int = 1,
    preview_frame: int = 380,
) -> DemoResult:
    """Generate the synthetic scene, run the full pipeline, and extract a preview."""
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    width = max(320, int(width))
    height = max(240, int(height))

    target = ensure_dir(Path(output_dir))
    input_dir = ensure_dir(target / "input")
    input_video = input_dir / "demo_input.mp4"
    manifest_path = input_dir / "demo_input.manifest.jsonl"

    log.info("Generating synthetic scene: %dx%d, %.1fs @ %d fps", width, height, duration, fps)
    vehicles = _build_vehicles(width, height, float(fps), duration, seed)
    frames = _render_video(
        input_video,
        manifest_path,
        vehicles,
        width=width,
        height=height,
        fps=float(fps),
        duration_s=duration,
    )
    log.info("Wrote %s (%d frames, %d vehicles)", input_video, frames, len(vehicles))

    config = AppConfig(
        camera_id="demo_camera_01",
        video=VideoConfig(path=str(input_video), frame_stride=frame_stride),
        lines=[
            LineConfig(
                name="demo_line",
                start=(0.16, LINE_Y_RATIO),
                end=(0.84, LINE_Y_RATIO),
                positive_direction="southbound",
                negative_direction="northbound",
            )
        ],
        counting=CountingConfig(
            base_timestamp="2026-11-06T08:00:00",
            min_confidence=0.3,
            max_track_age_frames=120,
        ),
        analytics=AnalyticsConfig(interval_seconds=10),
        output=OutputConfig(dir=str(target)),
    )
    config.validate()

    annotated_video = target / "demo_annotated.mp4"
    events_path = target / "demo_events.jsonl"
    reports_dir = target / "reports"

    run = process_video(
        config,
        detector=ManifestDetector(manifest_path),
        output_video=annotated_video,
        events_path=events_path,
        reports_dir=reports_dir,
        run_id="demo_run",
    )

    preview_path = None
    if preview_frame >= 0:
        preview = _extract_preview(annotated_video, preview_frame // max(1, frame_stride), target / "preview.jpg")
        preview_path = preview

    return DemoResult(
        run=run,
        input_video=str(input_video),
        manifest_path=str(manifest_path),
        preview_path=preview_path,
        vehicle_count=len(vehicles),
    )
