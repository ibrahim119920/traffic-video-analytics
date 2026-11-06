"""End-to-end video processing pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .analytics import summarize_events, write_reports
from .annotator import Annotator
from .config import AppConfig, as_plain_dict
from .counter import CrossingLine, LineCounter
from .detector import Detector, YoloByteTrackDetector
from .models import CrossingEvent
from .utils import ensure_dir, human_duration, parse_timestamp, timestamp_for_frame

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: str
    camera_id: str
    input_video: str
    output_video: str | None
    events_path: str
    reports_dir: str | None
    total_events: int
    frames_read: int
    frames_processed: int
    source_fps: float
    output_fps: float
    duration_s: float
    processing_s: float
    summary: dict[str, Any]
    report_paths: dict[str, str] = field(default_factory=dict)
    events: list[CrossingEvent] = field(default_factory=list, repr=False)

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.processing_s if self.processing_s > 0 else 0.0


def _safe_run_id(value: str) -> str:
    allowed = {character for character in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"}
    return "".join(character if character in allowed else "_" for character in value).strip("_") or "run"


def process_video(
    config: AppConfig,
    *,
    detector: Detector | None = None,
    video_path: str | Path | None = None,
    output_video: str | Path | None = None,
    events_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
    write_video: bool = True,
    generate_reports: bool = True,
    run_id: str | None = None,
) -> RunResult:
    """Run detection, tracking, counting, annotation, and reporting on a video."""
    config.validate()

    source_path = Path(video_path or config.video.path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Video not found: {source_path}. Put a traffic clip in data/raw/ or pass --video."
        )

    run_id = _safe_run_id(run_id or f"{config.camera_id}_{datetime.now():%Y%m%dT%H%M%S}")
    output_root = Path(config.output.dir)

    video_setting = output_video or config.output.video
    events_setting = events_path or config.output.events
    reports_setting = reports_dir or config.output.reports_dir

    output_video_path = (
        Path(video_setting) if video_setting else output_root / "videos" / f"{run_id}_annotated.mp4"
    )
    events_out_path = (
        Path(events_setting) if events_setting else output_root / "events" / f"{run_id}.jsonl"
    )
    reports_out_path = (
        Path(reports_setting) if reports_setting else output_root / "reports" / run_id
    )

    if detector is None:
        log.info("Loading detector: %s", config.model.weights)
        detector = YoloByteTrackDetector(
            weights=config.model.weights,
            device=config.model.device,
            imgsz=config.model.imgsz,
            conf=config.model.conf,
            iou=config.model.iou,
            classes=config.model.classes,
            tracker=config.model.tracker,
        )

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not source_fps or source_fps != source_fps or source_fps <= 0:  # NaN-safe
        source_fps = 25.0
        log.warning("Video reports an invalid FPS; assuming %.1f", source_fps)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame dimensions from {source_path}")

    stride = max(1, int(config.video.frame_stride))
    output_fps = float(config.video.output_fps) if config.video.output_fps else source_fps / stride
    output_fps = max(output_fps, 1.0)
    max_frames = config.video.max_frames

    base_timestamp = (
        parse_timestamp(config.counting.base_timestamp)
        if config.counting.base_timestamp
        else datetime.now().replace(microsecond=0)
    )

    lines = [
        CrossingLine(
            name=line.name,
            start=tuple(line.start),
            end=tuple(line.end),
            positive_direction=line.positive_direction,
            negative_direction=line.negative_direction,
        )
        for line in config.lines
    ]
    # Counting works in pixel space (track centres are pixels); the annotator
    # expects normalized points so it can render at any frame size.
    pixel_lines = [
        CrossingLine(
            name=line.name,
            start=(line.start[0] * width, line.start[1] * height),
            end=(line.end[0] * width, line.end[1] * height),
            positive_direction=line.positive_direction,
            negative_direction=line.negative_direction,
        )
        for line in lines
    ]
    counter = LineCounter(
        pixel_lines,
        camera_id=config.camera_id,
        min_confidence=config.counting.min_confidence,
        count_reentries=config.counting.count_reentries,
        max_track_age_frames=config.counting.max_track_age_frames,
    )
    detector_name = getattr(detector, "name", detector.__class__.__name__)
    backend_label = (
        "YOLO + ByteTrack"
        if "yolo" in detector_name.lower()
        else detector_name.replace("_", " ").replace("-", " ")
    )
    annotator = Annotator(lines, config.camera_id, backend_label=backend_label)

    ensure_dir(events_out_path.parent)
    writer: cv2.VideoWriter | None = None
    if write_video:
        ensure_dir(output_video_path.parent)
        suffix = output_video_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "XVID"))
        writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                f"OpenCV could not create a video writer for {output_video_path}. "
                "Try an .avi output path or reinstall opencv-python with FFmpeg support."
            )

    log.info("Source: %s (%dx%d @ %.2f fps, %s frames)", source_path, width, height, source_fps, total_frames or "?")
    log.info(
        "Processing: stride=%d, output_fps=%.2f, detector=%s, lines=%s",
        stride,
        output_fps,
        getattr(detector, "name", detector.__class__.__name__),
        ", ".join(line.name for line in lines),    )

    frames_read = 0
    frames_processed = 0
    event_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    processed_limit = min(total_frames, max_frames) if total_frames > 0 and max_frames else (max_frames or total_frames)

    try:
        with events_out_path.open("w", encoding="utf-8") as event_handle:
            while True:
                if max_frames is not None and frames_read >= max_frames:
                    break
                if not capture.grab():
                    break

                frame_index = frames_read
                frames_read += 1
                if frame_index % stride != 0:
                    continue

                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    log.warning("Failed to decode frame %d; stopping", frame_index)
                    break

                objects = detector.update(frame, frame_index)
                timestamp = timestamp_for_frame(base_timestamp, frame_index, source_fps)
                events = counter.update(
                    objects,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    video_time_s=frame_index / source_fps,
                )
                if events:
                    for event in events:
                        record = event.to_dict()
                        event_records.append(record)
                        event_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    event_handle.flush()

                frames_processed += 1
                if writer is not None:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    throughput = frames_processed / elapsed
                    annotated = annotator.draw(
                        frame,
                        objects,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        counts=counter.snapshot(),
                        processing_fps=throughput,
                    )
                    writer.write(annotated)

                if frames_processed % 100 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    throughput = frames_processed / elapsed
                    if processed_limit:
                        progress = f"{min(frames_processed, processed_limit) / processed_limit * 100:5.1f}%"
                    else:
                        progress = "  n/a"
                    log.info(
                        "Processed %d frames (%s) | %.1f fps | %d events",
                        frames_processed,
                        progress,
                        throughput,
                        len(event_records),
                    )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    processing_s = time.perf_counter() - started
    duration_s = frames_read / source_fps if source_fps > 0 else 0.0

    meta = {
        "run_id": run_id,
        "source_video": str(source_path),
        "detector": getattr(detector, "name", detector.__class__.__name__),
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "frame_stride": stride,
        "source_fps": round(source_fps, 3),
        "output_fps": round(output_fps, 3),
        "duration_s": round(duration_s, 3),
        "processing_seconds": round(processing_s, 3),
        "processing_fps": round(frames_processed / processing_s, 2) if processing_s > 0 else 0.0,
        "base_timestamp": base_timestamp.isoformat(timespec="seconds"),
    }
    summary = summarize_events(
        event_records,
        interval_seconds=config.analytics.interval_seconds,
        camera_id=config.camera_id,
        meta=meta,
    )

    report_paths: dict[str, str] = {}
    actual_reports_dir: Path | None = None
    if generate_reports:
        actual_reports_dir = ensure_dir(reports_out_path)
        report_paths = write_reports(
            event_records,
            summary,
            actual_reports_dir,
            interval_seconds=config.analytics.interval_seconds,
        )
        config_snapshot = actual_reports_dir / "run_config.json"
        config_snapshot.write_text(json.dumps(as_plain_dict(config), indent=2), encoding="utf-8")
        report_paths["run_config_json"] = str(config_snapshot)

    log.info(
        "Done: %d events, %d frames in %s (%s)",
        len(event_records),
        frames_processed,
        human_duration(processing_s),
        f"{frames_processed / processing_s:.1f} fps" if processing_s > 0 else "n/a",
    )

    return RunResult(
        run_id=run_id,
        camera_id=config.camera_id,
        input_video=str(source_path),
        output_video=str(output_video_path) if writer is not None else None,
        events_path=str(events_out_path),
        reports_dir=str(actual_reports_dir) if actual_reports_dir else None,
        total_events=len(event_records),
        frames_read=frames_read,
        frames_processed=frames_processed,
        source_fps=source_fps,
        output_fps=output_fps,
        duration_s=duration_s,
        processing_s=processing_s,
        summary=summary,
        report_paths=report_paths,
        events=[CrossingEvent(**record) for record in event_records],
    )
