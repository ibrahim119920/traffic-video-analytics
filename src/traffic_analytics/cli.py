"""Command line interface: analyze, summarize, demo, preview-lines, doctor."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .config import ConfigError, describe_config, load_config
from .utils import ensure_dir, format_counts, human_duration, setup_logging

if TYPE_CHECKING:
    from .pipeline import RunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-analytics",
        description="Detect, track, count, and report vehicles in traffic video.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="process a real traffic video with YOLO + ByteTrack")
    analyze.add_argument("--config", default="config/config.yaml", help="path to the YAML config")
    analyze.add_argument("--video", help="override video.path")
    analyze.add_argument("--output-video", help="override the annotated video path")
    analyze.add_argument("--events", help="override the JSONL events path")
    analyze.add_argument("--reports-dir", help="override the reports directory")
    analyze.add_argument("--run-id", help="stable run identifier used in output filenames")
    analyze.add_argument("--camera-id", help="override camera_id")
    analyze.add_argument("--frame-stride", type=int, help="process every Nth frame")
    analyze.add_argument("--max-frames", type=int, help="stop after N source frames")
    analyze.add_argument("--model", help="override model.weights (e.g. yolov8n.pt)")
    analyze.add_argument("--device", help="cpu or GPU index, e.g. 0")
    analyze.add_argument("--imgsz", type=int, help="inference size")
    analyze.add_argument("--conf", type=float, help="detection confidence threshold")
    analyze.add_argument("--no-video", action="store_true", help="skip annotated video rendering")
    analyze.add_argument("--no-reports", action="store_true", help="skip CSV/JSON/Markdown reports")

    summarize = subparsers.add_parser("summarize", help="build reports from an existing events JSONL")
    summarize.add_argument("events", help="path to a JSONL or CSV events file")
    summarize.add_argument("--out", default="output/reports/summary", help="output directory")
    summarize.add_argument("--interval", type=int, default=300, help="time bucket in seconds")
    summarize.add_argument("--camera-id", help="override the camera id from the events")

    demo = subparsers.add_parser(
        "demo", help="generate a synthetic scene and run the whole pipeline offline"
    )
    demo.add_argument("--output-dir", default="output/demo")
    demo.add_argument("--duration", type=float, default=30.0, help="seconds of synthetic video")
    demo.add_argument("--fps", type=int, default=25)
    demo.add_argument("--width", type=int, default=1280)
    demo.add_argument("--height", type=int, default=720)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--frame-stride", type=int, default=1)
    demo.add_argument("--preview-frame", type=int, default=380)

    preview = subparsers.add_parser(
        "preview-lines", help="draw the configured counting lines on one video frame"
    )
    preview.add_argument("--config", default="config/config.yaml")
    preview.add_argument("--video", help="override video.path")
    preview.add_argument("--frame", type=int, default=0, help="source frame index to render")
    preview.add_argument("--out", default="output/line_preview.jpg")

    subparsers.add_parser("doctor", help="check the local environment and dependencies")
    return parser


def _apply_overrides(config, args: argparse.Namespace):
    if args.video:
        config.video.path = args.video
    if args.frame_stride is not None:
        config.video.frame_stride = args.frame_stride
    if args.max_frames is not None:
        config.video.max_frames = args.max_frames
    if args.camera_id:
        config.camera_id = args.camera_id
    if args.model:
        config.model.weights = args.model
    if args.device:
        config.model.device = args.device
    if args.imgsz is not None:
        config.model.imgsz = args.imgsz
    if args.conf is not None:
        config.model.conf = args.conf
    config.validate()
    return config


def _print_run_summary(result: "RunResult") -> None:
    summary = result.summary
    print()
    print("Traffic analytics summary")
    print("-------------------------")
    print(f"Run ID       : {result.run_id}")
    print(f"Camera       : {result.camera_id}")
    print(f"Events       : {result.total_events}")
    print(f"By class     : {format_counts(summary.get('by_class') or {})}")
    print(f"By direction : {format_counts(summary.get('by_direction') or {})}")
    peak = summary.get("peak_interval")
    if peak:
        print(f"Peak interval: {peak['interval_start']} ({peak['count']} vehicles)")
    print(
        "Processed    : "
        f"{result.frames_processed} frames in {human_duration(result.processing_s)} "
        f"({result.processing_fps:.1f} fps)"
    )
    if result.output_video:
        print(f"Video        : {result.output_video}")
    else:
        print("Video        : disabled")
    print(f"Events file  : {result.events_path}")
    if result.reports_dir:
        print(f"Reports      : {result.reports_dir}")


def _preview_lines(args: argparse.Namespace) -> int:
    import cv2

    from .annotator import Annotator
    from .counter import CrossingLine

    config = load_config(args.config)
    if args.video:
        config.video.path = args.video
    config.validate()

    source = Path(config.video.path)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = max(0, args.frame)
        if frame_count > 0:
            frame_index = min(frame_index, frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {source}")
    finally:
        capture.release()

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
    annotator = Annotator(lines, config.camera_id)
    annotator.draw_lines(frame)

    target = Path(args.out)
    ensure_dir(target.parent)
    cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Wrote {target} (frame {frame_index})")
    for line in config.lines:
        print(
            f"  {line.name}: start=({line.start[0]:.3f}, {line.start[1]:.3f}) "
            f"end=({line.end[0]:.3f}, {line.end[1]:.3f}) | "
            f"+{line.positive_direction} / -{line.negative_direction}"
        )
    return 0


def _doctor() -> int:
    print(f"traffic-video-analytics {__version__}")
    print(f"Python      : {platform.python_version()} ({platform.platform()})")
    print(f"CPU cores   : {os.cpu_count()}")
    print()
    packages = [
        ("opencv-python", "cv2", True),
        ("numpy", "numpy", True),
        ("pandas", "pandas", True),
        ("PyYAML", "yaml", True),
        ("ultralytics", "ultralytics", False),
        ("torch", "torch", False),
    ]
    missing_demo = []
    missing_full = []
    for distribution, module, required_for_demo in packages:
        if importlib.util.find_spec(module) is None:
            status = "MISSING"
            (missing_demo if required_for_demo else missing_full).append(distribution)
        else:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
            status = f"ok ({version})"
        print(f"{distribution:<12}: {status}")

    print()
    if missing_full:
        print(
            "Real video analysis needs: "
            + ", ".join(missing_full)
            + ". Install with 'pip install -e .' (CPU torch wheels recommended)."
        )
    else:
        print("Full YOLO + ByteTrack runtime looks ready.")
    if missing_demo:
        print("The offline demo still needs: " + ", ".join(missing_demo))
    else:
        print("Offline demo dependencies are ready: 'traffic-analytics demo'")
    return 0 if not missing_demo else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        if args.command == "analyze":
            from .pipeline import process_video

            config = load_config(args.config)
            config = _apply_overrides(config, args)
            print(f"Using config: {Path(args.config)}")
            print(describe_config(config))
            print()
            result = process_video(
                config,
                output_video=args.output_video,
                events_path=args.events,
                reports_dir=args.reports_dir,
                write_video=not args.no_video,
                generate_reports=not args.no_reports,
                run_id=args.run_id,
            )
            _print_run_summary(result)
            return 0

        if args.command == "summarize":
            from .analytics import load_events, summarize_events, write_reports

            events = load_events(args.events)
            summary = summarize_events(
                events,
                interval_seconds=args.interval,
                camera_id=args.camera_id,
            )
            paths = write_reports(events, summary, args.out, interval_seconds=args.interval)
            print(f"Loaded {len(events)} events from {args.events}")
            print(f"Total events: {summary['total_events']}")
            print(f"By class    : {format_counts(summary.get('by_class') or {})}")
            print(f"By direction: {format_counts(summary.get('by_direction') or {})}")
            for name, path in paths.items():
                print(f"{name:<14}: {path}")
            return 0

        if args.command == "demo":
            from .demo import run_demo

            demo = run_demo(
                args.output_dir,
                duration=args.duration,
                fps=args.fps,
                width=args.width,
                height=args.height,
                seed=args.seed,
                frame_stride=args.frame_stride,
                preview_frame=args.preview_frame,
            )
            _print_run_summary(demo.run)
            print(f"Input video  : {demo.input_video}")
            print(f"Manifest     : {demo.manifest_path}")
            if demo.preview_path:
                print(f"Preview      : {demo.preview_path}")
            print(f"Scene vehicles: {demo.vehicle_count} generated, {demo.run.total_events} counted")
            print()
            print("Note: this scene is synthetic. Use 'analyze' with real footage for YOLO detection.")
            return 0

        if args.command == "preview-lines":
            return _preview_lines(args)

        if args.command == "doctor":
            return _doctor()

        parser.error(f"unknown command: {args.command}")
        return 2
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
