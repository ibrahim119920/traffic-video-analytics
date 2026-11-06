"""Aggregate crossing events into traffic reports (JSON, CSV, Markdown)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .utils import ensure_dir, format_counts

EVENT_COLUMNS = [
    "camera_id",
    "timestamp",
    "track_id",
    "object_class",
    "confidence",
    "event_type",
    "direction",
    "line_name",
    "class_id",
    "frame_index",
    "video_time_s",
    "schema_version",
    "event_id",
]


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Read an events JSONL (or CSV) file into a list of dictionaries."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Events file not found: {source}")
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source).fillna("").to_dict(orient="records")

    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {source} is not a JSON object")
            events.append(record)
    return events


def _as_naive_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _floor_to_interval(value: datetime, interval_seconds: int) -> datetime:
    epoch = datetime(1970, 1, 1)
    total = int((value - epoch).total_seconds())
    return epoch + timedelta(seconds=(total // interval_seconds) * interval_seconds)


def _counts(mapping: Iterable[str]) -> dict[str, int]:
    counter = Counter(mapping)
    return {key: int(value) for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))}


def summarize_events(
    events: Sequence[dict[str, Any]],
    *,
    interval_seconds: int = 300,
    camera_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete summary dictionary from raw crossing events."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    generated_at = datetime.now().replace(microsecond=0).isoformat()
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "camera_id": camera_id or "unknown",
        "generated_at": generated_at,
        "time_range": {"start": None, "end": None},
        "interval_seconds": interval_seconds,
        "total_events": 0,
        "by_class": {},
        "by_direction": {},
        "by_line": {},
        "by_class_direction": [],
        "peak_interval": None,
        "intervals": [],
        "run": meta or {},
    }
    if not events:
        return summary

    rows: list[tuple[datetime, str, str, str]] = []
    for index, event in enumerate(events):
        try:
            timestamp = _as_naive_datetime(str(event["timestamp"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timestamp in event #{index}: {event.get('timestamp')!r}") from exc
        rows.append(
            (
                timestamp,
                str(event.get("object_class") or "unknown"),
                str(event.get("direction") or "unknown"),
                str(event.get("line_name") or "unnamed"),
            )
        )

    if camera_id is None:
        cameras = Counter(str(event.get("camera_id") or "unknown") for event in events)
        summary["camera_id"] = cameras.most_common(1)[0][0]

    by_class = _counts(row[1] for row in rows)
    by_direction = _counts(row[2] for row in rows)
    by_line = _counts(row[3] for row in rows)

    class_direction: Counter[tuple[str, str]] = Counter()
    buckets: dict[datetime, list[tuple[str, str]]] = {}
    for timestamp, class_name, direction, _ in rows:
        class_direction[(class_name, direction)] += 1
        bucket = _floor_to_interval(timestamp, interval_seconds)
        buckets.setdefault(bucket, []).append((class_name, direction))

    intervals: list[dict[str, Any]] = []
    for bucket_start in sorted(buckets):
        bucket_rows = buckets[bucket_start]
        intervals.append(
            {
                "interval_start": bucket_start.isoformat(timespec="seconds"),
                "interval_end": (bucket_start + timedelta(seconds=interval_seconds)).isoformat(
                    timespec="seconds"
                ),
                "count": len(bucket_rows),
                "by_class": _counts(row[0] for row in bucket_rows),
                "by_direction": _counts(row[1] for row in bucket_rows),
            }
        )

    summary.update(
        {
            "total_events": len(rows),
            "time_range": {
                "start": min(row[0] for row in rows).isoformat(timespec="seconds"),
                "end": max(row[0] for row in rows).isoformat(timespec="seconds"),
            },
            "by_class": by_class,
            "by_direction": by_direction,
            "by_line": by_line,
            "by_class_direction": [
                {"object_class": key[0], "direction": key[1], "count": int(count)}
                for key, count in sorted(
                    class_direction.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "peak_interval": max(intervals, key=lambda item: item["count"]) if intervals else None,
            "intervals": intervals,
        }
    )
    return summary


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def write_reports(
    events: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: str | Path,
    *,
    interval_seconds: int | None = None,
) -> dict[str, str]:
    """Write events.csv, summary.json, summary.md, timeseries.csv, breakdown.csv."""
    target = ensure_dir(Path(out_dir))
    interval = int(interval_seconds or summary.get("interval_seconds") or 300)
    paths: dict[str, str] = {}

    frame = pd.DataFrame(list(events), columns=EVENT_COLUMNS) if events else _empty_frame(EVENT_COLUMNS)
    events_csv = target / "events.csv"
    frame.to_csv(events_csv, index=False)
    paths["events_csv"] = str(events_csv)

    summary_json = target / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["summary_json"] = str(summary_json)

    if events:
        data = pd.DataFrame(list(events))
        parsed = pd.to_datetime(data["timestamp"])
        interval_start = parsed.dt.floor(f"{interval}s")
        interval_end = interval_start + pd.Timedelta(seconds=interval)

        timeseries = (
            pd.DataFrame(
                {
                    "interval_start": interval_start,
                    "interval_end": interval_end,
                    "object_class": data["object_class"],
                    "direction": data["direction"],
                }
            )
            .groupby(["interval_start", "interval_end", "object_class", "direction"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["interval_start", "object_class", "direction"])
        )
        timeseries["interval_start"] = timeseries["interval_start"].map(
            lambda value: value.isoformat(timespec="seconds")
        )
        timeseries["interval_end"] = timeseries["interval_end"].map(
            lambda value: value.isoformat(timespec="seconds")
        )

        breakdown = (
            data.groupby(["object_class", "direction"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["count", "object_class", "direction"], ascending=[False, True, True])
        )
    else:
        timeseries = _empty_frame(
            ["interval_start", "interval_end", "object_class", "direction", "count"]
        )
        breakdown = _empty_frame(["object_class", "direction", "count"])

    timeseries_csv = target / "timeseries.csv"
    timeseries.to_csv(timeseries_csv, index=False)
    paths["timeseries_csv"] = str(timeseries_csv)

    breakdown_csv = target / "breakdown.csv"
    breakdown.to_csv(breakdown_csv, index=False)
    paths["breakdown_csv"] = str(breakdown_csv)

    summary_md = target / "summary.md"
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    paths["summary_md"] = str(summary_md)
    return paths


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a shareable Markdown report from a summary dictionary."""
    run = dict(summary.get("run") or {})
    total = int(summary.get("total_events") or 0)
    time_range = summary.get("time_range") or {}

    lines = [
        "# Traffic Analytics Summary",
        "",
        f"- **Camera:** `{summary.get('camera_id', 'unknown')}`",
        f"- **Generated:** {summary.get('generated_at', '-')}",
        f"- **Period:** {time_range.get('start') or '-'} to {time_range.get('end') or '-'}",
        f"- **Bucket size:** {summary.get('interval_seconds', '-')} seconds",
        f"- **Total line-crossing events:** **{total}**",
    ]
    if run.get("source_video"):
        lines.append(f"- **Source video:** `{run['source_video']}`")
    if run.get("frames_processed") is not None:
        lines.append(
            f"- **Frames processed:** {run.get('frames_processed')} "
            f"(stride {run.get('frame_stride', '-')}, {run.get('processing_fps', '-')} fps processing)"
        )
    lines.append("")

    if total == 0:
        lines.extend(["No line-crossing events were recorded.", ""])
        return "\n".join(lines)

    lines.extend(["## Vehicles by class", ""])
    lines.append(
        _md_table(
            ["Class", "Count", "Share"],
            [
                [name, count, f"{count / total * 100:.1f}%"]
                for name, count in (summary.get("by_class") or {}).items()
            ],
        )
    )
    lines.append("")

    lines.extend(["## Vehicles by direction", ""])
    lines.append(
        _md_table(
            ["Direction", "Count", "Share"],
            [
                [name, count, f"{count / total * 100:.1f}%"]
                for name, count in (summary.get("by_direction") or {}).items()
            ],
        )
    )
    lines.append("")

    lines.extend(["## Class x direction", ""])
    lines.append(
        _md_table(
            ["Class", "Direction", "Count"],
            [
                [row["object_class"], row["direction"], row["count"]]
                for row in summary.get("by_class_direction") or []
            ],
        )
    )
    lines.append("")

    if summary.get("by_line"):
        lines.extend(["## Counting lines", ""])
        lines.append(
            _md_table(
                ["Line", "Count"],
                [[name, count] for name, count in summary["by_line"].items()],
            )
        )
        lines.append("")

    lines.extend(["## Traffic by time interval", "", _md_table(
        ["Interval start", "Interval end", "Total", "By class", "By direction"],
        [
            [
                interval["interval_start"],
                interval["interval_end"],
                interval["count"],
                format_counts(interval.get("by_class") or {}),
                format_counts(interval.get("by_direction") or {}),
            ]
            for interval in summary.get("intervals") or []
        ],
    ), ""])

    peak = summary.get("peak_interval")
    if peak:
        lines.append(
            f"**Peak interval:** {peak['interval_start']} with {peak['count']} vehicles."
        )
        lines.append("")

    if run:
        lines.extend(["## Run metadata", ""])
        lines.append(
            _md_table(
                ["Key", "Value"],
                [[key, value] for key, value in sorted(run.items())],
            )
        )
        lines.append("")

    return "\n".join(lines)
