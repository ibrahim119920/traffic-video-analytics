"""Tests for event summarization and report writing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traffic_analytics.analytics import load_events, summarize_events, write_reports


def event(timestamp: str, object_class: str, direction: str, track_id: int = 1) -> dict:
    return {
        "camera_id": "camera_01",
        "timestamp": timestamp,
        "track_id": track_id,
        "object_class": object_class,
        "confidence": 0.94,
        "event_type": "line_crossing",
        "direction": direction,
        "line_name": "main_line",
        "class_id": 2,
        "frame_index": track_id * 10,
        "video_time_s": float(track_id),
        "event_id": f"event-{track_id}",
    }


EVENTS = [
    event("2026-11-06T08:00:10", "car", "southbound", 1),
    event("2026-11-06T08:01:20", "truck", "northbound", 2),
    event("2026-11-06T08:06:30", "car", "northbound", 3),
    event("2026-11-06T08:07:40", "car", "southbound", 4),
]


def test_summary_counts_classes_directions_and_intervals():
    summary = summarize_events(EVENTS, interval_seconds=300)

    assert summary["total_events"] == 4
    assert summary["camera_id"] == "camera_01"
    assert summary["by_class"] == {"car": 3, "truck": 1}
    assert summary["by_direction"] == {"northbound": 2, "southbound": 2}
    assert summary["time_range"]["start"] == "2026-11-06T08:00:10"
    assert len(summary["intervals"]) == 2
    assert [interval["count"] for interval in summary["intervals"]] == [2, 2]
    assert summary["peak_interval"] is not None
    assert {
        (row["object_class"], row["direction"]): row["count"]
        for row in summary["by_class_direction"]
    } == {("car", "southbound"): 2, ("car", "northbound"): 1, ("truck", "northbound"): 1}


def test_empty_summary_is_valid():
    summary = summarize_events([], interval_seconds=60)
    assert summary["total_events"] == 0
    assert summary["intervals"] == []
    assert summary["by_class"] == {}
    assert summary["peak_interval"] is None


def test_write_reports_creates_consistent_files(tmp_path: Path):
    summary = summarize_events(EVENTS, interval_seconds=300)
    paths = write_reports(EVENTS, summary, tmp_path, interval_seconds=300)

    for key in ("events_csv", "summary_json", "summary_md", "timeseries_csv", "breakdown_csv"):
        assert Path(paths[key]).exists(), key

    summary_json = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert summary_json["total_events"] == 4

    frame = pd.read_csv(paths["timeseries_csv"])
    assert int(frame["count"].sum()) == 4

    breakdown = pd.read_csv(paths["breakdown_csv"])
    assert int(breakdown["count"].sum()) == 4
    assert set(breakdown["object_class"]) == {"car", "truck"}


def test_write_reports_handles_no_events(tmp_path: Path):
    summary = summarize_events([])
    paths = write_reports([], summary, tmp_path)
    assert Path(paths["events_csv"]).exists()
    assert Path(paths["timeseries_csv"]).exists()
    assert Path(paths["summary_md"]).exists()


def test_load_events_round_trip(tmp_path: Path):
    source = tmp_path / "events.jsonl"
    source.write_text("\n".join(json.dumps(item) for item in EVENTS) + "\n", encoding="utf-8")
    loaded = load_events(source)
    assert len(loaded) == 4
    assert loaded[0]["object_class"] == "car"
