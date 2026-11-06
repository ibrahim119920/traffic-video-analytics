"""Contract test: emitted events must validate against the published JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from traffic_analytics.counter import CrossingLine, LineCounter
from traffic_analytics.models import TrackedObject

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "schema" / "vehicle_event.schema.json"


def test_emitted_event_matches_published_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    line = CrossingLine("main", (0.0, 0.5), (1.0, 0.5), "southbound", "northbound")
    counter = LineCounter([line], camera_id="camera_01")

    counter.update(
        [TrackedObject(42, 2, "car", 0.94, (0.4, 0.4, 0.6, 0.45))],
        frame_index=1,
        timestamp="2026-11-06T08:30:14",
        video_time_s=0.04,
    )
    events = counter.update(
        [TrackedObject(42, 2, "car", 0.94, (0.4, 0.5, 0.6, 0.55))],
        frame_index=2,
        timestamp="2026-11-06T08:30:15",
        video_time_s=0.08,
    )

    assert len(events) == 1
    record = events[0].to_dict()
    jsonschema.validate(record, schema)
    assert record["schema_version"] == "1.0"
    assert record["direction"] == "southbound"
