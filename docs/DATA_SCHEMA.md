# Data Schema

This document defines the vehicle event contract and the report artifacts produced by the
pipeline. The machine-readable event schema lives at
[`schema/vehicle_event.schema.json`](schema/vehicle_event.schema.json).

## Event stream (JSONL)

Each processed vehicle crossing appends exactly one JSON object to
`output/events/<run_id>.jsonl`:

- UTF-8, one object per line, no trailing commas or wrapping array.
- Events appear in crossing order (frame index ascending).
- Each line is flushed immediately, so partial files remain valid JSONL.
- An empty file means no vehicles crossed the configured line(s).

Example:

```json
{"camera_id":"demo_camera_01","timestamp":"2026-11-06T08:00:00","track_id":17,"object_class":"car","confidence":0.89,"event_type":"line_crossing","direction":"southbound","line_name":"demo_line","class_id":2,"frame_index":3,"video_time_s":0.12,"schema_version":"1.0","event_id":"6262aa0cf20647c4b107ca075b384721"}
```

### Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `camera_id` | string | yes | Logical camera identifier from config. |
| `timestamp` | string | yes | Naive local ISO-8601 time: `YYYY-MM-DDTHH:MM:SS`. Derived from `base_timestamp + frame_index / source_fps`. |
| `track_id` | integer | yes | ByteTrack track identifier. Unique within a run, not globally. |
| `object_class` | string | yes | Vehicle class name: `car`, `motorcycle`, `bus`, `truck` (config-dependent). |
| `confidence` | number | yes | Detection confidence, `0.0 - 1.0`, rounded to 4 decimals. |
| `event_type` | string | yes | Always `line_crossing` in schema 1.0. |
| `direction` | string | yes | Direction label configured for the crossed line, e.g. `southbound`. |
| `line_name` | string | no | Name of the crossed virtual line. |
| `class_id` | integer | no | COCO class id (`2` car, `3` motorcycle, `5` bus, `7` truck). |
| `frame_index` | integer | no | Source frame index (not processed-frame index). |
| `video_time_s` | number | no | Source video time in seconds: `frame_index / source_fps`. |
| `schema_version` | string | no | Event contract version, currently `1.0`. |
| `event_id` | string | no | UUID4 hex, 32 characters. Use for downstream deduplication. |

The first seven fields form the **stable contract**. New optional fields may be added in
future versions; consumers should ignore unknown keys and check `schema_version` if strict.

### Semantics

- A `line_crossing` event is emitted when the segment between two consecutive observed
  track centres intersects the line segment.
- By default a track is counted at most once per line for the whole run
  (`counting.count_reentries: false`). Enabling re-entries can emit multiple events for one
  `track_id` on the same line.
- `direction` values are not hardcoded in code; they come from `lines[].positive_direction`
  and `lines[].negative_direction`. A downstream pipeline should treat them as labels, not
  an enum.
- `confidence` reflects the detector score at the crossing frame. A second filter applies
  `counting.min_confidence` before counting.
- `track_id` may be reused by the tracker after very long gaps; `event_id` is the safe
  unique key.

## CSV exports

### `events.csv`

Same records as `events.jsonl`, one row per event, columns:

```text
camera_id, timestamp, track_id, object_class, confidence, event_type, direction,
line_name, class_id, frame_index, video_time_s, schema_version, event_id
```

### `timeseries.csv`

Aggregated by time bucket, class, and direction:

| Column | Description |
| --- | --- |
| `interval_start` | Bucket start timestamp (ISO-8601). |
| `interval_end` | Bucket end timestamp (start + `analytics.interval_seconds`). |
| `object_class` | Vehicle class. |
| `direction` | Direction label. |
| `count` | Number of events in this bucket/class/direction. |

Summing `count` over all rows equals `summary.total_events`.

### `breakdown.csv`

| Column | Description |
| --- | --- |
| `object_class` | Vehicle class. |
| `direction` | Direction label. |
| `count` | Total events for the pair. |

## Summary JSON

`summary.json` is the machine-readable run report:

```json
{
  "schema_version": "1.0",
  "camera_id": "demo_camera_01",
  "generated_at": "2026-11-06T00:51:58",
  "time_range": {"start": "2026-11-06T08:00:00", "end": "2026-11-06T08:00:29"},
  "interval_seconds": 10,
  "total_events": 47,
  "by_class": {"car": 30, "motorcycle": 9, "truck": 6, "bus": 2},
  "by_direction": {"southbound": 24, "northbound": 23},
  "by_line": {"demo_line": 47},
  "by_class_direction": [
    {"object_class": "car", "direction": "northbound", "count": 16}
  ],
  "peak_interval": {
    "interval_start": "2026-11-06T08:00:10",
    "interval_end": "2026-11-06T08:00:20",
    "count": 18,
    "by_class": {"car": 12, "motorcycle": 3, "truck": 3},
    "by_direction": {"northbound": 9, "southbound": 9}
  },
  "intervals": [
    {
      "interval_start": "2026-11-06T08:00:00",
      "interval_end": "2026-11-06T08:00:10",
      "count": 15,
      "by_class": {"car": 9, "motorcycle": 3, "bus": 2, "truck": 1},
      "by_direction": {"southbound": 8, "northbound": 7}
    }
  ],
  "run": {"run_id": "demo_run", "detector": "synthetic-manifest"}
}
```

*Abridged example: `intervals` shows one bucket and `run` shows two keys.*

| Top-level key | Description |
| --- | --- |
| `schema_version` | Summary schema version. |
| `camera_id` | Camera used for the run (or most common in loaded events). |
| `generated_at` | Report creation time, local naive ISO-8601. |
| `time_range` | First/last event timestamps. |
| `interval_seconds` | Bucket width. |
| `total_events` | Total counted crossings. |
| `by_class`, `by_direction`, `by_line` | Count maps, sorted by descending count. |
| `by_class_direction` | List of `{object_class, direction, count}`. |
| `peak_interval` | Busiest interval object, or `null` when empty. |
| `intervals` | Ordered list of interval objects (`interval_start`, `interval_end`, `count`, `by_class`, `by_direction`). |
| `run` | Run metadata (see below). |

### `run` metadata

| Field | Description |
| --- | --- |
| `run_id` | Output run identifier. |
| `source_video` | Input path at processing time. |
| `detector` | Backend description (`yolo+bytetrack:...` or `synthetic-manifest`). |
| `frames_read` | Source frames read (including skipped). |
| `frames_processed` | Frames that went through detection/annotation. |
| `frame_stride` | Sampling stride. |
| `source_fps` | Source FPS detected by OpenCV. |
| `output_fps` | FPS used for the annotated video. |
| `duration_s` | Covered source duration. |
| `processing_seconds` | Wall-clock processing time. |
| `processing_fps` | Processed frames per wall-clock second. |
| `base_timestamp` | Timestamp mapped to source frame 0. |

## Markdown report

`summary.md` renders the same information for humans: class and direction shares, class x
direction, per-line totals, interval table, peak interval, and run metadata.

## Downstream pipeline expectations

A data pipeline such as **Real-Time Video Analytics Data Pipeline** can:

1. Read JSONL line by line and validate with the JSON Schema.
2. Deduplicate on `event_id` (idempotent ingestion).
3. Partition by `camera_id` and event date.
4. Aggregate by `object_class`, `direction`, and time bucket.
5. Join multiple runs/cameras on the stable core fields.

Recommended handling:

- Parse `timestamp` as local naive time and attach the camera timezone from a reference
  table, or derive UTC at ingestion.
- Do not assume `line_name`, `class_id`, or direction labels are fixed enums.
- Expect additional fields in future `schema_version` releases.
- Treat `track_id` as run-scoped; aggregate vehicles using `event_id`/`system_id`, not
  `track_id`, across runs.
