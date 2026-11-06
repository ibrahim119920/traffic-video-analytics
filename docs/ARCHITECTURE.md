# Architecture

## Goals

This project is a small, inspectable computer-vision pipeline that converts traffic video
into analysis-ready data. It prioritises:

1. **Reproducibility** - typed config, deterministic demo, committed sample artifacts.
2. **CPU usability** - pretrained small models, frame sampling, optional annotation.
3. **Clean data contract** - streamed JSONL events with a documented schema.
4. **Separation of concerns** - detection, counting, rendering, and reporting are independent.

## Pipeline

```text
 video file
    |
    v
+----------------+     grab()/retrieve()     +---------------------+
| Frame sampler  | -------------------------> | Detector backend    |
| stride = N     |                            | YOLO + ByteTrack    |
+----------------+                            +----------+----------+
                                                         |
                                               list[TrackedObject]
                                                         |
                                                         v
                                              +---------------------+
                                              | LineCounter         |
                                              | pixel-space segments|
                                              | direction + dedupe  |
                                              +----+-----------+----+
                                                   |           |
                                      CrossingEvent|           |snapshot()
                                                   v           v
                                        +----------------+  +----------------+
                                        | JSONL writer   |  | Annotator      |
                                        | (streamed)     |  | OpenCV overlay |
                                        +-------+--------+  +-------+--------+
                                                |                   |
                                                v                   v
                                        events/<run>.jsonl   videos/<run>.mp4
                                                |
                                                v
                                        +----------------+
                                        | analytics.py   |
                                        | Pandas reports |
                                        +----------------+
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Parse YAML into dataclasses; validate ranges, lines, and enums. |
| `models.py` | `TrackedObject` and `CrossingEvent` domain models. |
| `detector.py` | `Detector` protocol and the YOLO + ByteTrack implementation. |
| `counter.py` | `CrossingLine` geometry and `LineCounter` state machine. |
| `annotator.py` | Boxes, labels, trails, line arrows, HUD, status bar. |
| `pipeline.py` | Video I/O, frame loop, event streaming, report orchestration. |
| `analytics.py` | Summaries, CSV exports, Markdown rendering. |
| `demo.py` | Synthetic scene generator and manifest replay backend. |
| `cli.py` | Argument parsing and command dispatch. |

## Coordinate spaces

Two coordinate spaces are used deliberately:

- **Configuration and drawing:** normalized `(x, y)` in `[0, 1]`. Resolution-independent;
  the annotator multiplies by frame width/height at render time.
- **Counting:** pixels. Track centres come from pixel bounding boxes, so the pipeline
  converts each normalized line to pixel coordinates once, after reading frame dimensions.

Keeping these separate avoids the classic bug where a normalized line is compared against
pixel centres. `tests/test_counter.py` exercises the geometry directly in pixels.

## Crossing algorithm

For each active track, the counter keeps the previous centre and frame index. On every
processed frame it tests the segment `previous -> current` against each line segment.

```text
              positive side (southbound)
    ------------------ line ------------------
              negative side (northbound)

  previous (side < 0)            current (side > 0)
            \                        /
             \   crossing point     /
              \--------------------/
```

1. Compute the signed side value:
   `side(p) = (end.x - start.x) * (p.y - start.y) - (end.y - start.y) * (p.x - start.x)`.
2. Test segment intersection in pixel space using the standard parametric solution. This
   prevents counting a vehicle that crosses the infinite line outside the drawn segment.
3. Direction comes from the sign transition:
   - negative -> positive: `positive_direction`
   - positive -> negative: `negative_direction`
4. Record `(line_name, track_id)` in a dedupe set. By default each track is counted once
   per line for the entire video.
5. Optionally allow re-entries (`counting.count_reentries: true`) with a short cooldown to
   suppress jitter around the line.

Edge cases handled:

- Parallel movement never counts.
- A line with zero length is rejected during config/line validation.
- A track that disappears for more than `max_track_age_frames` is not connected across the
  gap, so teleporting IDs do not fabricate crossings.
- Interpolated readings between sampled frames are safe because direction is decided from
  the two observed centres that form the crossing segment.

## Frame sampling and time

`frame_stride = N` reads all frames with `grab()` but only decodes and processes every Nth
frame. This reduces inference and annotation cost while preserving source timing:

```text
timestamp(frame) = base_timestamp + frame_index / source_fps
output_fps       = source_fps / frame_stride   (unless explicitly overridden)
```

Consequences:

- `max_frames` counts **source frames**, not processed frames.
- `max_track_age_frames` is expressed in **source frames**.
- The output video plays at `output_fps`, so its duration approximates the input duration.

`base_timestamp` can be fixed in config for reproducible datasets, or left empty to use
the run start time. Timestamps are naive local ISO-8601 strings; downstream systems should
attach the camera timezone explicitly if needed.

## Detection and tracking

The default backend calls:

```python
model.track(
    source=frame,
    persist=True,
    tracker="bytetrack.yaml",
    conf=0.35,
    iou=0.5,
    imgsz=640,
    classes=[2, 3, 5, 7],   # car, motorcycle, bus, truck
    device="cpu",
)
```

ByteTrack's Kalman-filter motion model plus low-confidence association keeps IDs stable
through short occlusions. The tracker state lives inside the Ultralytics model instance and
must not be reused across unrelated videos; the pipeline creates a fresh detector per run.

The `Detector` protocol is intentionally tiny:

```python
class Detector(Protocol):
    name: str
    def update(self, frame, frame_index) -> list[TrackedObject]: ...
```

This allows alternative backends (a different model, a remote inference service, or the
synthetic manifest replay used by the demo) without touching the pipeline.

## Annotation

The annotator is stateful only for short motion trails. Every frame it draws:

- virtual line(s) with direction arrows and labels;
- per-track corner boxes with class, ID, and confidence;
- optional trails;
- a HUD with camera, timestamp, class counts, total count, and direction split;
- a status bar with frame index and measured processing throughput.

Annotation is skipped entirely with `--no-video`, which is the fastest mode for batch data
generation.

## Reports

`analytics.summarize_events()` computes:

- totals by class, direction, line, and class x direction;
- min/max event timestamps;
- fixed-width time buckets (`interval_seconds`) with per-class and per-direction counts;
- the peak interval.

`write_reports()` materialises `events.csv`, `summary.json`, `summary.md`,
`timeseries.csv`, and `breakdown.csv`. The pipeline also snapshots the effective config as
`run_config.json` for reproducibility.

## Failure behaviour

- Missing config/video or invalid settings fail fast with a clear `ConfigError`/`FileNotFoundError`.
- Events stream to JSONL as they happen; if a run is interrupted, the events collected so
  far remain readable.
- Reports are generated after the frame loop; an interrupted run will not produce a summary.
- OpenCV writer failure raises a targeted error suggesting an alternate container/install.

## Extension points

| Need | Where to change |
| --- | --- |
| Different detector/model | Implement `Detector.update` in `detector.py`. |
| Zone or multi-line counting | Extend `counter.py` with polygon containment events. |
| Speed estimation | Add a calibrated homography and per-track timestamps. |
| New report format | Extend `analytics.write_reports`. |
| Live stream | Replace file reading in `pipeline.py` with a bounded RTSP reader. |
