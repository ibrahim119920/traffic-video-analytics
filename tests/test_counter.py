"""Unit tests for virtual-line crossing logic."""

from __future__ import annotations

from traffic_analytics.counter import CrossingLine, LineCounter
from traffic_analytics.models import TrackedObject


def make_object(track_id: int, x: float, y: float, confidence: float = 0.9) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=confidence,
        bbox=(x - 10, y - 6, x + 10, y + 6),
    )


def make_line() -> CrossingLine:
    return CrossingLine(
        name="main",
        start=(100.0, 100.0),
        end=(300.0, 100.0),
        positive_direction="southbound",
        negative_direction="northbound",
    )


def update(counter: LineCounter, y: float, frame: int, track_id: int = 7, confidence: float = 0.9):
    return counter.update(
        [make_object(track_id, 200.0, y, confidence)],
        frame_index=frame,
        timestamp=f"2026-11-06T08:00:{frame % 60:02d}",
        video_time_s=frame / 25.0,
    )


def test_downward_crossing_emits_one_event():
    counter = LineCounter([make_line()], camera_id="camera_01")

    assert update(counter, 80.0, 1) == []
    events = update(counter, 120.0, 2)
    assert len(events) == 1

    event = events[0]
    assert event.camera_id == "camera_01"
    assert event.direction == "southbound"
    assert event.object_class == "car"
    assert event.track_id == 7
    assert event.event_type == "line_crossing"
    assert event.line_name == "main"
    assert event.frame_index == 2

    # The same track is never counted twice on the same line by default.
    assert update(counter, 160.0, 3) == []
    assert counter.event_count == 1
    assert counter.snapshot()["total"] == 1


def test_upward_crossing_uses_negative_direction():
    counter = LineCounter([make_line()])
    update(counter, 140.0, 1)
    events = update(counter, 60.0, 2)
    assert len(events) == 1
    assert events[0].direction == "northbound"


def test_crossing_outside_line_segment_is_ignored():
    counter = LineCounter([make_line()])
    first = counter.update(
        [make_object(3, 50.0, 80.0)], frame_index=1, timestamp="2026-11-06T08:00:01"
    )
    second = counter.update(
        [make_object(3, 50.0, 120.0)], frame_index=2, timestamp="2026-11-06T08:00:02"
    )
    assert first == [] and second == []


def test_movement_parallel_to_line_is_ignored():
    counter = LineCounter([make_line()])
    counter.update([make_object(4, 150.0, 90.0)], frame_index=1, timestamp="2026-11-06T08:00:01")
    events = counter.update(
        [make_object(4, 250.0, 90.0)], frame_index=2, timestamp="2026-11-06T08:00:02"
    )
    assert events == []


def test_low_confidence_objects_are_ignored():
    counter = LineCounter([make_line()], min_confidence=0.5)
    update(counter, 80.0, 1, confidence=0.2)
    events = update(counter, 120.0, 2, confidence=0.2)
    assert events == []


def test_reentries_can_be_counted_when_enabled():
    counter = LineCounter([make_line()], count_reentries=True)
    update(counter, 80.0, 1)
    assert len(update(counter, 120.0, 10)) == 1
    assert len(update(counter, 80.0, 20)) == 1
    assert len(update(counter, 120.0, 30)) == 1
    assert counter.snapshot()["total"] == 3


def test_stale_tracks_do_not_produce_crossings():
    counter = LineCounter([make_line()], max_track_age_frames=5, count_reentries=True)
    update(counter, 80.0, 1)
    # A long gap means the two observations cannot be treated as one movement.
    events = counter.update(
        [make_object(7, 200.0, 120.0)],
        frame_index=100,
        timestamp="2026-11-06T08:01:40",
    )
    assert events == []


def test_multiple_lines_are_counted_independently():
    first = CrossingLine("line_a", (100.0, 100.0), (300.0, 100.0), "south", "north")
    second = CrossingLine("line_b", (100.0, 200.0), (300.0, 200.0), "south", "north")
    counter = LineCounter([first, second])
    update(counter, 80.0, 1)
    events = update(counter, 220.0, 2)
    assert {event.line_name for event in events} == {"line_a", "line_b"}
    assert counter.snapshot()["by_line"] == {"line_a": 1, "line_b": 1}
