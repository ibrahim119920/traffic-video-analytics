"""Tests for YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from traffic_analytics.config import ConfigError, load_config


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = """
camera_id: camera_01
model:
  weights: yolo11n.pt
  classes: [2, 3, 5, 7]
video:
  path: data/raw/traffic.mp4
  frame_stride: 2
lines:
  - name: main_line
    start: [0.15, 0.55]
    end: [0.85, 0.55]
    positive_direction: southbound
    negative_direction: northbound
analytics:
  interval_seconds: 300
"""


def test_load_valid_config(tmp_path: Path):
    config = load_config(write_config(tmp_path, VALID))
    assert config.camera_id == "camera_01"
    assert config.video.frame_stride == 2
    assert config.lines[0].name == "main_line"
    assert config.lines[0].start == (0.15, 0.55)
    assert config.model.classes == [2, 3, 5, 7]


def test_unknown_top_level_key_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(write_config(tmp_path, VALID + "\nnot_a_setting: 1\n"))


def test_unknown_nested_key_is_rejected(tmp_path: Path):
    text = VALID.replace("frame_stride: 2", "frame_stride: 2\n  mystery: true")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(write_config(tmp_path, text))


def test_zero_frame_stride_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="frame_stride"):
        load_config(write_config(tmp_path, VALID.replace("frame_stride: 2", "frame_stride: 0")))


def test_normalized_line_coordinates_are_enforced(tmp_path: Path):
    with pytest.raises(ConfigError, match="normalized"):
        load_config(write_config(tmp_path, VALID.replace("start: [0.15, 0.55]", "start: [1.5, 0.55]")))


def test_duplicate_line_names_are_rejected(tmp_path: Path):
    duplicate = """
lines:
  - name: same
    start: [0.1, 0.5]
    end: [0.9, 0.5]
  - name: same
    start: [0.1, 0.6]
    end: [0.9, 0.6]
"""
    with pytest.raises(ConfigError, match="duplicated"):
        load_config(write_config(tmp_path, VALID + duplicate))


def test_missing_line_points_are_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="requires both"):
        load_config(write_config(tmp_path, VALID.replace("    start: [0.15, 0.55]\n", "")))


def test_missing_file_has_clear_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml")
