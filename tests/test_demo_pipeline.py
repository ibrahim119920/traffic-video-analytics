"""End-to-end smoke test for the synthetic demo pipeline."""

from __future__ import annotations

from pathlib import Path

from traffic_analytics.demo import run_demo


def test_demo_pipeline_runs_and_writes_artifacts(tmp_path: Path):
    result = run_demo(
        tmp_path,
        duration=10.0,
        fps=25,
        width=640,
        height=360,
        seed=11,
        frame_stride=1,
        preview_frame=60,
    )

    assert Path(result.input_video).exists()
    assert Path(result.manifest_path).exists()
    assert result.run.output_video is not None and Path(result.run.output_video).exists()
    assert Path(result.run.events_path).exists()
    assert result.run.reports_dir is not None
    assert (Path(result.run.reports_dir) / "summary.json").exists()
    assert (Path(result.run.reports_dir) / "summary.md").exists()
    assert result.run.frames_processed == 250
    assert result.run.total_events > 0
    assert result.run.summary["total_events"] == result.run.total_events
    assert result.preview_path is not None and Path(result.preview_path).exists()
