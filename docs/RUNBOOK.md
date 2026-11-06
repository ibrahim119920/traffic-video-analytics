# Runbook

Operational instructions for the two supported modes: the offline synthetic demo and real
video analysis with YOLO + ByteTrack.

## 1. Prerequisites

- Python 3.10+ (tested on 3.11).
- Windows, Linux, or macOS. CPU-only is supported.
- Optional: FFmpeg for re-encoding difficult videos (`ffmpeg -version`).

Create an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Installation

### Demo only (no PyTorch)

```powershell
pip install -r requirements-demo.txt
pip install -e . --no-deps
traffic-analytics doctor
```

Expected: OpenCV, NumPy, Pandas, PyYAML are present; Ultralytics/Torch may be missing.
On headless Linux, replace `opencv-python` with `opencv-python-headless` to avoid GUI
library requirements.

### Full analysis (CPU PyTorch)

Windows/Linux:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
traffic-analytics doctor
```

macOS:

```powershell
pip install -e .
```

Verify CPU-only torch:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` for CUDA is expected and fine.

## 3. Demo workflow

```powershell
traffic-analytics demo --output-dir output/demo
```

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--duration` | 30 | Seconds of synthetic scene. |
| `--fps` | 25 | Source FPS. |
| `--width`, `--height` | 1280, 720 | Output resolution. |
| `--seed` | 7 | Deterministic traffic pattern. |
| `--frame-stride` | 1 | Sampling stride for the pipeline. |
| `--preview-frame` | 380 | Frame used for `preview.jpg`. |

The demo writes an input clip and a per-frame manifest, then replays the manifest through
the same counting/annotation/reporting pipeline. It validates pipeline behavior, not YOLO
accuracy. Use real footage for model evaluation.

Inspect:

- `output/demo/demo_annotated.mp4`
- `output/demo/demo_events.jsonl`
- `output/demo/reports/summary.md`

## 4. Real video workflow

### 4.1 Prepare the clip

Place footage in `data/raw/` and set `video.path`, or pass `--video`. H.264/AAC MP4 is the
most reliable input. If OpenCV cannot decode a file, re-encode:

```bash
ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p -an data/raw/traffic.mp4
```

Keep resolution reasonable (720p or 1080p). Very high-resolution footage slows
detection without improving counts much.

### 4.2 Calibrate the counting line

Draw the current configuration on a representative frame:

```powershell
traffic-analytics preview-lines --video data/raw/traffic.mp4 --frame 0
```

Open `output/line_preview.jpg`. Coordinates are normalized:

```text
x_norm = pixel_x / frame_width
y_norm = pixel_y / frame_height
```

Edit `config/config.yaml`:

```yaml
lines:
  - name: north_gate
    start: [0.18, 0.62]
    end: [0.82, 0.58]      # diagonal lines are supported
    positive_direction: southbound
    negative_direction: northbound
```

Direction semantics:

- The line has a direction from `start` to `end`.
- The positive side is where the cross product is positive; for a horizontal
  left-to-right line this is the lower part of the image.
- A track moving from negative to positive side reports `positive_direction`.
- If reports are inverted, swap `positive_direction` and `negative_direction`.

Test again with `preview-lines`; repeat until arrows/labels match real traffic flow.

### 4.3 Run

```powershell
traffic-analytics analyze --video data/raw/traffic.mp4
```

CPU-friendly first pass:

```powershell
traffic-analytics analyze --video data/raw/traffic.mp4 --frame-stride 3 --imgsz 480 --no-video
```

Outputs go to `output/videos`, `output/events`, and `output/reports/<run_id>`.

### 4.4 Verify a run

1. Open `summary.md` and check counts are plausible for the clip duration.
2. Open the annotated video and confirm boxes/IDs align with vehicles.
3. Check that every event has a direction and that totals equal the sum of intervals.
4. Spot-check `events.jsonl` timestamps increase and `event_id` values are unique.

### 4.5 Google Colab (GPU)

The project does not need to be rewritten as a notebook. Use
[`notebooks/colab_quickstart.ipynb`](../notebooks/colab_quickstart.ipynb) as a thin runner:

1. Select **Runtime > Change runtime type > T4 GPU** (or another GPU).
2. Upload the project zip to Drive or clone the repository into `/content`.
3. Install with `pip install -q ultralytics` and `pip install -q -e . --no-deps`.
   Do **not** install the CPU PyTorch wheel; Colab already has CUDA PyTorch.
4. Set the video path, calibrate the line, and run:

   ```python
   !traffic-analytics analyze --config config/config.yaml --device 0
   ```

5. Copy `output/` to Google Drive; Colab deletes `/content` when the runtime ends.

The notebook converts the annotated `mp4v` video to H.264 before inline playback because
browsers do not reliably play MPEG-4 Part 2. For long clips, use `--max-frames` or process
in chunks to stay inside the session limit.

## 5. Tuning

### Accuracy vs speed

| Symptom | Adjustment |
| --- | --- |
| Too slow on CPU | Increase `frame_stride`, lower `imgsz` (640 -> 480), use `yolo11n.pt`, run with `--no-video`. |
| Missing small/distant vehicles | Increase `imgsz` to 960/1280, use a larger model, lower `model.conf`. |
| Too many false positives | Raise `model.conf` (0.4-0.5), raise `counting.min_confidence`, restrict `model.classes`. |
| IDs switching | Use stride 1, raise confidence, tune ByteTrack (below), avoid severe occlusions. |
| Counts doubled | Check whether the line is duplicated in config; keep `count_reentries: false`. |
| Missed counts | Ensure the line fully spans the road; verify arrow/direction labels; check `min_confidence`. |
| Wrong directions | Swap the direction labels for that line. |

### ByteTrack configuration

Copy Ultralytics' `bytetrack.yaml` to `config/bytetrack_custom.yaml` and adjust:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30       # frames to keep lost tracks alive
match_thresh: 0.8
fuse_score: true
```

Then set `model.tracker: config/bytetrack_custom.yaml`. Increase `track_buffer` for longer
occlusions; decrease it if IDs get incorrectly reattached.

### CPU threading

PyTorch/OpenCV can oversubscribe small machines. If throughput is unstable, cap threads:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
```

## 6. Troubleshooting

| Error / symptom | Cause | Fix |
| --- | --- | --- |
| `ultralytics is not installed` | Demo-only install used for real analysis. | Install full runtime; see section 2. |
| `ModuleNotFoundError: cv2` | Core dependencies missing. | `pip install -r requirements-demo.txt`. |
| `Video not found` | Wrong path or file not downloaded. | Check `--video` and `data/raw/`. |
| `Config file not found` | Running outside project root or wrong `--config`. | `cd` to project root or pass an absolute config path. |
| `Unknown key(s) in ...` | Typo in YAML. | Fix the key; unknown keys are rejected intentionally. |
| Zero events | Line misplaced, wrong direction config, confidence too high, or no traffic in clip. | Run `preview-lines`, lower thresholds, verify footage. |
| All events have the same direction | One-way traffic or line direction assumption wrong. | Check arrows and swap labels if needed. |
| `OpenCV could not create a video writer` | Codec/container issue. | Write to `.avi` (`--output-video out.avi`) or reinstall `opencv-python` with FFmpeg. |
| Annotated video too fast/slow | Custom `output_fps` or unusual source FPS. | Leave `video.output_fps: null` or set it explicitly. |
| Timestamps offset | `base_timestamp` unset or FPS misdetected. | Set `counting.base_timestamp`; check `source_fps` in `summary.json`. |
| Very low processing FPS | Large model/resolution, stride 1, annotation enabled. | Stride 2-3, imgsz 480, `--no-video`. |
| `scipy requires numpy<...` warning | Existing environment has conflicting packages. | Use a clean virtualenv. |

## 7. Reprocessing existing events

Rebuild reports without touching the video:

```powershell
traffic-analytics summarize output/events/<run_id>.jsonl --interval 60 --out output/reports/rebuild
```

This is useful for changing bucket size, correcting camera metadata, or re-running
analytics in a data pipeline.

## 8. Housekeeping and storage

- Raw/processed media and generated `output/` are gitignored.
- A 30-second 720p annotated run is typically 10-20 MB; use `--no-video` for bulk jobs.
- Committed sample artifacts live in `docs/sample_output/`; the generated input clip is not
  committed to keep the repository small.
- Delete old runs manually or add a retention policy in your orchestration layer.

## 9. Licensing notes

Use footage you are allowed to process. Public datasets and stock clips have their own
licenses. Ultralytics YOLO is AGPL-3.0 by default; confirm compliance before distributing
derived weights or a hosted service. This repository's own code is MIT.
