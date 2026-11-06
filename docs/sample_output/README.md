# Sample output

These artifacts were produced with the offline synthetic demo, so they are reproducible
without YOLO, PyTorch, or a video download:

```powershell
traffic-analytics demo --output-dir docs/sample_output --duration 30 --fps 25 --preview-frame 380
```

> This scene is synthetic. It validates the detection-agnostic pipeline (tracking replay,
> counting, annotation, reporting) but is **not** an evaluation of YOLO accuracy. Use real
> footage with `traffic-analytics analyze` for model evaluation.

## Files

| File | Description |
| --- | --- |
| `demo_annotated.mp4` | Annotated 30-second result with boxes, IDs, trails, line, and HUD. |
| `demo_events.jsonl` | 47 line-crossing events, one JSON object per line. |
| `preview.jpg` | Single annotated frame used in the main README. |
| `reports/summary.md` | Human-readable traffic report. |
| `reports/summary.json` | Machine-readable summary. |
| `reports/events.csv` | Event table. |
| `reports/timeseries.csv` | Counts by 10-second interval, class, and direction. |
| `reports/breakdown.csv` | Class x direction totals. |
| `reports/run_config.json` | Effective configuration snapshot for reproducibility. |

## Headline numbers

| Metric | Value |
| --- | --- |
| Duration | 30 s @ 25 fps (750 frames) |
| Counted vehicles | 47 of 59 generated (late spawns had not crossed when the clip ended) |
| Classes | car 30, motorcycle 9, truck 6, bus 2 |
| Directions | southbound 24, northbound 23 |
| Peak interval | 08:00:10 - 08:00:20 (18 vehicles) |
| Processing | 13.6 s, ~55 fps including annotation (host-dependent) |

The demo also generates `input/demo_input.mp4` and an input manifest. Those files are
intentionally not committed to keep the repository small; rerun the command above to
regenerate them. The `source_video` path inside the reports therefore points to a
regenerated input file.

## Rebuild reports from the committed events

```powershell
traffic-analytics summarize docs/sample_output/demo_events.jsonl --interval 10 --out output/reports/sample
```
