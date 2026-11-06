# Traffic Analytics Summary

- **Camera:** `demo_camera_01`
- **Generated:** 2026-11-06T00:58:42
- **Period:** 2026-11-06T08:00:00 to 2026-11-06T08:00:29
- **Bucket size:** 10 seconds
- **Total line-crossing events:** **47**
- **Source video:** `docs\sample_output\input\demo_input.mp4`
- **Frames processed:** 750 (stride 1, 55.15 fps processing)

## Vehicles by class

| Class | Count | Share |
| --- | --- | --- |
| car | 30 | 63.8% |
| motorcycle | 9 | 19.1% |
| truck | 6 | 12.8% |
| bus | 2 | 4.3% |

## Vehicles by direction

| Direction | Count | Share |
| --- | --- | --- |
| southbound | 24 | 51.1% |
| northbound | 23 | 48.9% |

## Class x direction

| Class | Direction | Count |
| --- | --- | --- |
| car | northbound | 16 |
| car | southbound | 14 |
| motorcycle | southbound | 7 |
| truck | northbound | 4 |
| motorcycle | northbound | 2 |
| truck | southbound | 2 |
| bus | northbound | 1 |
| bus | southbound | 1 |

## Counting lines

| Line | Count |
| --- | --- |
| demo_line | 47 |

## Traffic by time interval

| Interval start | Interval end | Total | By class | By direction |
| --- | --- | --- | --- | --- |
| 2026-11-06T08:00:00 | 2026-11-06T08:00:10 | 15 | car=9, motorcycle=3, bus=2, truck=1 | southbound=8, northbound=7 |
| 2026-11-06T08:00:10 | 2026-11-06T08:00:20 | 18 | car=12, motorcycle=3, truck=3 | northbound=9, southbound=9 |
| 2026-11-06T08:00:20 | 2026-11-06T08:00:30 | 14 | car=9, motorcycle=3, truck=2 | northbound=7, southbound=7 |

**Peak interval:** 2026-11-06T08:00:10 with 18 vehicles.

## Run metadata

| Key | Value |
| --- | --- |
| base_timestamp | 2026-11-06T08:00:00 |
| detector | synthetic-manifest |
| duration_s | 30.0 |
| frame_stride | 1 |
| frames_processed | 750 |
| frames_read | 750 |
| output_fps | 25.0 |
| processing_fps | 55.15 |
| processing_seconds | 13.599 |
| run_id | demo_run |
| source_fps | 25.0 |
| source_video | docs\sample_output\input\demo_input.mp4 |
