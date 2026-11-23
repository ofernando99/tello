# Tello Autonomous Navigation (Mapping, Path Planning & Drone Control)

Autonomous navigation and real-time visual object tracking pipeline for the DJI Tello drone using occupancy grid path planning and computer vision.

## Contributors
- **Jessie W** (Occupancy Grid Mapping, Benchmarking)
- **Anna Zarubina** (Path Planning, Algorithms)
- **Oshanda Fernando** (Drone Control, Navigation)
- **Ganbat Selenge** (OpenCV, Object Tracking)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate 

pip install -r requirements.txt
```

## Occupancy Grid Mapping

Converts 2D arena maps (`.json`, `.txt`, `.png`, `.py`) into `occupancy_grid` (`0` = free, `1` = obstacle), `start`, and `destination` (see [`mapping/README.md`](mapping/README.md)):

```python
from mapping.mapping import load_map

occupancy_grid, start, destination = load_map("mapping/maps/example_arena.png")
```

## Checkerboard tracking

Run from the `tello` directory.

### Live tracking on the ground

On Windows PowerShell, install the dependencies while connected to the internet.

```powershell
Set-Location 'C:\Users\ganbat\Projects\robotics\tello'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Power on the Tello, leave it on a stable surface, and connect your computer to its TELLO Wi-Fi. Close any other Tello program using UDP ports 8889 or 11111. Point the camera toward a checkerboard with 15 by 10 inner corners (16 by 11 squares).

```powershell
.\.venv\Scripts\python.exe -m object_tracking.tracking_ground --seconds 60
```

This mode sends only `command`, `streamon`, and `streamoff`. It writes a CSV under `log/`. Press Esc or Q in the preview to stop early. Add `--headless` for terminal output and use Ctrl+C to stop.

### Recorded video and tracking during flight

```bash
python -m object_tracking.tracker ../Tello_video.mp4
python drone_control_sdk_go.py --hover-seconds 10
```

For live tracking, connect to the Tello Wi-Fi, keep UDP port 11111 free, then select a map and confirm the route. Either flight script tracks during the destination hover, saves logs under `log/`, and attempts to land when tracking ends.

The preview shows pixel coordinates or `target lost`. Add `--headless` to print results in the terminal. Press Esc or Q to end the preview.

[Pipeline](tracking_figures/pipeline.png) | [Annotated frames](tracking_figures/detections.png) | [Coordinate trace](tracking_figures/centre_trace.png)
