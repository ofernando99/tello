# Tello Autonomous Navigation (Mapping, Path Planning & Drone Control)

Autonomous navigation and real-time visual object tracking pipeline for the DJI Tello drone using occupancy grid path planning and computer vision.

## Contributors
- **Jessie W** (Occupancy Grid Mapping)
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
