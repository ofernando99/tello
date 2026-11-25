# Tello Autonomous Navigation Pipeline

From 2D arena maps to Tello flight commands and live visual target tracking.

## Contributors
- **Jessie W** (Occupancy Grid Mapping, Benchmarks)
- **Anna Zarubina** (Path Planning, Algorithms)
- **Oshanda Fernando** (Drone Control, Navigation)
- **Ganbat Selenge** (Object Tracking)

## Project Structure

```text
tello/
├── main.py                  # End-to-end navigation entrypoint
├── requirements.txt         # Python dependencies
├── mapping/                 # Occupancy grid loaders and obstacle inflation
│   └── maps/                # Sample arena and maze maps
├── path_planning/           # A*, Dijkstra, Greedy, Theta*, RRT, RRT*, and LOS smoothing
├── drone_control/           # Tello SDK wrapper, waypoint transforms, and command generation
├── object_tracking/         # Checkerboard detector and live/recorded tracking scripts
├── simulation/              # DroneBlocks and SDK flight simulation scripts
├── benchmarks/              # Benchmark runner (run_benchmarks.py) and summary report
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate 

pip install -r requirements.txt
```

## Occupancy Grid Mapping

Converts 2D arena maps (`.json`, `.txt`, `.png`/`.jpg`, `.py`) into `occupancy_grid` (`0` = free, `1` = obstacle), `start`, and `destination` (see [`mapping/README.md`](mapping/README.md)):

```python
from mapping.mapping import load_map

occupancy_grid, start, destination = load_map("mapping/maps/example_arena.png")
```

Optional: grow obstacles by 10 cm safety margin

```python
from mapping.mapping import load_map

gmap = load_map("mapping/maps/example_arena.png")
gmap = gmap.inflate(10)  
occupancy_grid, start, destination = gmap
```

## Path Planning

Plans a collision-free `(row, col)` waypoint list using 8-way A\* and line-of-sight smoothing (`path_planning/path_planner.py`):

```python
from path_planning.path_planner import plan_path

waypoints = plan_path(occupancy_grid, start, destination, cell_size_cm=20.0)
```

Additional planners (`dijkstra`, `greedy_best_first`, `theta_star`, `rrt_grid`, `rrt_star`) are available in `path_planning/path_planner_algorithms.py`.

## Drone Control & Navigation

Converts planned waypoints into Tello SDK `go x y 0 speed` commands (`drone_control/path_command.py`), previews the route, checks `battery?`, prompts for confirmation, and saves command logs under `log/`:

```bash
python main.py
```

Or call `navigation(cell_size, starting_angle, speed, occupancy_grid, waypoints)` directly from `drone_control.path_command`.

## Checkerboard Tracking

Detects a **15 × 10 inner-corner** (16 × 11 square) checkerboard (`object_tracking/tracker.py`):

```python
from object_tracking.tracker import detect_board

detection = detect_board(frame)  # returns (corners, (x, y)) or None
```

### 1. Video / Camera (`object_tracking.tracker`)
Accepts a video path, camera index, or stream URL:

```bash
python -m object_tracking.tracker data/Tello_video.mp4
python -m object_tracking.tracker 0 --headless
```

### 2. Live Ground Test (`object_tracking.tracking_ground`)
Connect to `TELLO` Wi-Fi (UDP `8889`/`11111` free). Streams without taking off and logs to `log/`:

```bash
python -m object_tracking.tracking_ground --seconds 60
```

### 3. Flight Hover API (`object_tracking.tracking_flight`)
`fly_and_track(commands, duration, headless)` handles `takeoff`, route execution, hover tracking, `land`, and CSV/command logging to `log/`.

### Controls
Press `Esc` or `Q` to quit the preview (`x`, `y` or `target lost`), or pass `--headless` for terminal output.

**Visuals:** [Pipeline](tracking_figures/pipeline.png) · [Annotated frames](tracking_figures/detections.png) · [Coordinate trace](tracking_figures/centre_trace.png)

## Benchmarks

Run the benchmark suite across mapping, path planning, Tello command generation, and checkerboard tracking:

```bash
python benchmarks/run_benchmarks.py                    # Full pipeline (Task 1 + Task 2)
python benchmarks/run_benchmarks.py --task 1           # Navigation only
python benchmarks/run_benchmarks.py --task 2           # Checkerboard tracking only
python benchmarks/run_benchmarks.py mapping/maps/example_arena.png # Selected map(s)
```

Results are saved to [`benchmarks/BENCHMARK_SUMMARY.md`](benchmarks/BENCHMARK_SUMMARY.md).

