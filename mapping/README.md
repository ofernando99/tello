# Occupancy Grid Mapping

Converts 2D arena and high-resolution maze maps (`.json`, `.txt`, `.png`/`.jpg`, `.py`) into a binary 2D NumPy `occupancy_grid` (`0` = free, `1` = obstacle) with `start` and `destination` `(row, col)` coordinates.

## Usage

```bash
# Run format & loader tests (low-res, high-res maze, and min_grid_size checks)
python mapping/test_mapping.py
python mapping/test_mapping.py mapping/maps/example_high_res_maze.jpg

# Generate a random solvable sparse arena map (.json, .txt, .png, .jpg, or .py)
python mapping/generate_map.py --style arena --obstacles 4 --out mapping/maps/random_arena.png

# Generate a high-resolution labyrinthine maze map (like example_high_res_maze.jpg)
python mapping/generate_map.py --style maze --out mapping/maps/random_high_res_maze.jpg
python mapping/generate_map.py --style maze --seed 42 --format py --out mapping/maps/random_high_res_maze.py
```

## Loading the Occupancy Grid

Use `load_map()` to load any map file directly as `(occupancy_grid, start, destination)`:

```python
from mapping.mapping import load_map

# Standard or high-resolution image map (guarantees at least a 20x20 grid by default)
occupancy_grid, start, destination = load_map("mapping/maps/example_high_res_maze.jpg")

# Optional: grow obstacles by a safety margin in cm (use include_walls=False to skip arena edges)
gmap = load_map("mapping/maps/example_arena.png", cell_cm=20)
gmap = gmap.inflate(10, include_walls=False)
occupancy_grid, start, destination = gmap

# Customize resolution or minimum grid dimensions for image maps
gmap = load_map(
    "mapping/maps/example_high_res_maze.jpg",
    cell_cm=10,                        # Finer cell resolution (e.g. 41x50 grid)
    arena_width_cm=500,                # Physical arena width in cm (default: 500)
    min_grid_size=20,                  # Clamps cell_cm so both rows and cols are >= min_grid_size
    obstacle_ratio_threshold=0.35,     # Fraction of dark pixels required to mark a cell occupied
)
```

## Map Generator Options (`mapping/generate_map.py`)

| Option | Default | Description |
| :--- | :--- | :--- |
| `--style {arena,maze}` | `arena` | `arena` places random sparse obstacles; `maze` carves a complex multi-branch high-res labyrinth with open start/goal pads and long winding routes. |
| `--out PATH` | `mapping/maps/random_arena.json` | Output file path (`.json`, `.txt`, `.png`, `.jpg`, `.py`). |
| `--format {json,txt,png,jpg,py}` | Auto from `--out` | Explicit output format override. |
| `--width` / `--height` | `500` / `400` | Physical arena width and height in cm. |
| `--cell-cm` | `20` | Grid cell size in cm. |
| `--obstacles` | `4` | Number of obstacles (`arena` style) or extra cycle branches/loops (`maze` style). |
| `--margin-cm` | `30.0` (`arena`) / `0.0` (`maze`) | Obstacle safety inflation margin in cm. |
| `--seed` | `None` | Random seed for reproducible map generation. |

