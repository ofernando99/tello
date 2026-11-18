# Occupancy Grid Mapping

Converts 2D arena maps (`.json`, `.txt`, `.png`/`.jpg`, `.py`) into a binary 2D NumPy `occupancy_grid` (`0` = free, `1` = obstacle) with `start` and `destination` `(row, col)` coordinates.

## Usage

```bash
# Run format & loader tests (or inspect specific map files)
python mapping/test_mapping.py
python mapping/test_mapping.py mapping/maps/example_arena.py

# Generate a random solvable map (.json, .txt, or .png)
python mapping/generate_map.py --obstacles 4 --out mapping/maps/random_arena.png
```

## Loading the Occupancy Grid

Use `load_map()` to load any map file directly as `(occupancy_grid, start, destination)`:

```python
from mapping.mapping import load_map

occupancy_grid, start, destination = load_map("mapping/maps/example_arena.png")
```
