"""
Author: Jessie Wang

Test runner for mapping/mapping.py across all supported map formats
(.json, .txt, .png/.jpg, .py, and in-memory NumPy arrays).

Usage:
    python mapping/test_mapping.py
    python mapping/test_mapping.py mapping/maps/example_arena.json mapping/maps/example_arena.py
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mapping.mapping import GridMap, load_map  # noqa: E402


def run_unit_checks() -> None:
    """Verify all input loaders, high/low resolution maps, min_grid_size enforcement, and NumPy export methods."""
    # 1. Low-res geometric JSON vs Scanned PNG consistency check
    g_json = load_map("mapping/maps/example_arena.json")
    g_img = load_map("mapping/maps/example_arena.png", cell_cm=20, arena_width_cm=500)
    assert g_json.grid == g_img.grid, "Image grid should match geometric JSON grid"
    assert g_json.start == g_img.start and g_json.destination == g_img.destination
    assert g_img.rows == 20 and g_img.cols == 25

    # 2. ASCII .txt loader check (low-res)
    g_txt = load_map("mapping/maps/example_arena.txt", cell_cm=20)
    assert g_txt.rows == 15 and g_txt.cols == 25
    assert g_txt.start == (12, 3) and g_txt.destination == (2, 19)

    # Coordinate-based .txt map (bottom-left y axis)
    g_coordinates = load_map("mapping/maps/project_2_layout_test.txt", cell_cm=20)
    assert g_coordinates.rows == 30 and g_coordinates.cols == 55
    assert g_coordinates.start == (24, 20)
    assert g_coordinates.destination == (19, 35)
    assert g_coordinates.occupancy_grid[25, 27] == 0
    assert g_coordinates.occupancy_grid[24, 27] == 1
    assert g_coordinates.occupancy_grid[14, 27] == 1

    # 3. Python .py NumPy grid loader check (low-res & high-res maze)
    occ, start, dest = load_map("mapping/maps/example_arena.py")
    assert occ.shape == (10, 12) and start == (0, 0) and dest == (9, 11)

    occ_hr_py, start_hr_py, dest_hr_py = load_map("mapping/maps/example_high_res_maze.py")
    assert occ_hr_py.shape == (20, 25) and start_hr_py == (1, 1) and dest_hr_py == (18, 23)
    assert occ_hr_py[start_hr_py] == 0 and occ_hr_py[dest_hr_py] == 0

    # 4. High-res maze image (.jpg) check (default >= 20x20 and finer resolution)
    g_hr_jpg = load_map("mapping/maps/example_high_res_maze.jpg")
    assert g_hr_jpg.rows >= 20 and g_hr_jpg.cols >= 20, (
        f"Expected at least 20x20 grid for high-res maze, got {g_hr_jpg.rows}x{g_hr_jpg.cols}"
    )
    assert g_hr_jpg.occupancy_grid.shape == (21, 25)
    assert g_hr_jpg.occupancy_grid[g_hr_jpg.start] == 0
    assert g_hr_jpg.occupancy_grid[g_hr_jpg.destination] == 0

    g_hr_fine = load_map("mapping/maps/example_high_res_maze.jpg", cell_cm=10, arena_width_cm=500)
    assert g_hr_fine.rows == 41 and g_hr_fine.cols == 50
    assert g_hr_fine.occupancy_grid[g_hr_fine.start] == 0
    assert g_hr_fine.occupancy_grid[g_hr_fine.destination] == 0

    # 5. min_grid_size enforcement check (e.g. cell_cm=100, arena_width_cm=500)
    # Without clamping, 500x400cm @ 100cm/cell would yield 4x5 (< 20x20); min_grid_size=20
    # must clamp cell size so both rows and cols are >= 20.
    g_clamped_arena = load_map("mapping/maps/example_arena.png", cell_cm=100, arena_width_cm=500)
    assert g_clamped_arena.rows >= 20 and g_clamped_arena.cols >= 20, (
        f"min_grid_size not respected on example_arena.png: got {g_clamped_arena.rows}x{g_clamped_arena.cols}"
    )
    assert g_clamped_arena.rows == 20 and g_clamped_arena.cols == 25
    assert np.isclose(g_clamped_arena.cell_cm, 20.0), f"Expected clamped cell_cm=20.0, got {g_clamped_arena.cell_cm}"

    g_clamped_maze = load_map("mapping/maps/example_high_res_maze.jpg", cell_cm=100, arena_width_cm=500)
    assert g_clamped_maze.rows >= 20 and g_clamped_maze.cols >= 20, (
        f"min_grid_size not respected on example_high_res_maze.jpg: got {g_clamped_maze.rows}x{g_clamped_maze.cols}"
    )
    assert g_clamped_maze.rows == 20 and g_clamped_maze.cols == 25
    assert np.isclose(g_clamped_maze.cell_cm, 20.0, atol=0.1)

    # Also test portrait orientation (width < height) to ensure BOTH rows and cols respect min_grid_size
    g_custom_min = load_map(
        "mapping/maps/example_high_res_maze.jpg",
        cell_cm=100,
        arena_width_cm=500,
        min_grid_size=25,
    )
    assert g_custom_min.rows >= 25 and g_custom_min.cols >= 25

    # 6. In-memory NumPy array & discrete JSON check
    arr = np.array([[0, 1, 0], [0, 0, 0]])
    g_arr = GridMap.from_array(arr, start=(0, 0), destination=(1, 2))
    assert np.array_equal(g_arr.occupancy_grid, arr)

    with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
        json.dump({"occupancy_grid": arr.tolist(), "start": [0, 0], "destination": [1, 2]}, tmp)
        tmp.flush()
        g_disc_json = load_map(tmp.name)
        assert np.array_equal(g_disc_json.occupancy_grid, arr)

    print("[PASS] All mapping input formats (low-res, high-res, min_grid_size clamping, and NumPy exports) verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "maps",
        nargs="*",
        default=[
            "mapping/maps/example_arena.json",
            "mapping/maps/example_arena.txt",
            "mapping/maps/example_arena.png",
            "mapping/maps/example_arena.py",
            "mapping/maps/example_high_res_maze.jpg",
            "mapping/maps/example_high_res_maze.py",
        ],
        help="Path(s) to map file(s) (.json, .txt, .png/.jpg, .py)",
    )
    parser.add_argument("--cell-cm", type=float, default=20.0, help="Grid cell size in cm (for .txt/.png/.py)")
    parser.add_argument("--arena-width-cm", type=float, default=500.0, help="Physical arena width in cm (for images)")
    parser.add_argument("--margin-cm", type=float, default=0.0, help="Optional obstacle inflation margin in cm")
    args = parser.parse_args()

    for map_path in args.maps:
        ext = os.path.splitext(map_path)[1].lower()
        kwargs = {}
        if ext in (".txt", ".py"):
            kwargs = {"cell_cm": args.cell_cm}
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            kwargs = {"cell_cm": args.cell_cm, "arena_width_cm": args.arena_width_cm}

        gmap = load_map(map_path, **kwargs)
        if args.margin_cm > 0:
            gmap = gmap.inflate(args.margin_cm)

        print(f"\n=== {map_path} ({gmap.rows}x{gmap.cols} cells @ {gmap.cell_cm:.0f} cm) ===")
        print(gmap.to_ascii())
        print()
        print(gmap.to_python_code())

    print()
    run_unit_checks()


if __name__ == "__main__":
    main()
