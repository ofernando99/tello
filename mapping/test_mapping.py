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
    """Verify all input loaders and NumPy export methods."""
    # 1. Geometric JSON vs Scanned PNG consistency check
    g_json = load_map("mapping/maps/example_arena.json")
    g_img = load_map("mapping/maps/example_arena.png", cell_cm=20, arena_width_cm=500)
    assert g_json.grid == g_img.grid, "Image grid should match geometric JSON grid"
    assert g_json.start == g_img.start and g_json.destination == g_img.destination

    # 2. ASCII .txt loader check
    g_txt = load_map("mapping/maps/example_arena.txt", cell_cm=20, heading="up")
    assert g_txt.rows == 15 and g_txt.cols == 25
    assert g_txt.start == (12, 3) and g_txt.destination == (2, 19)

    # 3. Python .py NumPy grid loader check
    occ, start, dest = load_map("mapping/maps/example_arena.py")
    assert occ.shape == (10, 12) and start == (0, 0) and dest == (9, 11)

    # 4. In-memory NumPy array & discrete JSON check
    arr = np.array([[0, 1, 0], [0, 0, 0]])
    g_arr = GridMap.from_array(arr, start=(0, 0), destination=(1, 2))
    assert np.array_equal(g_arr.occupancy_grid, arr)

    with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
        json.dump({"occupancy_grid": arr.tolist(), "start": [0, 0], "destination": [1, 2]}, tmp)
        tmp.flush()
        g_disc_json = load_map(tmp.name)
        assert np.array_equal(g_disc_json.occupancy_grid, arr)

    print("[PASS] All 6 mapping input formats and NumPy exports verified.")


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
        ],
        help="Path(s) to map file(s) (.json, .txt, .png/.jpg, .py)",
    )
    parser.add_argument("--cell-cm", type=float, default=20.0, help="Grid cell size in cm (for .txt/.png/.py)")
    parser.add_argument("--arena-width-cm", type=float, default=500.0, help="Physical arena width in cm (for images)")
    parser.add_argument("--heading", default="up", help="Initial drone heading (for .txt/.png/.py)")
    parser.add_argument("--margin-cm", type=float, default=0.0, help="Optional obstacle inflation margin in cm")
    args = parser.parse_args()

    for map_path in args.maps:
        ext = os.path.splitext(map_path)[1].lower()
        kwargs = {}
        if ext in (".txt", ".py"):
            kwargs = {"cell_cm": args.cell_cm, "heading": args.heading}
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            kwargs = {"cell_cm": args.cell_cm, "arena_width_cm": args.arena_width_cm, "heading": args.heading}

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
