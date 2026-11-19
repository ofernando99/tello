"""
Author: Jessie Wang

Generate random, guaranteed-solvable arena or high-resolution maze maps for testing and benchmarking.
Usage:
    python mapping/generate_map.py
    python mapping/generate_map.py --obstacles 6 --out mapping/maps/random_arena.json
    python mapping/generate_map.py --obstacles 5 --format txt --out mapping/maps/random_arena.txt
    python mapping/generate_map.py --style maze --out mapping/maps/random_high_res_maze.jpg
    python mapping/generate_map.py --style maze --seed 42 --format py --out mapping/maps/random_high_res_maze.py
"""
import argparse
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass

sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mapping.mapping import GridMap, load_json  # noqa: E402
from path_planning.path_planner import astar  # noqa: E402

DEFAULT_SAFETY_MARGIN_CM = 30.0


@dataclass
class _PlanCheckResult:
    found: bool
    path: list
    cost_cells: float


def _check_solvable(gmap, diagonal: bool = False) -> _PlanCheckResult:
    """Verify map solvability using path_planner.astar."""
    try:
        path, stats = astar(
            gmap.occupancy_grid,
            gmap.start,
            gmap.destination,
            allow_diagonal=diagonal,
        )
        return _PlanCheckResult(found=True, path=path, cost_cells=float(stats["cost"]))
    except Exception:
        return _PlanCheckResult(found=False, path=[], cost_cells=0.0)


def _grid_to_geometric_obstacles(grid: list[list[int]], cell_cm: float) -> list[dict]:
    """Convert a binary 2D occupancy grid into merged horizontal rectangle obstacles in cm."""
    obstacles = []
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] == 1:
                c_start = c
                while c < cols and grid[r][c] == 1:
                    c += 1
                obstacles.append({
                    "type": "rect",
                    "x": int(c_start * cell_cm),
                    "y": int(r * cell_cm),
                    "w": int((c - c_start) * cell_cm),
                    "h": int(cell_cm),
                })
            else:
                c += 1
    return obstacles


def generate_maze_map(
    width_cm: int = 500,
    height_cm: int = 400,
    cell_cm: int = 20,
    extra_cycles: int = 6,
    seed: int | None = None,
    max_tries: int = 50,
):
    """Generate a high-resolution labyrinthine maze map matching the visual and route
    complexity of example_high_res_maze.jpg (multi-branch corridors, top-left start pad,
    bottom-right goal pad, optional circular obstacle accent, and long winding A* route)."""
    if seed is not None:
        random.seed(seed)

    cols = max(21, int(width_cm // cell_cm))
    rows = max(20, int(height_cm // cell_cm))
    width_cm = cols * cell_cm
    height_cm = rows * cell_cm
    min_path_len = int((rows + cols) * 1.15)

    for _ in range(max_tries):
        # Start with all walls (1)
        grid = [[1] * cols for _ in range(rows)]

        # Carve a randomized depth-first spanning tree on odd cell coordinates (1..rows-2, 1..cols-2)
        max_odd_r = rows - 2 if (rows - 2) % 2 == 1 else rows - 3
        max_odd_c = cols - 2 if (cols - 2) % 2 == 1 else cols - 3

        start_node = (1, 1)
        grid[start_node[0]][start_node[1]] = 0
        stack = [start_node]
        visited = {start_node}

        while stack:
            cr, cc = stack[-1]
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = cr + dr, cc + dc
                if 1 <= nr <= max_odd_r and 1 <= nc <= max_odd_c and (nr, nc) not in visited:
                    neighbors.append((nr, nc, cr + dr // 2, cc + dc // 2))
            if neighbors:
                nr, nc, wr, wc = random.choice(neighbors)
                grid[wr][wc] = 0
                grid[nr][nc] = 0
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Carve open start room in top-left (3x5 free pad) and goal room in bottom-right (3x5 free pad)
        for r in range(0, min(3, rows)):
            for c in range(0, min(5, cols)):
                grid[r][c] = 0
        # Ensure connection from top-left pad into the maze interior
        grid[2][4] = 0
        grid[3][4] = 0
        if max_odd_r >= 3:
            grid[3][3] = 0

        for r in range(max(0, rows - 3), rows):
            for c in range(max(0, cols - 5), cols):
                grid[r][c] = 0
        # Connect bottom-right pad to the nearest carved maze corridor
        grid[rows - 3][cols - 5] = 0
        grid[rows - 4][cols - 5] = 0
        grid[max_odd_r][max_odd_c] = 0
        for r in range(max_odd_r, rows - 2):
            grid[r][max_odd_c] = 0
        for c in range(max_odd_c, cols - 4):
            grid[rows - 3][c] = 0

        # Knock out a controlled number of internal walls to create alternate routes/loops
        internal_walls = []
        for r in range(2, max_odd_r):
            for c in range(2, max_odd_c):
                if grid[r][c] == 1:
                    horiz_clear = grid[r][c - 1] == 0 and grid[r][c + 1] == 0 and grid[r - 1][c] == 1 and grid[r + 1][c] == 1
                    vert_clear = grid[r - 1][c] == 0 and grid[r + 1][c] == 0 and grid[r][c - 1] == 1 and grid[r][c + 1] == 1
                    if horiz_clear or vert_clear:
                        internal_walls.append((r, c))

        random.shuffle(internal_walls)
        for wr, wc in internal_walls[: max(0, extra_cycles)]:
            grid[wr][wc] = 0

        start_cell = (1, 1)
        goal_cell = (rows - 2, cols - 2)
        grid[start_cell[0]][start_cell[1]] = 0
        grid[goal_cell[0]][goal_cell[1]] = 0

        gmap = GridMap(grid, float(cell_cm), start_cell, goal_cell)
        res = _check_solvable(gmap, diagonal=False)
        if not res.found or len(res.path) < min_path_len:
            continue

        start_cm = [int((start_cell[1] + 0.5) * cell_cm), int((start_cell[0] + 0.5) * cell_cm)]
        goal_cm = [int((goal_cell[1] + 0.5) * cell_cm), int((goal_cell[0] + 0.5) * cell_cm)]
        obstacles = _grid_to_geometric_obstacles(grid, cell_cm)

        # Place a signature circular obstacle over an interior 2x2 wall block (like example_high_res_maze.jpg)
        circle_candidates = []
        for r in range(4, rows - 5):
            for c in range(4, cols - 5):
                if grid[r][c] == 1 and grid[r + 1][c] == 1 and grid[r][c + 1] == 1 and grid[r + 1][c + 1] == 1:
                    circle_candidates.append((r + 1, c + 1))
        if circle_candidates:
            cr, cc = random.choice(circle_candidates)
            obstacles.append({
                "type": "circle",
                "x": int(cc * cell_cm),
                "y": int(cr * cell_cm),
                "r": int(cell_cm * 0.9),
            })

        spec = {
            "_comment": "Randomly generated high-resolution solvable maze map.",
            "style": "maze",
            "arena_cm": [int(width_cm), int(height_cm)],
            "cell_cm": int(cell_cm),
            "origin": "top-left",
            "start": start_cm,
            "goal": goal_cm,
            "start_heading": "right",
            "obstacles": obstacles,
        }
        return spec, gmap, res

    raise RuntimeError(f"Could not generate a complex solvable maze after {max_tries} attempts.")


def generate_valid_map(width_cm=500, height_cm=400, cell_cm=20, n_obs=4,
                       margin_cm=DEFAULT_SAFETY_MARGIN_CM, seed=None, max_tries=50):
    if seed is not None:
        random.seed(seed)

    cols = int(width_cm // cell_cm)
    rows = int(height_cm // cell_cm)

    for _ in range(max_tries):
        # Place start near bottom-left and goal near top-right
        start_c = random.randint(2, min(5, cols - 3))
        start_r = random.randint(max(rows - 6, 2), rows - 3)
        goal_c = random.randint(max(cols - 6, 2), cols - 3)
        goal_r = random.randint(2, min(5, rows - 3))

        start_cm = [int((start_c + 0.5) * cell_cm), int((start_r + 0.5) * cell_cm)]
        goal_cm = [int((goal_c + 0.5) * cell_cm), int((goal_r + 0.5) * cell_cm)]

        obstacles = []
        for _ in range(n_obs):
            kind = random.choice(["rect", "circle"])
            if kind == "rect":
                w = random.choice([40, 60, 80])
                h = random.choice([40, 60, 80, 120])
                x = random.randint(60, max(60, int(width_cm - w - 60)))
                y = random.randint(60, max(60, int(height_cm - h - 60)))
                obstacles.append({"type": "rect", "x": x, "y": y, "w": w, "h": h})
            else:
                r = random.choice([20, 30, 40])
                x = random.randint(60, max(60, int(width_cm - r - 60)))
                y = random.randint(60, max(60, int(height_cm - r - 60)))
                obstacles.append({"type": "circle", "x": x, "y": y, "r": r})

        spec = {
            "_comment": "Randomly generated solvable arena map.",
            "arena_cm": [int(width_cm), int(height_cm)],
            "cell_cm": int(cell_cm),
            "origin": "top-left",
            "start": start_cm,
            "goal": goal_cm,
            "start_heading": "up",
            "obstacles": obstacles,
        }

        # Build and check feasibility
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                tmp_path = tmp.name
                json.dump(spec, tmp)
            try:
                raw = load_json(tmp_path)
                gmap = raw.inflate(margin_cm)
                res = _check_solvable(gmap, diagonal=False)
                if res.found:
                    return spec, gmap, res
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception:
            continue

    raise RuntimeError(f"Could not generate a solvable map after {max_tries} attempts. Try fewer obstacles.")


def export_txt(gmap, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("; ASCII arena map generated by mapping/generate_map.py\n")
        f.write(f"; cell_cm={gmap.cell_cm}\n")
        f.write(gmap.to_ascii())
        f.write("\n")


def export_py(gmap, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("import numpy as np\n\n")
        f.write(gmap.to_python_code())
        f.write("\n")


def export_png(spec, out_path, scale: float | None = None):
    """Render a synthetic scanned 2D drawing compatible with mapping.load_image():
    - White background
    - Dark ink obstacles (gray < 100, saturation < 80)
    - Green blob for Start, Red blob for Goal
    """
    import cv2
    import numpy as np

    W, H = spec["arena_cm"]
    if scale is None:
        scale = 1152.0 / W if spec.get("style") == "maze" else 2.0
    flip = spec.get("origin", "top-left") == "bottom-left"
    img = np.full((int(round(H * scale)), int(round(W * scale)), 3), 255, dtype=np.uint8)

    def fy(y, h=0.0):
        return H - y - h if flip else y

    for ob in spec.get("obstacles", []):
        kind = ob.get("type", "rect")
        if kind == "rect":
            ox, oy = ob["x"], fy(ob["y"], ob["h"])
            x0, y0 = int(round(ox * scale)), int(round(oy * scale))
            x1, y1 = int(round((ox + ob["w"]) * scale)), int(round((oy + ob["h"]) * scale))
            cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), -1)
        elif kind == "circle":
            cx, cy = ob["x"], fy(ob["y"])
            cv2.circle(
                img,
                (int(round(cx * scale)), int(round(cy * scale))),
                int(round(ob["r"] * scale)),
                (20, 20, 20),
                -1,
            )

    sx, sy = spec["start"][0], fy(spec["start"][1])
    gx, gy = spec["goal"][0], fy(spec["goal"][1])
    radius = max(8, int(round(spec.get("cell_cm", 20) * 0.5 * scale)))
    cv2.circle(img, (int(round(sx * scale)), int(round(sy * scale))), radius, (0, 200, 0), -1)   # Green start
    cv2.circle(img, (int(round(gx * scale)), int(round(gy * scale))), radius, (0, 0, 220), -1)   # Red goal

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-json", default=None, help="convert an existing JSON map to --format (txt/png/jpg/py)")
    p.add_argument(
        "--style",
        choices=["arena", "maze"],
        default="arena",
        help="map generation style: 'arena' (random sparse obstacles) or 'maze' (high-res multi-branch labyrinth)",
    )
    p.add_argument("--width", type=int, default=500, help="arena width in cm (default: 500)")
    p.add_argument("--height", type=int, default=400, help="arena height in cm (default: 400)")
    p.add_argument("--cell-cm", type=int, default=20, help="grid cell size in cm (default: 20)")
    p.add_argument("--obstacles", type=int, default=4, help="number of obstacles for arena style, or extra cycle branches for maze style (default: 4)")
    p.add_argument("--margin-cm", type=float, default=None, help="safety margin in cm (default: 30.0 for arena, 0.0 for maze)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--out", default="mapping/maps/random_arena.json", help="output file path (.json, .txt, .png, .jpg, .py)")
    p.add_argument("--format", choices=["json", "txt", "png", "jpg", "py"], default=None, help="export format (auto-detected from --out if omitted)")
    args = p.parse_args()

    margin_cm = args.margin_cm
    if margin_cm is None:
        margin_cm = 0.0 if args.style == "maze" else DEFAULT_SAFETY_MARGIN_CM

    if args.from_json:
        with open(args.from_json) as f:
            spec = json.load(f)
        raw = load_json(args.from_json)
        gmap = raw.inflate(margin_cm) if margin_cm > 0 else raw
        res = _check_solvable(gmap, diagonal=False)
    elif args.style == "maze":
        spec, gmap, res = generate_maze_map(
            width_cm=args.width,
            height_cm=args.height,
            cell_cm=args.cell_cm,
            extra_cycles=args.obstacles,
            seed=args.seed,
        )
        if margin_cm > 0:
            gmap = gmap.inflate(margin_cm)
            res = _check_solvable(gmap, diagonal=False)
    else:
        spec, gmap, res = generate_valid_map(
            width_cm=args.width, height_cm=args.height, cell_cm=args.cell_cm,
            n_obs=args.obstacles, margin_cm=margin_cm, seed=args.seed
        )

    out_path = args.out
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    fmt = args.format or (ext if ext in ("json", "txt", "png", "jpg", "py") else "json")
    if fmt == "txt":
        if not out_path.endswith(".txt"):
            out_path = os.path.splitext(out_path)[0] + ".txt"
        export_txt(gmap, out_path)
    elif fmt in ("png", "jpg"):
        if not (out_path.endswith(".png") or out_path.endswith(".jpg") or out_path.endswith(".jpeg")):
            out_path = os.path.splitext(out_path)[0] + f".{fmt}"
        export_png(spec, out_path)
    elif fmt == "py":
        if not out_path.endswith(".py"):
            out_path = os.path.splitext(out_path)[0] + ".py"
        export_py(gmap, out_path)
    else:
        if not out_path.endswith(".json"):
            out_path = os.path.splitext(out_path)[0] + ".json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(spec, f, indent=2)

    print(f"\n[Generated Solvable Map ({args.style}): {out_path}]")
    print(gmap.to_ascii(res.path))
    print(f"\nA* path length: {len(res.path)} cells, cost: {res.cost_cells:.1f}")
    print(f"Start: {spec['start']}, Goal: {spec['goal']}, Obstacles: {len(spec['obstacles'])}")


if __name__ == "__main__":
    main()
