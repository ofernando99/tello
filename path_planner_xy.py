import heapq
import math
import time
import numpy as np
#from environment_mapping import generate_occupancy_grid

#==================================================================#
#some parts were modified to map occupancy grid into x y corrdinates
#==================================================================#

def validate_grid(grid: np.ndarray, start: tuple, goal: tuple) -> None:
    """Verify the occupancy grid and start/destination coordinates are valid."""
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        raise ValueError("occupancy_grid must be a 2D NumPy array.")
    rows, cols = grid.shape
    for name, (x, y) in [("start", start), ("destination", goal)]:
        #=========================================================#
        r = rows - 1 - y
        c = x
        #=========================================================#
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} {(r, c)} is out of bounds for grid shape {grid.shape}.")
        if grid[r, c] != 0:
            raise ValueError(f"{name} {(r, c)} lies on an obstacle.")


def get_neighbors(grid: np.ndarray, node: tuple, allow_diagonal: bool = True) -> list:
    """Return valid adjacent cells and step costs while preventing diagonal corner-cutting."""
    rows, cols = grid.shape
    #=========================================================#
    x, y = node
    r = rows - 1 - y
    c = x
    #=========================================================#
    cardinal = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    diagonal = [
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    neighbors = []
    for dr, dc, cost in cardinal:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            #=========================================================#
            neighbors.append(((nc, rows - 1 - nr), cost))
            #=========================================================#
    if allow_diagonal:
        for dr, dc, cost in diagonal:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                if grid[r + dr, c] == 0 and grid[r, c + dc] == 0:
                    #=========================================================#
                    neighbors.append(((nc, rows - 1 - nr), cost))
                    #=========================================================#
    return neighbors


def _reconstruct(came_from: dict, current: tuple) -> list:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid: np.ndarray, start: tuple, goal: tuple, allow_diagonal: bool = True) -> tuple:
    """Compute the optimal collision-free grid path using A* with Euclidean heuristic."""
    validate_grid(grid, start, goal)
    t0 = time.perf_counter()

    def heuristic(a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) if allow_diagonal else abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    expanded = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            stats = {
                "cost": round(current_g, 2),
                "expanded": expanded,
                "time_ms": round(elapsed_ms, 3),
            }
            return _reconstruct(came_from, current), stats

        if current_g > g_score.get(current, float("inf")):
            continue
        expanded += 1

        for nxt, step_cost in get_neighbors(grid, current, allow_diagonal):
            tentative_g = current_g + step_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, nxt))

    raise RuntimeError("No collision-free path found between start and destination.")


def _has_line_of_sight(grid: np.ndarray, p1: tuple, p2: tuple, margin_cells: float = 0.85) -> bool:
    """Check whether a straight line segment clears all obstacle centers by at least margin_cells."""
    #=========================================================#
    obstacles = np.argwhere(grid == 1)
    obstacles = np.column_stack((obstacles[:, 1], grid.shape[0] - 1 - obstacles[:, 0]))
    #=========================================================#
    if len(obstacles) == 0:
        return True
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(int(dist * 12), 1)
    for i in range(steps + 1):
        t = i / steps
        r = p1[0] + (p2[0] - p1[0]) * t
        c = p1[1] + (p2[1] - p1[1]) * t
        if np.any(np.hypot(obstacles[:, 0] - r, obstacles[:, 1] - c) < margin_cells):
            return False
    return True


def smooth_waypoints_los(grid: np.ndarray, path: list, margin_cells: float = 0.85) -> list:
    """Collapse stair-step grid cells into minimal turning waypoints for smooth drone flight."""
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    anchor_idx = 0
    while anchor_idx < len(path) - 1:
        furthest_idx = anchor_idx + 1
        for candidate_idx in range(len(path) - 1, anchor_idx, -1):
            if _has_line_of_sight(grid, path[anchor_idx], path[candidate_idx], margin_cells):
                furthest_idx = candidate_idx
                break
        smoothed.append(path[furthest_idx])
        anchor_idx = furthest_idx

    return smoothed


def plan_path(
    occupancy_grid: np.ndarray,
    start: tuple,
    destination: tuple,
    cell_size_cm: float = 30.0, #change if want more precision 
    drone_clearance_cm: float = 18.0, #adjust if drone drifts 
    smooth: bool = True,
) -> list:
    """
    Main pipeline function matching the architecture
    Takes (occupancy_grid, start, destination) and returns an ordered list of (row, col) waypoints.
    """
    raw_path, _ = astar(occupancy_grid, start, destination, allow_diagonal=True)
    if not smooth:
        return raw_path

    margin_cells = max(0.85, drone_clearance_cm / cell_size_cm)
    return smooth_waypoints_los(occupancy_grid, raw_path, margin_cells=margin_cells)

if __name__ == "__main__":
    # Local integration test using the 10x12 map from the architecture diagram
    occupancy_grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    start = (0, 0) #(0x,0y)
    destination = (11, 9) #(11x,9y)

    path = plan_path(occupancy_grid, start, destination, cell_size_cm=20.0)
    print("Planned Waypoints:", path)
