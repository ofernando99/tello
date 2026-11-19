import heapq
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def validate_grid(grid: np.ndarray, start: tuple, goal: tuple) -> None:
    if grid.ndim != 2:
        raise ValueError("occupancy_grid must be a 2D NumPy array.")
    rows, cols = grid.shape
    for name, (r, c) in [("start", start), ("destination", goal)]:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} {(r, c)} is out of bounds for shape {grid.shape}.")
        if grid[r, c] != 0:
            raise ValueError(f"{name} {(r, c)} lies on an obstacle.")


def get_neighbors(grid: np.ndarray, node: tuple, allow_diagonal: bool = True):
    rows, cols = grid.shape
    r, c = node
    cardinal = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    diagonal = [(-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    neighbors = []
    for dr, dc, cost in cardinal:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append(((nr, nc), cost))

    if allow_diagonal:
        for dr, dc, cost in diagonal:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                # Prevent corner-cutting past adjacent obstacle cells
                if grid[r + dr, c] == 0 and grid[r, c + dc] == 0:
                    neighbors.append(((nr, nc), cost))
    return neighbors


def _reconstruct(came_from: dict, current: tuple) -> list:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid: np.ndarray, start: tuple, goal: tuple, allow_diagonal: bool = True):
    validate_grid(grid, start, goal)
    t0 = time.perf_counter()

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) if allow_diagonal else abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    expanded = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return _reconstruct(came_from, current), {"cost": round(current_g, 2), "expanded": expanded, "time_ms": round(elapsed_ms, 3)}

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

    raise RuntimeError("No collision-free path found.")


def dijkstra(grid: np.ndarray, start: tuple, goal: tuple, allow_diagonal: bool = True):
    validate_grid(grid, start, goal)
    t0 = time.perf_counter()

    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    expanded = 0

    while open_heap:
        current_g, current = heapq.heappop(open_heap)
        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return _reconstruct(came_from, current), {"cost": round(current_g, 2), "expanded": expanded, "time_ms": round(elapsed_ms, 3)}

        if current_g > g_score.get(current, float("inf")):
            continue
        expanded += 1

        for nxt, step_cost in get_neighbors(grid, current, allow_diagonal):
            tentative_g = current_g + step_cost
            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                heapq.heappush(open_heap, (tentative_g, nxt))

    raise RuntimeError("No collision-free path found.")


def rrt_grid(grid: np.ndarray, start: tuple, goal: tuple, max_iter: int = 3000, seed: int = 42):
    validate_grid(grid, start, goal)
    random.seed(seed)
    t0 = time.perf_counter()
    rows, cols = grid.shape
    parent = {start: None}
    nodes = [start]

    def line_free(a, b):
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2
        if steps == 0:
            return True
        for i in range(steps + 1):
            r = round(a[0] + (b[0] - a[0]) * (i / steps))
            c = round(a[1] + (b[1] - a[1]) * (i / steps))
            if not (0 <= r < rows and 0 <= c < cols) or grid[r, c] != 0:
                return False
        return True

    for _ in range(max_iter):
        sample = goal if random.random() < 0.15 else (random.randrange(rows), random.randrange(cols))
        nearest = min(nodes, key=lambda n: math.hypot(n[0] - sample[0], n[1] - sample[1]))
        dist = math.hypot(sample[0] - nearest[0], sample[1] - nearest[1])
        if dist == 0:
            continue
        step = min(2.0, dist)
        new_node = (
            round(nearest[0] + (sample[0] - nearest[0]) * (step / dist)),
            round(nearest[1] + (sample[1] - nearest[1]) * (step / dist)),
        )
        if new_node not in parent and line_free(nearest, new_node):
            parent[new_node] = nearest
            nodes.append(new_node)
            if math.hypot(new_node[0] - goal[0], new_node[1] - goal[1]) <= 1.5 and line_free(new_node, goal):
                if goal not in parent:
                    parent[goal] = new_node
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                cost = sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1))
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return path, {"cost": round(cost, 2), "expanded": len(nodes), "time_ms": round(elapsed_ms, 3)}

    raise RuntimeError("RRT failed to reach destination within max_iter.")

def _has_line_of_sight(grid: np.ndarray, p1: tuple, p2: tuple, margin_cells: float = 0.85) -> bool:
    obstacles = np.argwhere(grid == 1)
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


def theta_star(grid: np.ndarray, start: tuple, goal: tuple):
    validate_grid(grid, start, goal)
    t0 = time.perf_counter()

    def h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = [(h(start, goal), 0.0, start)]
    parent = {start: start}
    g_score = {start: 0.0}
    closed = set()
    expanded = 0

    while open_heap:
        _, cur_g, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        closed.add(s)

        if s == goal:
            path = [s]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return path, {"cost": round(cur_g, 2), "expanded": expanded, "time_ms": round(elapsed_ms, 3)}

        expanded += 1
        for nxt, step_cost in get_neighbors(grid, s, allow_diagonal=True):
            if nxt in closed:
                continue
            p_s = parent[s]
            # Path 2: direct line-of-sight from parent(s) to neighbor
            if _has_line_of_sight(grid, p_s, nxt):
                new_g = g_score[p_s] + h(p_s, nxt)
                new_parent = p_s
            else:
                # Path 1: standard grid step through s
                new_g = g_score[s] + step_cost
                new_parent = s

            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                parent[nxt] = new_parent
                heapq.heappush(open_heap, (new_g + h(nxt, goal), new_g, nxt))

    raise RuntimeError("No collision-free path found.")


def rrt_star(grid: np.ndarray, start: tuple, goal: tuple, max_iter: int = 1500, rewire_radius: float = 3.5, seed: int = 42):
    """Asymptotically optimal RRT* with neighborhood rewiring."""
    validate_grid(grid, start, goal)
    random.seed(seed)
    t0 = time.perf_counter()
    rows, cols = grid.shape

    parent = {start: None}
    cost_to_node = {start: 0.0}
    nodes = [start]

    for _ in range(max_iter):
        sample = goal if random.random() < 0.15 else (random.randrange(rows), random.randrange(cols))
        nearest = min(nodes, key=lambda n: math.hypot(n[0] - sample[0], n[1] - sample[1]))
        dist = math.hypot(sample[0] - nearest[0], sample[1] - nearest[1])
        if dist == 0:
            continue

        step = min(2.0, dist)
        new_node = (
            round(nearest[0] + (sample[0] - nearest[0]) * (step / dist)),
            round(nearest[1] + (sample[1] - nearest[1]) * (step / dist)),
        )
        # Do not insert goal directly into the tree so goal connection happens cleanly at the end
        if new_node == goal or new_node in parent or not _has_line_of_sight(grid, nearest, new_node):
            continue

        near_nodes = [n for n in nodes if math.hypot(n[0] - new_node[0], n[1] - new_node[1]) <= rewire_radius]
        best_parent = nearest
        best_cost = cost_to_node[nearest] + math.hypot(new_node[0] - nearest[0], new_node[1] - nearest[1])

        for near in near_nodes:
            c = cost_to_node[near] + math.hypot(new_node[0] - near[0], new_node[1] - near[1])
            if c < best_cost and _has_line_of_sight(grid, near, new_node):
                best_parent = near
                best_cost = c

        parent[new_node] = best_parent
        cost_to_node[new_node] = best_cost
        nodes.append(new_node)

        for near in near_nodes:
            new_near_cost = best_cost + math.hypot(near[0] - new_node[0], near[1] - new_node[1])
            if new_near_cost < cost_to_node[near] and _has_line_of_sight(grid, new_node, near):
                parent[near] = new_node
                cost_to_node[near] = new_near_cost

    # Connect best non-goal node near destination
    goal_candidates = [
        n for n in nodes
        if n != goal and math.hypot(n[0] - goal[0], n[1] - goal[1]) <= 2.5 and _has_line_of_sight(grid, n, goal)
    ]
    if not goal_candidates:
        raise RuntimeError("RRT* failed to reach destination within max_iter.")

    best_goal_parent = min(goal_candidates, key=lambda n: cost_to_node[n] + math.hypot(goal[0] - n[0], goal[1] - n[1]))
    parent[goal] = best_goal_parent
    total_cost = cost_to_node[best_goal_parent] + math.hypot(goal[0] - best_goal_parent[0], goal[1] - best_goal_parent[1])

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return path, {"cost": round(total_cost, 2), "expanded": len(nodes), "time_ms": round(elapsed_ms, 3)}



def greedy_best_first(grid: np.ndarray, start: tuple, goal: tuple, allow_diagonal: bool = True):
    """Greedy Best-First Search (f(n) = h(n)): ignores path-so-far cost g(n) for pure heuristic speed."""
    validate_grid(grid, start, goal)
    t0 = time.perf_counter()

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) if allow_diagonal else abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = {start}
    expanded = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return _reconstruct(came_from, current), {"cost": round(current_g, 2), "expanded": expanded, "time_ms": round(elapsed_ms, 3)}

        expanded += 1
        for nxt, step_cost in get_neighbors(grid, current, allow_diagonal):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                g_score[nxt] = current_g + step_cost
                heapq.heappush(open_heap, (heuristic(nxt, goal), g_score[nxt], nxt))

    raise RuntimeError("No collision-free path found.")

def smooth_waypoints_los(grid: np.ndarray, path: list, margin_cells: float = 0.85) -> list:
    """Prune unnecessary grid turns whenever a straight line between two waypoints clears all obstacles."""
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

def simplify_waypoints(path: list) -> list:
    """Remove intermediate collinear grid cells so the Tello flies long straight segments."""
    if len(path) <= 2:
        return path
    waypoints = [path[0]]
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            waypoints.append(path[i])
    waypoints.append(path[-1])
    return waypoints




def waypoints_to_drone_commands(waypoints: list, cell_size_cm: float, initial_heading_deg: float = 0.0) -> list:
    """
    Convert (row, col) waypoints into (turn_deg, forward_cm) pairs for Drone Control.
    Convention: heading 0 deg = facing East (+col), 90 deg = facing South (+row).
    Positive turn_deg = clockwise (cw), negative turn_deg = counter-clockwise (ccw).
    """
    commands = []
    heading = initial_heading_deg

    for i in range(len(waypoints) - 1):
        r1, c1 = waypoints[i]
        r2, c2 = waypoints[i + 1]
        dr, dc = r2 - r1, c2 - c1

        target_heading = math.degrees(math.atan2(dr, dc))
        turn_deg = (target_heading - heading + 180) % 360 - 180
        distance_cm = round(math.hypot(dr, dc) * cell_size_cm)

        commands.append((round(turn_deg), distance_cm))
        heading = target_heading

    return commands


def plot_comparison(grid: np.ndarray, results: dict, start: tuple, goal: tuple):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes_flat = axes.flatten()

    for idx, (ax, (name, (path, waypoints, stats))) in enumerate(zip(axes_flat, results.items())):
        ax.imshow(grid, cmap="Greys", origin="upper")

        pr, pc = zip(*path)
        ax.plot(pc, pr, linestyle="--", linewidth=1.5, label="Full Path")

        wr, wc = zip(*waypoints)
        ax.plot(wc, wr, marker="o", linewidth=2, label="Waypoints")

        ax.scatter(start[1], start[0], color="green", s=90, zorder=5, label="Start")
        ax.scatter(goal[1], goal[0], color="blue", s=90, zorder=5, label="Goal")
        ax.set_title(f"{name}\nCost: {stats['cost']} | Nodes: {stats['expanded']} | {stats['time_ms']}ms", fontsize=10)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot slots if fewer than 6 algorithms are plotted
    for ax in axes_flat[len(results):]:
        ax.axis("off")

    axes_flat[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent/"algorithm_comparison.png", dpi=200)
    plt.show()





if __name__ == "__main__":
    # Sample 10x12 grid matching architecture diagram
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
    start = (0, 0)
    destination = (9, 11)


    results = {}
    for name, fn in [  ("Dijkstra", dijkstra),
            ("A*", astar),
            ("Theta*", theta_star),
            ("RRT", rrt_grid),
            ("RRT*", rrt_star),
            ("Greedy Best First)", greedy_best_first)]:
        
        p, stats = fn(occupancy_grid, start, destination)
        #wp = simplify_waypoints(p)
        wp = smooth_waypoints_los(occupancy_grid, p)
        results[name] = (p, wp, stats)
        print(f"{name:10s} -> {stats} | Waypoints: {wp}")

    plot_comparison(occupancy_grid, results, start, destination)
    print("Finished!")


