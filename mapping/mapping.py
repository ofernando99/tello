"""
Author: Jessie Wang

Occupancy mapping:
Turns the 2D drawing into a grid of free (0) / occupied (1) cells.

Supported inputs
----------------
* ``.json``  geometric description measured off the drawing (recommended)
* ``.txt``   hand-drawn ASCII grid  ('#' obstacle, '.' free, 'S' start, 'G' goal)
* ``.png/.jpg`` scanned drawing; dark pixels = obstacles (needs OpenCV)

Map frame convention (same as an image / the drawing on paper):
    x -> right (columns), y -> down (rows), origin = top-left corner.
JSON files may set ``"origin": "bottom-left"`` and are converted.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

Cell = tuple[int, int]  # (row, col)

@dataclass
class GridMap:
    grid: list[list[int]]
    cell_cm: float
    start: Cell
    destination: Cell
    raw: list[list[int]] | None = field(default=None, repr=False)  # pre-inflation

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0])

    @property
    def occupancy_grid(self):
        """Return the occupancy grid as a 2D NumPy integer array (0 = free, 1 = occupied)."""
        import numpy as np
        return np.array(self.grid, dtype=int)

    @classmethod
    def from_array(
        cls,
        occupancy_grid,
        start: Cell,
        destination: Cell,
        cell_cm: float = 20,
    ) -> "GridMap":
        """Construct a ``GridMap`` directly from a 2D NumPy array (or nested list),
        ``start`` cell ``(row, col)``, and ``destination`` cell ``(row, col)``."""
        return load_array(occupancy_grid, start, destination, cell_cm=cell_cm)

    def to_numpy_tuple(self):
        """Return ``(occupancy_grid, start, destination)`` where ``occupancy_grid``
        is a 2D ``np.ndarray`` of 0s and 1s."""
        return self.occupancy_grid, self.start, self.destination

    def __iter__(self):
        """Allow unpacking directly: ``occupancy_grid, start, destination = load_map(...)``."""
        return iter(self.to_numpy_tuple())

    def to_python_code(self) -> str:
        """Format the map as an executable Python snippet defining
        ``occupancy_grid = np.array([...])``, ``start``, and ``destination``."""
        row_strings = ["        [" + ", ".join(str(val) for val in row) + "]," for row in self.grid]
        lines = [
            "occupancy_grid = np.array([",
            *row_strings,
            "    ])",
            f"start = {self.start}",
            f"destination = {self.destination}",
        ]
        return "\n".join(lines)

    def in_bounds(self, cell: Cell) -> bool:
        """Return True if ``cell`` (row, col) lies within the grid dimensions."""
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------ #
    def inflate(self, margin_cm: float, include_walls: bool = True) -> "GridMap":
        """Grow obstacles by ``margin_cm`` so the planner keeps a safe distance
        (configuration-space trick: treat the drone as a point)."""
        radius_cells = int(math.ceil(margin_cm / self.cell_cm))
        inflated_grid = [row[:] for row in self.grid]

        self._expand_obstacles(inflated_grid, radius_cells)
        if include_walls and radius_cells > 0:
            self._inflate_arena_walls(inflated_grid, radius_cells)
        self._clear_start_and_destination(inflated_grid)

        return GridMap(
            inflated_grid,
            self.cell_cm,
            self.start,
            self.destination,
            raw=self.grid,
        )

    def _expand_obstacles(self, inflated_grid: list[list[int]], radius_cells: int) -> None:
        """Expand each occupied cell in ``self.grid`` outward in a Euclidean disk
        of radius ``radius_cells``, marking surrounding cells in ``inflated_grid``."""
        radius_sq = radius_cells * radius_cells
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] != 1:
                    continue
                for row_offset in range(-radius_cells, radius_cells + 1):
                    for col_offset in range(-radius_cells, radius_cells + 1):
                        if row_offset * row_offset + col_offset * col_offset <= radius_sq:
                            neighbor_row = row + row_offset
                            neighbor_col = col + col_offset
                            if self.in_bounds((neighbor_row, neighbor_col)):
                                inflated_grid[neighbor_row][neighbor_col] = 1

    def _inflate_arena_walls(self, inflated_grid: list[list[int]], radius_cells: int) -> None:
        """Pad the outer boundary of the arena with a keep-out margin so the
        drone maintains clearance from surrounding walls or safety nets.
        Skipped if start or destination lies on the outer border (e.g. pre-discretized grids)."""
        wall_margin_cells = max(1, radius_cells - 1)  # walls/nets: slightly smaller margin
        for row, col in (self.start, self.destination):
            if (
                row < wall_margin_cells
                or row >= self.rows - wall_margin_cells
                or col < wall_margin_cells
                or col >= self.cols - wall_margin_cells
            ):
                return
        for row in range(self.rows):
            for col in range(self.cols):
                near_top_or_bottom = row < wall_margin_cells or row >= self.rows - wall_margin_cells
                near_left_or_right = col < wall_margin_cells or col >= self.cols - wall_margin_cells
                if near_top_or_bottom or near_left_or_right:
                    inflated_grid[row][col] = 1

    def _clear_start_and_destination(self, inflated_grid: list[list[int]]) -> None:
        """Verify that start and destination do not lie inside actual obstacles, and
        ensure safety-margin inflation never blocks the drone from starting or landing."""
        for row, col in (self.start, self.destination):
            if self.grid[row][col] == 1:
                raise ValueError(f"start/destination {(row, col)} lies inside an obstacle")
            inflated_grid[row][col] = 0

    def to_ascii(self, path: list[Cell] | None = None) -> str:
        """Render the occupancy grid (and optional path) as a human-readable ASCII string."""
        path_cells = set(path or [])
        raw_grid = self.raw or self.grid
        lines = []
        for row in range(self.rows):
            row_symbols = []
            for col in range(self.cols):
                cell = (row, col)
                if cell == self.start:
                    symbol = "S"
                elif cell == self.destination:
                    symbol = "G"
                elif cell in path_cells:
                    symbol = "*"
                elif raw_grid[row][col] == 1:
                    symbol = "#"
                elif self.grid[row][col] == 1:
                    symbol = "+"  # free in reality, blocked by safety margin
                else:
                    symbol = "."
                row_symbols.append(symbol)
            lines.append(" ".join(row_symbols))
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Loaders
# ---------------------------------------------------------------------- #
def load_map(path: str, **kwargs) -> GridMap:
    """Dispatch map loading based on file extension (.json, .txt, .py, or image)."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        return load_json(path)
    if extension == ".txt":
        return load_ascii(path, **kwargs)
    if extension == ".py":
        return load_python(path, **kwargs)
    if extension in (".png", ".jpg", ".jpeg", ".bmp"):
        return load_image(path, **kwargs)
    raise ValueError(f"Unsupported map format: {extension}")


def load_array(
    occupancy_grid,
    start: Cell,
    destination: Cell,
    cell_cm: float = 20,
) -> GridMap:
    """Construct a ``GridMap`` from a 2D ``np.ndarray`` (or nested list) of 0s and 1s,
    ``start = (row, col)``, and ``destination = (row, col)``."""
    if hasattr(occupancy_grid, "tolist"):
        raw_rows = occupancy_grid.tolist()
    else:
        raw_rows = occupancy_grid
    grid = [[1 if int(val) != 0 else 0 for val in row] for row in raw_rows]
    start_cell = (int(start[0]), int(start[1]))
    dest_cell = (int(destination[0]), int(destination[1]))
    return GridMap(grid, float(cell_cm), start_cell, dest_cell)


def load_python(path: str, cell_cm: float = 20) -> GridMap:
    """Load a ``GridMap`` from a Python file defining ``occupancy_grid`` (or ``grid``),
    ``start = (row, col)``, and ``destination = (row, col)`` (or ``goal``)."""
    import textwrap
    import numpy as np

    namespace: dict = {"np": np}
    with open(path) as file:
        source = file.read()
    # Dedent in case the file contains an indented snippet
    exec(compile(textwrap.dedent(source), path, "exec"), namespace)
    occupancy_grid = namespace.get("occupancy_grid")
    if occupancy_grid is None:
        occupancy_grid = namespace.get("grid")
    start = namespace.get("start")
    destination = namespace.get("destination") or namespace.get("goal")
    if occupancy_grid is None or start is None or destination is None:
        raise ValueError(
            f"Python map '{path}' must define 'occupancy_grid', 'start', and 'destination'"
        )
    return load_array(
        occupancy_grid,
        start,
        destination,
        cell_cm=namespace.get("cell_cm", cell_cm),
    )


def load_ascii(path: str, cell_cm: float = 20) -> GridMap:
    """Load an occupancy grid from a hand-drawn ASCII text file."""
    grid, start, destination = [], None, None
    with open(path) as file:
        raw_lines = [line.rstrip("\n") for line in file if line.strip() and not line.startswith(";")]
    for row_idx, line in enumerate(raw_lines):
        grid_row = []
        for col_idx, char in enumerate(line.replace(" ", "")):
            if char in "#X1":
                grid_row.append(1)
            else:
                grid_row.append(0)
                if char == "S":
                    start = (row_idx, col_idx)
                elif char == "G":
                    destination = (row_idx, col_idx)
        grid.append(grid_row)
    if start is None or destination is None:
        raise ValueError("ASCII map needs one 'S' and one 'G'")
    max_cols = max(len(grid_row) for grid_row in grid)
    grid = [grid_row + [1] * (max_cols - len(grid_row)) for grid_row in grid]
    return GridMap(grid, cell_cm, start, destination)


def init_grid(
    arena_width_cm: float, arena_height_cm: float, cell_cm: float
) -> tuple[list[list[int]], int, int]:
    """Create an empty (all-zero) 2D occupancy grid for an arena of the given
    dimensions and cell resolution, returning ``(grid, rows, cols)``."""
    rows = int(math.ceil(arena_height_cm / cell_cm))
    cols = int(math.ceil(arena_width_cm / cell_cm))
    grid = [[0] * cols for _ in range(rows)]
    return grid, rows, cols


def render_obstacles(
    grid: list[list[int]],
    obstacles: list[dict],
    cell_cm: float,
    to_top_left_y,
) -> None:
    """Rasterize geometric obstacles (rectangles and circles) onto ``grid`` in-place,
    marking any cell whose physical bounding box intersects an obstacle as occupied (1)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for obstacle in obstacles:
        obstacle_type = obstacle.get("type", "rect")
        for row in range(rows):
            for col in range(cols):
                cell_x0, cell_y0 = col * cell_cm, row * cell_cm
                cell_x1, cell_y1 = cell_x0 + cell_cm, cell_y0 + cell_cm
                if obstacle_type == "rect":
                    rect_x = obstacle["x"]
                    rect_y = to_top_left_y(obstacle["y"], obstacle["h"])
                    rect_w, rect_h = obstacle["w"], obstacle["h"]
                    intersects_obstacle = (
                        cell_x0 < rect_x + rect_w
                        and cell_x1 > rect_x
                        and cell_y0 < rect_y + rect_h
                        and cell_y1 > rect_y
                    )
                elif obstacle_type == "circle":
                    center_x = obstacle["x"]
                    center_y = to_top_left_y(obstacle["y"])
                    radius = obstacle["r"]
                    nearest_x = min(max(center_x, cell_x0), cell_x1)
                    nearest_y = min(max(center_y, cell_y0), cell_y1)
                    intersects_obstacle = (
                        (nearest_x - center_x) ** 2 + (nearest_y - center_y) ** 2 < radius ** 2
                    )
                else:
                    raise ValueError(f"unknown obstacle type {obstacle_type}")
                if intersects_obstacle:
                    grid[row][col] = 1


def load_json(path: str) -> GridMap:
    """Load an occupancy grid from a geometric JSON specification.

    Example::

        {
          "arena_cm": [400, 300],          # width, height
          "cell_cm": 20,
          "origin": "top-left",            # or "bottom-left"
          "start": [60, 240], "goal": [340, 60],      # cm
          "obstacles": [
             {"type": "rect", "x": 120, "y": 0, "w": 40, "h": 180},
             {"type": "circle", "x": 260, "y": 180, "r": 30}
          ]
        }
    """
    with open(path) as file:
        spec = json.load(file)
    cell_cm = float(spec.get("cell_cm", 20))
    destination = spec.get("destination") or spec.get("goal")

    # Support direct discrete occupancy_grid in JSON as well as geometric arena_cm specs
    if "occupancy_grid" in spec or "grid" in spec:
        raw_grid = spec.get("occupancy_grid") or spec.get("grid")
        return load_array(raw_grid, tuple(spec["start"]), tuple(destination), cell_cm=cell_cm)

    arena_width_cm, arena_height_cm = spec["arena_cm"]
    flip_y_axis = spec.get("origin", "top-left") == "bottom-left"
    grid, rows, cols = init_grid(arena_width_cm, arena_height_cm, cell_cm)

    def to_top_left_y(y_cm: float, height_cm: float = 0.0) -> float:
        return arena_height_cm - y_cm - height_cm if flip_y_axis else y_cm

    render_obstacles(grid, spec.get("obstacles", []), cell_cm, to_top_left_y)

    def cm_to_cell(point_cm: list[float] | tuple[float, float]) -> Cell:
        x_cm, y_cm = point_cm[0], to_top_left_y(point_cm[1])
        return (min(rows - 1, int(y_cm // cell_cm)), min(cols - 1, int(x_cm // cell_cm)))

    return GridMap(
        grid,
        cell_cm,
        cm_to_cell(spec["start"]),
        cm_to_cell(destination),
    )


def load_image(
    path: str,
    cell_cm: float = 20,
    arena_width_cm: float = 500,
    start_px=None,
    goal_px=None,
    dark_threshold: int = 100,
    obstacle_ratio_threshold: float = 0.35,
    min_grid_size: int = 20,
) -> GridMap:
    """Scanned drawing -> grid. Dark ink = obstacle. Start/goal are found as
    green / red blobs, or passed in pixel coordinates."""
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img_height_px, img_width_px = img.shape[:2]
    cm_per_px = arena_width_cm / img_width_px
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def find_blob_centre(mask):
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    if start_px is None:
        green_mask = cv2.inRange(hsv_img, (40, 80, 80), (85, 255, 255))
        start_px = find_blob_centre(green_mask)
    if goal_px is None:
        red_low = cv2.inRange(hsv_img, (0, 80, 80), (10, 255, 255))
        red_high = cv2.inRange(hsv_img, (170, 80, 80), (180, 255, 255))
        goal_px = find_blob_centre(red_low | red_high)
    if start_px is None or goal_px is None:
        raise ValueError("could not find start (green) / goal (red); pass start_px/goal_px")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Dark AND unsaturated pixels count as walls (so coloured start/goal blobs are ignored)
    obstacle_mask = ((gray_img < dark_threshold) & (hsv_img[:, :, 1] < 80)).astype(np.uint8)
    cell_size_px = cell_cm / cm_per_px
    if min_grid_size > 0:
        max_cell_size_px = min(img_width_px, img_height_px) / float(min_grid_size)
        if cell_size_px > max_cell_size_px:
            cell_size_px = max_cell_size_px
            cell_cm = cell_size_px * cm_per_px

    grid, rows, cols = init_grid(img_width_px, img_height_px, cell_size_px)
    for row in range(rows):
        for col in range(cols):
            y0_px, y1_px = int(row * cell_size_px), int((row + 1) * cell_size_px)
            x0_px, x1_px = int(col * cell_size_px), int((col + 1) * cell_size_px)
            cell_patch = obstacle_mask[y0_px:y1_px, x0_px:x1_px]
            if cell_patch.size and cell_patch.mean() > obstacle_ratio_threshold:
                grid[row][col] = 1

    def px_to_cell(point_px: tuple[float, float]) -> Cell:
        return (
            min(rows - 1, int(point_px[1] // cell_size_px)),
            min(cols - 1, int(point_px[0] // cell_size_px)),
        )

    start_cell = px_to_cell(start_px)
    goal_cell = px_to_cell(goal_px)
    grid[start_cell[0]][start_cell[1]] = 0
    grid[goal_cell[0]][goal_cell[1]] = 0

    return GridMap(grid, cell_cm, start_cell, goal_cell)
