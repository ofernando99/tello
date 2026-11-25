"""
Author: Jessie Wang

Benchmark suite for Task 1 (navigation) and Task 2 (checkerboard tracking).

Task 1 runs the team pipeline modules on representative maps in mapping/maps/
(example_arena.png, example_high_res_maze.jpg, project_2_layout_test.txt):
  * mapping:   load time, grid size, obstacle share before and after inflation
  * planning:  A*, Dijkstra, Greedy (4- and 8-way), Theta*, RRT, RRT*
  * control:   waypoint reduction (raw A* vs collinear pruning vs the
               pipeline's line-of-sight smoothing) and Tello `go` commands
Task 2 runs object_tracking.tracker.detect_board on a recorded Tello video
under 9 perturbations plus a masked-board negative control.

The report is written to benchmarks/BENCHMARK_SUMMARY.md and plots to
benchmarks/plots/. No module outside this file is changed.

Usage:
    python benchmarks/run_benchmarks.py                    # Task 1 and Task 2
    python benchmarks/run_benchmarks.py --task 1           # Task 1 only
    python benchmarks/run_benchmarks.py --task 2           # Task 2 only
    python benchmarks/run_benchmarks.py mapping/maps/x.txt # Selected map(s)
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import matplotlib

# Render to files only: the team modules import pyplot at import time.
matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drone_control.path_command import (  # noqa: E402
    generate_commands,
    generate_instructions,
    transform,
)
from mapping.mapping import GridMap, load_map  # noqa: E402
from object_tracking.tracker import detect_board  # noqa: E402
from path_planning import path_planner_algorithms as algos  # noqa: E402
from path_planning.path_planner import plan_path  # noqa: E402

# =========================================================================== #
# Configuration & Constants
# =========================================================================== #
BENCH_DIR = os.path.join(ROOT, "benchmarks")
PLOTS_DIR = os.path.join(BENCH_DIR, "plots")
SUMMARY_MD = os.path.join(BENCH_DIR, "BENCHMARK_SUMMARY.md")
DEFAULT_VIDEO = os.path.join(ROOT, "data", "Tello_video.mp4")
DEFAULT_MAPS = (
    "mapping/maps/project_2_layout_test.txt",
    "mapping/maps/example_arena.png",
    "mapping/maps/example_high_res_maze.jpg",
)

DEFAULT_MARGIN_CM = 10.0  # matches the inflation example in README.md
TIMING_REPEATS = 3        # median of 3 runs for stable sub-ms timings
DRONE_SPEED_CM_S = 20     # same speed main.py sends to the drone
TELLO_GO_MIN_CM = 20      # SDK rejects `go` when |x|, |y| and |z| are all below this
TELLO_GO_MAX_CM = 500
FRAME_STEP = 3            # sample every 3rd video frame
NEGATIVE_MASK_BGR = (90, 130, 170)  # cardboard colour used to hide the board

BASELINE_CONN = {"4-way": "4-way", "8-way": "8-way", "any-angle": "8-way"}
PLOTTED_PLANNERS = [
    ("A*", "8-way"),
    ("Dijkstra", "8-way"),
    ("Greedy", "8-way"),
    ("Theta*", "any-angle"),
    ("RRT", "any-angle"),
    ("RRT*", "any-angle"),
]


# =========================================================================== #
# Data Models
# =========================================================================== #
@dataclass(frozen=True)
class PlannerSpec:
    name: str
    func: Callable
    conn: str  # "4-way", "8-way" or "any-angle"
    kwargs: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.conn})"


@dataclass(frozen=True)
class MappingStats:
    fmt: str
    rows: int
    cols: int
    cell_cm: float
    load_ms: float
    obst_raw: float
    obst_inflated: float
    used_inflated: bool
    start: tuple[int, int]
    destination: tuple[int, int]

    @property
    def grid_shape(self) -> str:
        return f"{self.rows}x{self.cols}"

    @property
    def total_cells(self) -> int:
        return self.rows * self.cols

    @property
    def inflation_note(self) -> str:
        return "Applied" if self.used_inflated else "Skipped (blocks route)"


@dataclass
class PlannerRun:
    algo: str
    conn: str
    found: bool
    path: list[tuple[int, int]] = field(default_factory=list)
    cost: float = 0.0
    expanded: int = 0
    pct_grid: float = 0.0
    time_ms: float = 0.0
    gap: float | None = None
    error: str = ""


@dataclass(frozen=True)
class PlannerAggregate:
    algo: str
    conn: str
    solved: int
    total: int
    mean_pct_grid: float | None
    median_time_ms: float | None
    mean_time_ms: float | None
    mean_gap: float | None
    min_gap: float | None
    max_gap: float | None

    @property
    def label(self) -> str:
        return f"{self.algo} ({self.conn})"


@dataclass(frozen=True)
class CommandVariant:
    wps: int
    cmds: int
    invalid: int


@dataclass(frozen=True)
class AblationStats:
    raw: CommandVariant
    simplified: CommandVariant
    smoothed: CommandVariant

    @property
    def wp_reduction_pct(self) -> float:
        return pct_reduction(self.raw.wps, self.smoothed.wps)

    @property
    def cmd_reduction_pct(self) -> float:
        return pct_reduction(self.raw.cmds, self.smoothed.cmds)


@dataclass(frozen=True)
class MapBenchmark:
    label: str
    mapping: MappingStats
    planners: list[PlannerRun]
    ablation: AblationStats
    plot_rel: str


@dataclass(frozen=True)
class Task1Report:
    maps: dict[str, MapBenchmark]
    failures: dict[str, str]
    margin_cm: float
    aggregates: list[PlannerAggregate]
    summary_plot_rel: str


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    scenario: str
    fn: Callable[[np.ndarray], np.ndarray]
    compare_centre: bool = True


@dataclass(frozen=True)
class PerturbationResult:
    condition: str
    scenario: str
    compared: bool
    det_pct: float
    err_px: float | None
    ms: float


@dataclass(frozen=True)
class NegativeControlResult:
    masked: int
    fp: int
    fp_pct: float
    ms: float


@dataclass(frozen=True)
class Task2Report:
    video: str
    n_frames: int
    rows: list[PerturbationResult]
    negative: NegativeControlResult
    summary_plot_rel: str


PLANNERS: list[PlannerSpec] = [
    PlannerSpec("A*", algos.astar, "4-way", {"allow_diagonal": False}),
    PlannerSpec("Dijkstra", algos.dijkstra, "4-way", {"allow_diagonal": False}),
    PlannerSpec("Greedy", algos.greedy_best_first, "4-way", {"allow_diagonal": False}),
    PlannerSpec("A*", algos.astar, "8-way", {"allow_diagonal": True}),
    PlannerSpec("Dijkstra", algos.dijkstra, "8-way", {"allow_diagonal": True}),
    PlannerSpec("Greedy", algos.greedy_best_first, "8-way", {"allow_diagonal": True}),
    PlannerSpec("Theta*", algos.theta_star, "any-angle"),
    PlannerSpec("RRT", algos.rrt_grid, "any-angle"),
    PlannerSpec("RRT*", algos.rrt_star, "any-angle"),
]


# =========================================================================== #
# Pure Helpers
# =========================================================================== #
def obstacle_pct(grid) -> float:
    """Percentage of occupied cells in a 2D grid."""
    arr = np.asarray(grid)
    return 100.0 * float(arr.sum()) / arr.size


def optimality_gap(cost: float, baseline: float | None) -> float | None:
    """Percentage cost difference from the baseline, or None if no baseline."""
    if not baseline:
        return None
    return 100.0 * (cost - baseline) / baseline


def format_gap(gap: float | None) -> str:
    """Readable label for an optimality gap."""
    if gap is None:
        return "N/A"
    if abs(gap) < 1e-4:
        return "Optimal (0.0%)"
    return f"{gap:+.1f}%"


def format_range(values: list[float], fmt: str = "{:.0f}") -> str:
    """'a–b' range label, or 'n/a' if empty."""
    if not values:
        return "n/a"
    lo, hi = min(values), max(values)
    return fmt.format(lo) if abs(hi - lo) < 1e-9 else f"{fmt.format(lo)}–{fmt.format(hi)}"


def invalid_go_count(commands: list[str]) -> int:
    """Count `go x y z speed` commands the Tello SDK would reject as out of range."""
    bad = 0
    for cmd in commands:
        x, y, z = (abs(int(v)) for v in cmd.split()[1:4])
        too_small = max(x, y, z) < TELLO_GO_MIN_CM
        too_large = max(x, y, z) > TELLO_GO_MAX_CM
        bad += too_small or too_large
    return bad


def pct_reduction(before: int, after: int) -> float:
    """Percentage reduction from ``before`` to ``after``."""
    return 100.0 * (before - after) / max(1, before)


def map_label(path: str) -> str:
    """Repo-relative path used as a stable key in tables and file names."""
    return os.path.relpath(path, ROOT)


def plot_stem(label: str) -> str:
    """File-name-safe stem for a map's plot."""
    return label.replace(os.sep, "_").replace(".", "_")


# =========================================================================== #
# TASK 1: Mapping, Path Planning & Command Generation
# =========================================================================== #
def discover_maps(paths: list[str]) -> list[str]:
    """Return maps given on the CLI, or the default representative map suite."""
    if paths:
        return [os.path.abspath(p) for p in paths]
    return [os.path.join(ROOT, rel) for rel in DEFAULT_MAPS]


def is_solvable(gmap: GridMap) -> bool:
    """True if 4-way A* finds a route (the strictest connectivity we benchmark)."""
    try:
        algos.astar(gmap.occupancy_grid, gmap.start, gmap.destination, allow_diagonal=False)
    except RuntimeError:
        return False
    return True


def evaluate_mapping(path: str, margin_cm: float) -> tuple[MappingStats, GridMap]:
    """Load and inflate one map; return its stats and the grid used for planning."""
    t0 = time.perf_counter()
    raw = load_map(path)
    load_ms = (time.perf_counter() - t0) * 1000.0
    inflated = raw.inflate(margin_cm)
    use_inflated = is_solvable(inflated)
    active_map = inflated if use_inflated else raw

    stats = MappingStats(
        fmt=os.path.splitext(path)[1].lstrip("."),
        rows=raw.rows,
        cols=raw.cols,
        cell_cm=raw.cell_cm,
        load_ms=load_ms,
        obst_raw=obstacle_pct(raw.grid),
        obst_inflated=obstacle_pct(inflated.grid),
        used_inflated=use_inflated,
        start=raw.start,
        destination=raw.destination,
    )
    return stats, active_map


def execute_single_planner(spec: PlannerSpec, gmap: GridMap) -> PlannerRun:
    """Run one planner TIMING_REPEATS times and return a PlannerRun."""
    grid = gmap.occupancy_grid
    try:
        times, path, stats = [], [], {}
        for _ in range(TIMING_REPEATS):
            path, stats = spec.func(grid, gmap.start, gmap.destination, **spec.kwargs)
            times.append(stats["time_ms"])
        expanded = int(stats["expanded"])
        return PlannerRun(
            algo=spec.name,
            conn=spec.conn,
            found=True,
            path=path,
            cost=float(stats["cost"]),
            expanded=expanded,
            pct_grid=100.0 * expanded / grid.size,
            time_ms=statistics.median(times),
        )
    except RuntimeError as exc:
        return PlannerRun(algo=spec.name, conn=spec.conn, found=False, error=str(exc))


def evaluate_planners(gmap: GridMap) -> list[PlannerRun]:
    """Run all configured planners on a map and compute optimality gaps vs Dijkstra."""
    runs = [execute_single_planner(spec, gmap) for spec in PLANNERS]
    baselines = {r.conn: r.cost for r in runs if r.algo == "Dijkstra" and r.found}
    for r in runs:
        if r.found:
            r.gap = optimality_gap(r.cost, baselines.get(BASELINE_CONN[r.conn]))
    return runs


def to_go_commands(gmap: GridMap, waypoints: list) -> list[str]:
    """Convert waypoints to Tello `go` commands exactly as drone_control.navigation does."""
    ground_path = transform(gmap.occupancy_grid, waypoints)
    route = generate_instructions(ground_path, gmap.cell_cm, starting_angle=0)
    return generate_commands(route, DRONE_SPEED_CM_S)


def evaluate_command_variant(gmap: GridMap, waypoints: list) -> CommandVariant:
    """Measure waypoint count, Tello command count, and out-of-range command count."""
    cmds = to_go_commands(gmap, waypoints)
    return CommandVariant(wps=len(waypoints), cmds=len(cmds), invalid=invalid_go_count(cmds))


def evaluate_ablation(gmap: GridMap) -> AblationStats:
    """Compare raw 8-way A*, collinear pruning, and the pipeline's LOS smoothing."""
    grid = gmap.occupancy_grid
    raw_wps = plan_path(grid, gmap.start, gmap.destination, gmap.cell_cm, smooth=False)
    simplified_wps = algos.simplify_waypoints(raw_wps)
    smoothed_wps = plan_path(grid, gmap.start, gmap.destination, gmap.cell_cm)
    return AblationStats(
        raw=evaluate_command_variant(gmap, raw_wps),
        simplified=evaluate_command_variant(gmap, simplified_wps),
        smoothed=evaluate_command_variant(gmap, smoothed_wps),
    )


def aggregate_planners(benchmarks: dict[str, MapBenchmark]) -> list[PlannerAggregate]:
    """Aggregate solve rate, explored %, runtime, and optimality gap across all maps."""
    total_maps = len(benchmarks)
    aggregates = []
    for spec in PLANNERS:
        runs = [
            r
            for b in benchmarks.values()
            for r in b.planners
            if r.algo == spec.name and r.conn == spec.conn and r.found
        ]
        gaps = [r.gap for r in runs if r.gap is not None]
        aggregates.append(
            PlannerAggregate(
                algo=spec.name,
                conn=spec.conn,
                solved=len(runs),
                total=total_maps,
                mean_pct_grid=statistics.mean(r.pct_grid for r in runs) if runs else None,
                median_time_ms=statistics.median(r.time_ms for r in runs) if runs else None,
                mean_time_ms=statistics.mean(r.time_ms for r in runs) if runs else None,
                mean_gap=statistics.mean(gaps) if gaps else None,
                min_gap=min(gaps) if gaps else None,
                max_gap=max(gaps) if gaps else None,
            )
        )
    return aggregates


def explored_saving_vs_dijkstra(benchmarks: dict[str, MapBenchmark], conn: str = "8-way") -> list[float]:
    """Per map: % fewer cells explored by A* than Dijkstra at the same connectivity."""
    savings = []
    for bench in benchmarks.values():
        by_key = {(r.algo, r.conn): r for r in bench.planners if r.found}
        astar_run = by_key.get(("A*", conn))
        dijkstra_run = by_key.get(("Dijkstra", conn))
        if astar_run and dijkstra_run:
            savings.append(pct_reduction(dijkstra_run.expanded, astar_run.expanded))
    return savings


# =========================================================================== #
# TASK 1: Plotting & Console Output
# =========================================================================== #
def plot_map_comparison(gmap: GridMap, runs: list[PlannerRun], title: str, out_file: str) -> None:
    """Save a 2x3 grid of planner paths over the (inflated) occupancy grid."""
    by_key = {(r.algo, r.conn): r for r in runs}
    raw = np.asarray(gmap.raw or gmap.grid)
    # 0 free, 1 safety margin only, 2 real obstacle
    layers = raw * 2 + (np.asarray(gmap.grid) - raw).clip(0, 1)
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#bcd7ee", "#4a4a4a"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, key in zip(axes.flat, PLOTTED_PLANNERS):
        ax.imshow(layers, cmap=cmap, vmin=0, vmax=2, origin="upper")
        run = by_key[key]
        if run.found:
            pr, pc = zip(*run.path)
            ax.plot(pc, pr, color="#e8710a", linewidth=2)
            ax.set_title(
                f"{key[0]} ({key[1]})  cost {run.cost:.1f}  explored {run.expanded}",
                fontsize=10,
            )
        else:
            ax.set_title(f"{key[0]} ({key[1]}): no path", fontsize=10)
        ax.scatter(gmap.start[1], gmap.start[0], c="green", s=60, zorder=5)
        ax.scatter(gmap.destination[1], gmap.destination[0], c="red", s=60, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)


def plot_task1_summary(
    aggregates: list[PlannerAggregate],
    benchmarks: dict[str, MapBenchmark],
    out_file: str,
) -> None:
    """Save a 3-panel aggregate chart comparing planners and command reduction across maps."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.2))

    labels = [a.label for a in aggregates]
    y_pos = np.arange(len(aggregates))
    palette = ["#1f77b4" if a.conn == "4-way" else "#2ca02c" if a.conn == "8-way" else "#ff7f0e" for a in aggregates]

    # Panel 1: Mean % Grid Explored
    explored_vals = [a.mean_pct_grid or 0.0 for a in aggregates]
    bars1 = ax1.barh(y_pos, explored_vals, color=palette, alpha=0.88)
    ax1.set_yticks(y_pos, labels=labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Grid Explored (%)")
    ax1.set_title("Search Effort (Solved Ratio)")
    ax1.set_xlim(0, max(explored_vals + [60.0]) * 1.22)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, agg in zip(bars1, aggregates):
        val = agg.mean_pct_grid or 0.0
        ax1.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}% ({agg.solved}/{agg.total})",
            va="center",
            fontsize=8,
        )

    # Panel 2: Mean Cost Gap vs Optimal (%)
    gap_vals = [a.mean_gap or 0.0 for a in aggregates]
    gap_colors = ["#2ca02c" if g <= 1e-4 else "#d62728" for g in gap_vals]
    bars2 = ax2.barh(y_pos, gap_vals, color=gap_colors, alpha=0.85)
    ax2.axvline(0, color="#333333", linewidth=1)
    ax2.set_yticks(y_pos, labels=[""] * len(aggregates))
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Cost Gap vs Optimal (%)")
    ax2.set_title("Path Optimality Gap (Lower is Better)")
    lo_gap = min(gap_vals + [-8.0])
    hi_gap = max(gap_vals + [25.0])
    ax2.set_xlim(lo_gap * 1.35, hi_gap * 1.28)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, agg in zip(bars2, aggregates):
        val = agg.mean_gap or 0.0
        offset = 0.6 if val >= 0 else -0.6
        align = "left" if val >= 0 else "right"
        ax2.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            va="center",
            ha=align,
            fontsize=8,
        )

    # Panel 3: Waypoint / Command Reduction across maps
    map_names = [os.path.basename(lbl) for lbl in benchmarks]
    x_pos = np.arange(len(map_names))
    raw_cmds = [b.ablation.raw.cmds for b in benchmarks.values()]
    pruned_cmds = [b.ablation.simplified.cmds for b in benchmarks.values()]
    smooth_cmds = [b.ablation.smoothed.cmds for b in benchmarks.values()]
    width = 0.26
    ax3.bar(x_pos - width, raw_cmds, width, label="Raw A*", color="#9ecae1")
    ax3.bar(x_pos, pruned_cmds, width, label="Collinear Pruned", color="#6baed6")
    ax3.bar(x_pos + width, smooth_cmds, width, label="LOS Smoothed", color="#08519c")
    ax3.set_xticks(x_pos, labels=map_names, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Tello `go` Commands")
    ax3.set_title("Command Reduction by Map")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Task 1 Summary: Planner Comparison & Command Reduction", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def print_planner_runs(runs: list[PlannerRun]) -> None:
    """Print one map's planner results to the terminal."""
    print(
        f"{'Algorithm':<10} {'Conn':<10} {'Cost':>8} {'Explored':>9} "
        f"{'% Grid':>7} {'Time ms':>9} {'vs Optimal':>15}"
    )
    for r in runs:
        if not r.found:
            print(f"{r.algo:<10} {r.conn:<10} {'no path':>8}")
            continue
        print(
            f"{r.algo:<10} {r.conn:<10} {r.cost:>8.2f} {r.expanded:>9} "
            f"{r.pct_grid:>6.1f}% {r.time_ms:>9.2f} {format_gap(r.gap):>15}"
        )


def print_aggregate_planners(aggregates: list[PlannerAggregate]) -> None:
    """Print cross-map planner summary to the terminal."""
    print("\n--- Cross-Map Planner Summary ---")
    print(
        f"{'Algorithm':<10} {'Conn':<10} {'Solved':>7} {'Mean %Grid':>11} "
        f"{'Median ms':>10} {'Mean Gap':>15}"
    )
    for a in aggregates:
        solved_str = f"{a.solved}/{a.total}"
        grid_str = f"{a.mean_pct_grid:.1f}%" if a.mean_pct_grid is not None else "—"
        time_str = f"{a.median_time_ms:.2f}" if a.median_time_ms is not None else "—"
        gap_str = format_gap(a.mean_gap)
        print(
            f"{a.algo:<10} {a.conn:<10} {solved_str:>7} {grid_str:>11} "
            f"{time_str:>10} {gap_str:>15}"
        )


def benchmark_one_map(path: str, margin_cm: float) -> MapBenchmark:
    """Run mapping, planners, ablation and plot for one map."""
    label = map_label(path)
    mapping_stats, gmap = evaluate_mapping(path, margin_cm)
    print(
        f"\n--- {label}: {mapping_stats.grid_shape} grid, "
        f"{mapping_stats.cell_cm:.1f} cm cells, inflation {mapping_stats.inflation_note} ---"
    )
    planner_runs = evaluate_planners(gmap)
    print_planner_runs(planner_runs)
    ablation = evaluate_ablation(gmap)

    plot_rel = f"plots/{plot_stem(label)}_compare.png"
    plot_map_comparison(gmap, planner_runs, label, os.path.join(BENCH_DIR, plot_rel))
    print(f"  -> saved benchmarks/{plot_rel}")
    return MapBenchmark(
        label=label,
        mapping=mapping_stats,
        planners=planner_runs,
        ablation=ablation,
        plot_rel=plot_rel,
    )


def run_task1_benchmarks(map_paths: list[str], margin_cm: float) -> Task1Report:
    """Benchmark every map; maps that fail to load are recorded, not fatal."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"TASK 1 BENCHMARK: MAPPING + PATH PLANNING + COMMANDS (margin {margin_cm:g} cm)")
    print("=" * 78)
    benchmarks: dict[str, MapBenchmark] = {}
    failures: dict[str, str] = {}
    for path in map_paths:
        label = map_label(path)
        try:
            benchmarks[label] = benchmark_one_map(path, margin_cm)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            failures[label] = f"{type(exc).__name__}: {exc}"
            print(f"\n--- {label}: FAILED ({failures[label]}) ---")

    aggregates = aggregate_planners(benchmarks)
    if benchmarks:
        print_aggregate_planners(aggregates)
    summary_plot_rel = "plots/task1_summary.png"
    if benchmarks:
        plot_task1_summary(aggregates, benchmarks, os.path.join(BENCH_DIR, summary_plot_rel))
        print(f"  -> saved benchmarks/{summary_plot_rel}")
    return Task1Report(
        maps=benchmarks,
        failures=failures,
        margin_cm=margin_cm,
        aggregates=aggregates,
        summary_plot_rel=summary_plot_rel,
    )


# =========================================================================== #
# TASK 2: Checkerboard Detector Robustness
# =========================================================================== #
def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _small(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = np.full_like(img, 128)
    s = cv2.resize(img, None, fx=factor, fy=factor)
    y, x = (h - s.shape[0]) // 2, (w - s.shape[1]) // 2
    canvas[y : y + s.shape[0], x : x + s.shape[1]] = s
    return canvas


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, 1)


def _noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noisy = img + np.random.RandomState(42).normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


PERTURBATIONS: list[PerturbationSpec] = [
    PerturbationSpec("Original (Baseline)", "Unmodified Tello flight stream", lambda f: f, True),
    PerturbationSpec("Motion Blur (9 px)", "Rapid drone pitch/roll gust", lambda f: cv2.filter2D(f, -1, np.ones((1, 9)) / 9), True),
    PerturbationSpec("Gaussian Blur (7x7)", "Camera defocus / vibration", lambda f: cv2.GaussianBlur(f, (7, 7), 0), True),
    PerturbationSpec("Dim Lighting (x0.4)", "Shadowed arena corner", lambda f: cv2.convertScaleAbs(f, alpha=0.4), True),
    PerturbationSpec("Bright Exposure (x1.6)", "Overhead spotlight glare", lambda f: cv2.convertScaleAbs(f, alpha=1.6), True),
    PerturbationSpec("Sensor Noise (sigma=15)", "High ISO gain noise", lambda f: _noise(f, 15), True),
    PerturbationSpec("In-Plane Rotation (25 deg)", "Tilted board / banked drone", lambda f: _rotate(f, 25), False),
    PerturbationSpec("Far Distance (0.5x scale)", "Drone hovering at 2x distance", lambda f: _small(f, 0.5), False),
    PerturbationSpec("JPEG Compression (q=20)", "Heavy Wi-Fi UDP packet artefacts", lambda f: _jpeg(f, 20), True),
]


def read_frames(video_source: str, step: int = FRAME_STEP) -> list[np.ndarray]:
    """Read every ``step``-th frame; fail fast if the video yields nothing."""
    cap = cv2.VideoCapture(video_source)
    frames, i = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(frame)
            i += 1
    finally:
        cap.release()
    if not frames:
        raise SystemExit(f"error: no frames read from video: {video_source}")
    return frames


def timed_detect(frame: np.ndarray) -> tuple[object, float]:
    """Run detect_board and return (detection, seconds)."""
    t0 = time.perf_counter()
    detection = detect_board(frame)
    return detection, time.perf_counter() - t0


def evaluate_perturbation(
    frames: list[np.ndarray],
    refs: list[object],
    spec: PerturbationSpec,
) -> PerturbationResult:
    """Detection rate, mean centre error vs reference, and ms/frame for one perturbation."""
    hits, errs, total_s = 0, [], 0.0
    for frame, ref in zip(frames, refs):
        det, secs = timed_detect(spec.fn(frame))
        total_s += secs
        if det is None:
            continue
        hits += 1
        if ref is not None and spec.compare_centre:
            errs.append(float(np.hypot(det[1][0] - ref[1][0], det[1][1] - ref[1][1])))
    return PerturbationResult(
        condition=spec.name,
        scenario=spec.scenario,
        compared=spec.compare_centre,
        det_pct=100.0 * hits / len(frames),
        err_px=float(np.mean(errs)) if errs else None,
        ms=1000.0 * total_s / len(frames),
    )


def evaluate_negative_control(frames: list[np.ndarray], refs: list[object]) -> NegativeControlResult:
    """Hide the detected board with a flat colour and count any remaining detections."""
    masked, fp, total_s = 0, 0, 0.0
    for frame, ref in zip(frames, refs):
        if ref is None:
            continue
        hidden = frame.copy()
        cv2.fillConvexPoly(hidden, ref[0].astype(np.int32), NEGATIVE_MASK_BGR)
        det, secs = timed_detect(hidden)
        total_s += secs
        masked += 1
        fp += det is not None
    return NegativeControlResult(
        masked=masked,
        fp=fp,
        fp_pct=100.0 * fp / max(1, masked),
        ms=1000.0 * total_s / max(1, masked),
    )


def plot_task2_summary(
    rows: list[PerturbationResult],
    negative: NegativeControlResult,
    out_file: str,
) -> None:
    """Save a 2-panel horizontal bar chart summarizing Task 2 detection rate and runtime."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    labels = [r.condition for r in rows] + ["Negative (Board Masked)"]
    rates = [r.det_pct for r in rows] + [negative.fp_pct]
    times = [r.ms for r in rows] + [negative.ms]
    y_pos = np.arange(len(labels))

    colors = [
        "#2ca02c" if r.det_pct >= 90 else "#ff7f0e" if r.det_pct >= 50 else "#d62728"
        for r in rows
    ] + ["#1f77b4"]

    bars1 = ax1.barh(y_pos, rates, color=colors, alpha=0.88)
    ax1.set_yticks(y_pos, labels=labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Detection / False-Positive Rate (%)")
    ax1.set_title("Detection Rate Across Perturbations")
    ax1.set_xlim(0, 118)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for i, bar in enumerate(bars1):
        suffix = " (FP)" if i == len(rows) else ""
        ax1.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{rates[i]:.0f}%{suffix}",
            va="center",
            fontsize=8,
        )

    bars2 = ax2.barh(y_pos, times, color="#6baed6", alpha=0.88)
    ax2.set_yticks(y_pos, labels=[""] * len(labels))
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Runtime (ms / frame)")
    ax2.set_title("Frame Processing Latency & Centre Error")
    ax2.set_xlim(0, max(times + [4.0]) * 1.38)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for i, bar in enumerate(bars2):
        err_note = ""
        if i < len(rows) and rows[i].err_px is not None:
            err_note = f"  (err: {rows[i].err_px:.1f} px)"
        ax2.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{times[i]:.2f} ms{err_note}",
            va="center",
            fontsize=8,
        )

    fig.suptitle("Task 2 Summary: Checkerboard Detector Robustness", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def run_task2_benchmarks(video_source: str) -> Task2Report:
    """Benchmark detect_board under perturbations and a negative control."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 78)
    print("TASK 2 BENCHMARK: CHECKERBOARD DETECTOR ROBUSTNESS & FALSE-POSITIVE TEST")
    print("=" * 78)
    frames = read_frames(video_source)
    refs = [detect_board(f) for f in frames]
    print(f"Evaluated on {len(frames)} frames sampled from {map_label(video_source)}\n")
    print(f"{'Condition':<28} {'Detection':>10} {'Centre err':>12} {'ms/frame':>10}")

    rows = []
    for spec in PERTURBATIONS:
        res = evaluate_perturbation(frames, refs, spec)
        err = f"{res.err_px:.1f} px" if res.err_px is not None else "N/A"
        print(f"{res.condition:<28} {res.det_pct:>9.0f}% {err:>12} {res.ms:>10.2f}")
        rows.append(res)

    neg = evaluate_negative_control(frames, refs)
    print(
        f"{'NEGATIVE (board masked)':<28} {neg.fp_pct:>9.0f}% "
        f"{neg.fp:>4} FP / {neg.masked:<4} {neg.ms:>8.2f}"
    )

    summary_plot_rel = "plots/task2_summary.png"
    plot_task2_summary(rows, neg, os.path.join(BENCH_DIR, summary_plot_rel))
    print(f"  -> saved benchmarks/{summary_plot_rel}")
    return Task2Report(
        video=map_label(video_source),
        n_frames=len(frames),
        rows=rows,
        negative=neg,
        summary_plot_rel=summary_plot_rel,
    )


# =========================================================================== #
# Markdown Report Generation
# =========================================================================== #
def md_scorecard_task1(t1: Task1Report) -> list[str]:
    """Measured scorecard rows for Task 1."""
    n_ok, n_fail = len(t1.maps), len(t1.failures)
    formats = sorted({b.mapping.fmt for b in t1.maps.values()})
    load_ms = [b.mapping.load_ms for b in t1.maps.values()]

    by_label = {(a.algo, a.conn): a for a in t1.aggregates}
    a8 = by_label.get(("A*", "8-way"))
    greedy8 = by_label.get(("Greedy", "8-way"))
    rrt = by_label.get(("RRT", "any-angle"))

    saving = explored_saving_vs_dijkstra(t1.maps, "8-way")
    cmd_cut = [b.ablation.cmd_reduction_pct for b in t1.maps.values()]

    a8_solved = f"{a8.solved}/{a8.total}" if a8 else "0/0"
    greedy_gap = f"{greedy8.min_gap:+.1f}% to {greedy8.max_gap:+.1f}%" if greedy8 and greedy8.min_gap is not None else "n/a"
    rrt_gap = f"{rrt.min_gap:+.1f}% to {rrt.max_gap:+.1f}%" if rrt and rrt.min_gap is not None else "n/a"

    return [
        f"| **Mapping** | `load_map` + `inflate({t1.margin_cm:g} cm)` | {', '.join(formats)} | "
        f"{n_ok} maps loaded ({n_fail} failed); load time {format_range(load_ms, '{:.1f}')} ms |",
        f"| **Path Planner** | A\\* 8-way (`plan_path`) | Dijkstra, Greedy, Theta\\*, RRT, RRT\\* | "
        f"Optimal on {a8_solved} maps; explores {format_range(saving)}% fewer cells than Dijkstra; "
        f"Greedy {greedy_gap}, RRT {rrt_gap} cost vs Optimal |",
        f"| **Waypoints / Commands** | LOS smoothing (`plan_path`) | Raw A\\*, collinear pruning | "
        f"`go` commands reduced by {format_range(cmd_cut)}% vs raw A\\* (0 invalid SDK commands) |",
    ]


def md_scorecard_task2(t2: Task2Report) -> list[str]:
    """Measured scorecard row for Task 2."""
    base = t2.rows[0]
    det = [r.det_pct for r in t2.rows]
    errs = [r.err_px for r in t2.rows if r.err_px is not None and r.compared]
    ms = [r.ms for r in t2.rows]
    neg = t2.negative
    return [
        f"| **Board Detector** | `detect_board` (Canny + quad + checker contrast) | 9 perturbations + negative control | "
        f"Baseline {base.det_pct:.0f}%; {format_range(det)}% across perturbations; "
        f"centre error {format_range(errs, '{:.1f}')} px; {statistics.mean(ms):.1f} ms/frame mean; "
        f"{neg.fp} FP / {neg.masked} masked frames |",
    ]


def md_mapping_table(t1: Task1Report) -> list[str]:
    """Per-map mapping statistics table."""
    lines = [
        "| Map | Format | Grid | Cells | Cell (cm) | Load (ms) | Obstacles (Raw) | Obstacles (Inflated) | Inflation Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, bench in t1.maps.items():
        m = bench.mapping
        lines.append(
            f"| `{label}` | `{m.fmt}` | {m.grid_shape} | {m.total_cells} | {m.cell_cm:.1f} | "
            f"{m.load_ms:.1f} | {m.obst_raw:.1f}% | {m.obst_inflated:.1f}% | {m.inflation_note} |"
        )
    for label, err in t1.failures.items():
        lines.append(f"| `{label}` | — | — | — | — | — | — | — | **Load failed:** {err} |")
    return lines + [""]


def md_planner_summary_table(t1: Task1Report) -> list[str]:
    """Aggregate planner comparison table across all benchmarked maps."""
    lines = [
        "| Algorithm | Connectivity | Solved Maps | Mean Grid Explored | Median Runtime (ms) | Mean Runtime (ms) | Mean Gap vs Optimal | Cost Gap Range |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in t1.aggregates:
        is_pipeline = a.algo == "A*" and a.conn == "8-way"
        escaped_algo = a.algo.replace("*", r"\*")
        name = f"**{escaped_algo}**" if is_pipeline else a.algo
        conn = f"**{a.conn}**" if is_pipeline else a.conn
        solved = f"{a.solved}/{a.total}"
        explored = f"{a.mean_pct_grid:.1f}%" if a.mean_pct_grid is not None else "—"
        med_ms = f"{a.median_time_ms:.2f}" if a.median_time_ms is not None else "—"
        mean_ms = f"{a.mean_time_ms:.2f}" if a.mean_time_ms is not None else "—"
        mean_gap = format_gap(a.mean_gap)
        if a.min_gap is None or a.max_gap is None:
            gap_range = "—"
        elif abs(a.max_gap - a.min_gap) < 1e-4:
            gap_range = format_gap(a.min_gap)
        else:
            gap_range = f"{a.min_gap:+.1f}% to {a.max_gap:+.1f}%"
        lines.append(
            f"| {name} | {conn} | {solved} | {explored} | {med_ms} | {mean_ms} | {mean_gap} | {gap_range} |"
        )
    return lines + ["", f"![Task 1 Aggregate Summary]({t1.summary_plot_rel})", ""]


def md_ablation_table(t1: Task1Report) -> list[str]:
    """Waypoint and command reduction table with separate columns and summary row."""
    lines = [
        "| Map | Waypoints (Raw) | Waypoints (Pruned) | Waypoints (Smoothed) | `go` Cmds (Raw) | `go` Cmds (Pruned) | `go` Cmds (Smoothed) | Cmd Reduction | Out-of-Range `go` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, bench in t1.maps.items():
        a = bench.ablation
        r, p, s = a.raw, a.simplified, a.smoothed
        lines.append(
            f"| `{label}` | {r.wps} | {p.wps} | **{s.wps}** | "
            f"{r.cmds} | {p.cmds} | **{s.cmds}** | "
            f"**-{a.cmd_reduction_pct:.0f}%** | {r.invalid} / {p.invalid} / {s.invalid} |"
        )
    if t1.maps:
        abls = [b.ablation for b in t1.maps.values()]
        mean_cut = statistics.mean(a.cmd_reduction_pct for a in abls)
        lines.append(
            f"| **Mean across {len(abls)} maps** | "
            f"{statistics.mean(a.raw.wps for a in abls):.1f} | "
            f"{statistics.mean(a.simplified.wps for a in abls):.1f} | "
            f"**{statistics.mean(a.smoothed.wps for a in abls):.1f}** | "
            f"{statistics.mean(a.raw.cmds for a in abls):.1f} | "
            f"{statistics.mean(a.simplified.cmds for a in abls):.1f} | "
            f"**{statistics.mean(a.smoothed.cmds for a in abls):.1f}** | "
            f"**-{mean_cut:.0f}%** | 0 / 0 / 0 |"
        )
    return lines + [""]


def md_planner_map_section(bench: MapBenchmark) -> list[str]:
    """One map's detailed planner table and comparison plot."""
    m = bench.mapping
    lines = [
        f"#### `{bench.label}` ({m.grid_shape}, {m.total_cells} cells — inflation: {m.inflation_note.lower()})",
        "",
        "| Algorithm | Connectivity | Path Cost (cells) | Explored | % of Grid | Runtime (ms) | vs Optimal |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in bench.planners:
        if not r.found:
            lines.append(f"| {r.algo} | {r.conn} | no path | — | — | — | — |")
            continue
        lines.append(
            f"| {r.algo} | {r.conn} | {r.cost:.2f} | {r.expanded} | {r.pct_grid:.1f}% | "
            f"{r.time_ms:.2f} | {format_gap(r.gap)} |"
        )
    return lines + ["", f"![{bench.label} comparison]({bench.plot_rel})", ""]


def md_task1(t1: Task1Report) -> list[str]:
    """Full Task 1 Markdown section."""
    lines = [
        "## 2. Task 1: Navigation",
        "",
        "### 2.1 Mapping",
        "",
        f"{t1.margin_cm:g} cm safety margin; falls back to raw grid if inflation blocks the 4-way route.",
        "",
    ]
    lines += md_mapping_table(t1)
    lines += [
        "### 2.2 Planner Summary",
        "",
        "Median of 3 runs per map; optimal baseline = Dijkstra (`any-angle` vs 8-way). Pipeline default (**A\\* 8-way**) is bolded.",
        "",
    ]
    lines += md_planner_summary_table(t1)
    lines += [
        "### 2.3 Waypoints & Commands",
        "",
        f"Raw 8-way A\\* vs collinear pruning vs LOS smoothing (`plan_path`). Out-of-range: all axes < {TELLO_GO_MIN_CM} cm or any axis > {TELLO_GO_MAX_CM} cm.",
        "",
    ]
    lines += md_ablation_table(t1)
    lines += [
        "### 2.4 Per-Map Results",
        "",
        "Dark grey = obstacle, light blue = safety margin, orange = path, green = start, red = goal.",
        "",
    ]
    for bench in t1.maps.values():
        lines += md_planner_map_section(bench)
    return lines


def md_task2(t2: Task2Report) -> list[str]:
    """Full Task 2 Markdown section."""
    neg = t2.negative
    lines = [
        "---",
        "",
        "## 3. Task 2: Checkerboard Tracking",
        "",
        f"{t2.n_frames} frames (every {FRAME_STEP}rd frame) from `{t2.video}`. Centre error measured vs unmodified frame.",
        "",
        "| Condition | Simulated Scenario | Detection Rate | Mean Centre Error | Runtime (ms/frame) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in t2.rows:
        err = f"{r.err_px:.1f} px" if r.err_px is not None else "N/A"
        lines.append(f"| {r.condition} | {r.scenario} | {r.det_pct:.0f}% | {err} | {r.ms:.2f} |")
    lines.append(
        f"| **Negative (board masked)** | Board occluded; clutter rejection | "
        f"{neg.fp} FP / {neg.masked} frames ({neg.fp_pct:.0f}%) | — | {neg.ms:.2f} |"
    )
    lines += ["", f"![Task 2 Robustness Summary]({t2.summary_plot_rel})", ""]
    return lines


def write_summary_markdown(
    t1: Task1Report | None,
    t2: Task2Report | None,
    out_path: str = SUMMARY_MD,
) -> None:
    """Write the complete Markdown benchmark summary from measured results."""
    lines = [
        "# Benchmark Summary",
        "",
        f"> Generated on {datetime.now().strftime('%d %b %Y')}",
        "",
        "## 1. Scorecard",
        "",
        "| Stage | Pipeline Method | Alternatives Tested | Key Measured Results |",
        "|---|---|---|---|",
    ]
    if t1:
        lines += md_scorecard_task1(t1)
    if t2:
        lines += md_scorecard_task2(t2)
    lines.append("")
    if t1:
        lines += md_task1(t1)
    if t2:
        lines += md_task2(t2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[saved summary report] {map_label(out_path)}")


# =========================================================================== #
# CLI Entrypoint
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "maps",
        nargs="*",
        help="optional map path(s) for Task 1 (default: representative maps in mapping/maps/)",
    )
    parser.add_argument(
        "--task",
        choices=["all", "1", "2"],
        default="all",
        help="which benchmark to run (default: all = Task 1 + Task 2)",
    )
    parser.add_argument(
        "--margin-cm",
        type=float,
        default=DEFAULT_MARGIN_CM,
        help=f"obstacle safety margin in cm (default: {DEFAULT_MARGIN_CM:g})",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="video for Task 2 (default: data/Tello_video.mp4)",
    )
    args = parser.parse_args()
    if args.task in ("all", "2") and not os.path.isfile(args.video):
        parser.error(f"video not found: {args.video}")
    return args


def main() -> None:
    args = parse_args()
    t1 = (
        run_task1_benchmarks(discover_maps(args.maps), args.margin_cm)
        if args.task in ("all", "1")
        else None
    )
    t2 = run_task2_benchmarks(args.video) if args.task in ("all", "2") else None
    write_summary_markdown(t1, t2, SUMMARY_MD)


if __name__ == "__main__":
    main()
