import time
import numpy as np
import matplotlib.pyplot as plt
from path_planner import astar, smooth_waypoints_los, waypoints_to_drone_commands


def get_test_scenarios() -> list:
    """Return 6 distinct 10x12 obstacle maps to stress-test A* and LoS smoothing."""
    scenarios = []

    # 1. Slalom / Alternating Walls (tests wide S-curve turns)
    grid_slalom = np.zeros((10, 12), dtype=int)
    grid_slalom[0:7, 3] = 1
    grid_slalom[3:10, 6] = 1
    grid_slalom[0:7, 9] = 1
    scenarios.append(("1. Slalom Walls", grid_slalom, (0, 0), (9, 11)))

    # 2. U-Trap / Dead-End Pocket (tests escaping heuristic cul-de-sacs)
    grid_utrap = np.zeros((10, 12), dtype=int)
    grid_utrap[2, 3:8] = 1
    grid_utrap[2:7, 7] = 1
    grid_utrap[6, 3:8] = 1
    scenarios.append(("2. U-Trap Cul-de-Sac", grid_utrap, (4, 1), (4, 10)))

    # 3. Diagonal Pinches (tests corner-cutting prevention between touching diagonals)
    grid_diag = np.zeros((10, 12), dtype=int)
    for r, c in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (6, 8), (7, 9)]:
        grid_diag[r, c] = 1
    scenarios.append(("3. Diagonal Corner Pinches", grid_diag, (0, 0), (9, 11)))

    # 4. Offset Chicane Gates (tests full-width arena traverses)
    grid_chicane = np.zeros((10, 12), dtype=int)
    grid_chicane[3, 0:9] = 1   # Gate on far right (cols 9-11)
    grid_chicane[6, 3:12] = 1  # Gate on far left (cols 0-2)
    scenarios.append(("4. Offset Chicane Gates", grid_chicane, (0, 0), (9, 11)))

    # 5. Scattered Pillars (tests LoS clearance around cluttered poles/cones)
    grid_pillars = np.zeros((10, 12), dtype=int)
    for r, c in [(1, 3), (1, 8), (3, 1), (3, 5), (3, 9), (5, 3), (5, 7), (7, 1), (7, 5), (7, 9)]:
        grid_pillars[r, c] = 1
    scenarios.append(("5. Scattered Pillars", grid_pillars, (0, 0), (9, 11)))

    # 6. Central Fortress + Flanking Blocks (tests bulky obstacle routing)
    grid_fortress = np.zeros((10, 12), dtype=int)
    grid_fortress[3:7, 4:8] = 1
    grid_fortress[1:3, 2] = 1
    grid_fortress[7:9, 9] = 1
    scenarios.append(("6. Central Blockade", grid_fortress, (0, 0), (9, 11)))

    return scenarios


def run_obstacle_stress_test(cell_size_cm: float = 30.0, drone_clearance_cm: float = 18.0):
    scenarios = get_test_scenarios()
    margin_cells = max(0.85, drone_clearance_cm / cell_size_cm)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for ax, (title, grid, start, goal) in zip(axes_flat, scenarios):
        raw_path, stats = astar(grid, start, goal, allow_diagonal=True)
        waypoints = smooth_waypoints_los(grid, raw_path, margin_cells=margin_cells)
        cmds = waypoints_to_drone_commands(waypoints, cell_size_cm=cell_size_cm, unit="m")

        print(f"{title:26s} | Cost: {stats['cost']:5.2f} | Waypoints: {len(waypoints)} | Commands (deg, m): {cmds}")

        # Plot occupancy grid
        ax.imshow(grid, cmap="Greys", origin="upper")

        # Plot raw A* cell-by-cell path (dashed blue)
        pr, pc = zip(*raw_path)
        ax.plot(pc, pr, linestyle="--", linewidth=1.5, color="tab:blue", label="Raw A* Path")

        # Plot smoothed flight waypoints (solid orange)
        wr, wc = zip(*waypoints)
        ax.plot(wc, wr, marker="o", linewidth=2.2, color="tab:orange", label="Flight Waypoints")

        # Plot start and destination markers
        ax.scatter(start[1], start[0], color="green", s=95, zorder=5, label="Start")
        ax.scatter(goal[1], goal[0], color="blue", s=95, zorder=5, label="Goal")

        ax.set_title(
            f"{title}\nCost: {stats['cost']} | Turns: {len(waypoints) - 2} | {stats['time_ms']}ms",
            fontsize=10,
        )
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, alpha=0.3)

    axes_flat[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig("obstacle_stress_test.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    run_obstacle_stress_test(cell_size_cm=30.0, drone_clearance_cm=18.0)
