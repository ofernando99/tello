import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from path_planning.path_planner import plan_path
from path_planning.test_obstacles import get_test_scenarios
from drone_control_sdk import generate_commands



def simulate_teammate_commands(route: list, start: tuple, cell_size: float, starting_angle: float = 90.0) -> list:
    """Reconstruct the physical (row, col) trajectory produced by drone_control_sdk.generate_commands()."""
    heading = starting_angle
    r, c = float(start[0]), float(start[1])
    sim_path = [(r, c)]

    for seg in route:
        # In teammate's convention: +angle = ccw, -angle = cw; 90 deg = East (+col), 0 deg = South (+row)
        heading = (heading + seg["angle"]) % 360
        direction = seg["direction"]

        if direction == "f":
            move_angle = heading
        elif direction == "b":
            move_angle = heading + 180
        elif direction == "l":
            move_angle = heading + 90
        elif direction == "r":
            move_angle = heading - 90
        else:
            move_angle = heading

        dist_cells = seg["distance"] / cell_size
        r += dist_cells * math.cos(math.radians(move_angle))
        c += dist_cells * math.sin(math.radians(move_angle))
        sim_path.append((r, c))

    return sim_path


def run_teammate_stress_test(cell_size_cm: float = 20.0, starting_angle: float = 90.0):
    scenarios = get_test_scenarios()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    for ax, (title, grid, start, goal) in zip(axes_flat, scenarios):
        waypoints = plan_path(grid, start, goal, cell_size_cm=cell_size_cm)
        route = generate_commands(waypoints, cell_size=cell_size_cm, starting_angle=starting_angle)
        sim_path = simulate_teammate_commands(route, start, cell_size=cell_size_cm, starting_angle=starting_angle)

        print(f"\n=== {title} ===")
        for seg in route:
            print(f"  dir={seg['direction']} | turn={seg['angle']}° | dist={seg['distance']}cm")

        # Corrected X/Y mapping: col -> X, row -> Y
        ax.imshow(grid, cmap="binary", vmin=0, vmax=1, origin="upper")

        wx, wy = [p[1] for p in waypoints], [p[0] for p in waypoints]
        sx, sy = [p[1] for p in sim_path], [p[0] for p in sim_path]

        ax.plot(wx, wy, linestyle="--", linewidth=2, color="tab:blue", label="Planner Waypoints")
        ax.plot(sx, sy, marker="o", linewidth=1.8, color="tab:orange", label="Simulated Flight")
        ax.scatter(start[1], start[0], color="green", s=95, zorder=5, label="Start")
        ax.scatter(goal[1], goal[0], color="red", s=95, zorder=5, label="Goal")

        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
        ax.grid(True, alpha=0.35)

    axes_flat[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent/"path_control_stress_test.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    run_teammate_stress_test(cell_size_cm=20.0, starting_angle=90.0)