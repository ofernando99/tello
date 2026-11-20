import time
import time
import threading
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mapping.mapping import load_map
from path_planning.path_planner_xy import plan_path
from DroneBlocksTelloSimulator.DroneBlocksSimulatorContextManager import DroneBlocksSimulatorContextManager

def generate_instructions(path, cell_size):

    instructions = []

    for i in range(len(path) - 1):

        current = path[i]
        next = path[i + 1]

        grid_x_change = next[0] - current[0]
        grid_y_change = next[1] - current[1]

        x_distance = round(grid_x_change * cell_size)
        y_distance = round(grid_y_change * cell_size)

        instructions.append({"distance_x": x_distance, "distance_y": y_distance})

    return instructions

def generate_commands(route, speed):

    commands = []

    for segment in route:

        x_distance = segment["distance_x"]
        y_distance = segment["distance_y"]

        command = "go" + " " + str(x_distance) + " " + str(y_distance) + " " + str(0) + " " + str(speed)

        commands.append(command)

    return commands

def plot_path(occupancy_grid, path):

    #plot the grid
    plt.imshow(occupancy_grid[::-1], cmap='binary', vmin=0, vmax=1)

    # Plot the path
    x = [point[0] for point in path]
    y = [point[1] for point in path]

    plt.plot(x, y, marker="o")

    plt.scatter(x[0], y[0], color="green", label="Start", s=100) 

    plt.scatter(x[-1], y[-1], color="red", label="Destination", s=100)

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Planned Drone Path")
    plt.legend()
    plt.axis("equal")
    plt.show()

class KeepAlive:
    def __init__(self, tello):
        self.alive = True
        self.lock = threading.Lock()
        self.tello = tello

    def __call__(self):

        ## get alive status
        with self.lock:
            alive = self.alive

        while alive:
            print("Living")
            self.tello.send_command("command")
            time.sleep(3)
            
            ## check if I need to keep living 
            with self.lock:
                alive = self.alive

        print("You killed me")

    def shutdown(self):
        print("Received Shutdown")
        with self.lock:
            self.alive = False

def main():

    starting_angle = 90
    cell_size = 20
    speed = 20

    # load map
    map_path = input("Enter map path : ")
    # connect simulation
    sim_key = input("Enter simulator key (visit : coding-sim.droneblocks.io) : ")
    occupancy_grid, start, destination = load_map(map_path)
    # converting back to x y coordinates
    rows, cols = occupancy_grid.shape
    start = (start[1] , rows - 1 - start[0])
    destination = (destination[1] , rows - 1 - destination[0])
    #plan path
    path = plan_path(occupancy_grid, start, destination, cell_size)

    # Generate route
    route = generate_instructions(path, cell_size)
    # Generate sdk commands
    commands = generate_commands(route, speed)

    #add takeoff and land
    commands.insert(0, "takeoff")
    commands.append("land")
    #print commands
    print("-------Command sequence-------")
    for command in commands: print(command)
    #plot path
    plot_path(occupancy_grid, path)

    print("Connecting to DroneBlocks...")
    with DroneBlocksSimulatorContextManager(simulator_key=sim_key) as drone:
        for command in commands:
            print("Sending command......")
            print(command)
            command = command.split()
            if command[0] == "takeoff":
                drone.takeoff()
            elif command[0] == "land":
                drone.land()
            elif command[0] == "go":
                drone.fly_to_xyz(int(command[2]), int(command[1])* -1, int(command[3]), 'cm')

if __name__ == "__main__":
    main()