import time
import math
import threading
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mapping.mapping import load_map
from path_planning.path_planner import plan_path
from DroneBlocksTelloSimulator.DroneBlocksSimulatorContextManager import DroneBlocksSimulatorContextManager

def generate_instructions(path, cell_size, starting_angle):

    theta = math.radians(starting_angle)

    instructions = []

    for i in range(len(path) - 1):

        current = path[i]
        next = path[i + 1]

        grid_x_change = next[0] - current[0]
        grid_y_change = next[1] - current[1]

        x_distance = grid_x_change * cell_size
        y_distance = grid_y_change * cell_size

        # transformation
        x = round(x_distance * math.cos(theta) - y_distance * math.sin(theta))
        y = round(-x_distance * math.sin(theta) - y_distance * math.cos(theta))

        instructions.append({"distance_x": x, "distance_y": y})

    return instructions

def generate_commands(route, speed):

    commands = []

    for segment in route:

        x_distance = segment["distance_x"]
        y_distance = segment["distance_y"]

        command = "go" + " " + str(x_distance) + " " + str(y_distance) + " " + str(0) + " " + str(speed)

        commands.append(command)

    return commands

def transform(occupancy_grid, path):
    trasform = []
    rows = occupancy_grid.shape[0]
    for coordinate in path:
        trasform.append((rows - 1 - coordinate[0], coordinate[1]))
    return trasform

def plot_path(occupancy_grid, path):

    #plot the grid
    plt.imshow(occupancy_grid[::-1], cmap='binary', vmin=0, vmax=1)

    # Plot the path (x vertical y horizontal)
    y = [point[0] for point in path]
    x = [point[1] for point in path]

    plt.plot(x, y, marker="o")

    plt.scatter(x[0], y[0], color="green", label="Start", s=100) 

    plt.scatter(x[-1], y[-1], color="red", label="Destination", s=100)

    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xlabel("Y")
    plt.ylabel("X")
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

    starting_angle = 0
    cell_size = 20
    speed = 20

    map_path = input("Enter map path : ")
    # connect simulation
    sim_key = input("Enter simulator key (visit : coding-sim.droneblocks.io) : ")
    # load map
    occupancy_grid, start, destination = load_map(map_path)
    print(occupancy_grid)
    print(start)
    print(destination)
    # plan path
    path = plan_path(occupancy_grid, start, destination, cell_size)

    # converting path into ground cordinates frame
    path = transform(occupancy_grid, path)
    # Generate route
    route = generate_instructions(path, cell_size, starting_angle)
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
                drone.fly_to_xyz(int(command[1]), int(command[2]), int(command[3]), 'cm')

if __name__ == "__main__":
    main()