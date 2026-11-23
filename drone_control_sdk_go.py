import os
import time
import math
import threading
import matplotlib.pyplot as plt
from datetime import datetime

from mapping.mapping import load_map
from path_planning.path_planner_xy import plan_path
from drone_control.tello import Tello

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
        x = round(x_distance * math.cos(theta) + y_distance * math.sin(theta))
        y = round(-x_distance * math.sin(theta) + y_distance * math.cos(theta))

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

    #load map
    occupancy_grid, start, destination = load_map(input("Enter map path : "))
    # converting back to x y coordinates
    rows, cols = occupancy_grid.shape
    start = (start[1] , rows - 1 - start[0])
    destination = (destination[1] , rows - 1 - destination[0])
    #plan path
    path = plan_path(occupancy_grid, start, destination, cell_size)

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

    print("Connecting to Tello...")
    # Initialize Tello
    tello = Tello()
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Connection check, enter SDK mode
    tello.send_command("command")
    # battery level check
    tello.send_command("battery?")

    # Manual check before sending commands
    proceed = input("Start sending commands (y/n) : ")

    if proceed == "y":

        for command in commands:
            print("Sending command......")
            print(command)
            tello.send_command(command)

        # Save log
        log = tello.get_log()
        os.makedirs("log", exist_ok=True)
        log_filename = f"log/{start_time}.txt"

        with open(log_filename, "w") as out:
            for stat in log:
                stat.print_stats()
                out.write(stat.return_stats())

        print(f"Log saved to {log_filename}")

    else:

        print("Existed without execution....")

if __name__ == "__main__":
    main()
