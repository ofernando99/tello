import time
import time
import threading
import math
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from mapping.mapping import load_map
from path_planning.path_planner_xy import plan_path
from DroneBlocksTelloSimulator.DroneBlocksSimulatorContextManager import DroneBlocksSimulatorContextManager

#===================================================
sim_key = '9cfd8715-1940-4fb4-9ea5-e4c9a6eb5f5f'
#===================================================

def generate_instructions(path, cell_size, starting_angle):

    instructions = []
    heading = starting_angle

    for i in range(len(path) - 1):

        current = path[i]
        next = path[i + 1]

        grid_x_change = next[0] - current[0]
        grid_y_change = next[1] - current[1]

        x_distance = grid_x_change * cell_size
        y_distance = grid_y_change * cell_size

        distance = round(math.sqrt(x_distance**2 + y_distance**2))
        angle = round(math.degrees(math.atan2(y_distance, x_distance)))

        turn = angle - heading
        
        # find the best turn direction (clockwise -, counterclockwise +)
        if turn > 180:
            turn = turn - 360
        elif turn < -180:
            turn = turn + 360

        #find the best movement direction
        if -90 < turn < 90:
            direction = "f"
            heading = angle

        elif turn == -90:
            direction = "r"
            turn = 0

        elif turn == 90:
            direction = "l"
            turn = 0

        else:
            direction = "b"
            turn = turn + 180 if turn < -90 else turn - 180
            heading = angle + 180 if angle < 0 else angle - 180
        
        instructions.append({"distance": distance, "angle": turn, "direction": direction})

    return instructions

def generate_commands(route):

    commands = []

    for segment in route:

        angle = segment["angle"]
        distance = segment["distance"]
        direction = segment["direction"]

        if direction == "f":
            if angle < 0:
                command = "cw " + str(abs(angle))
                command = command.strip()
                commands.append(command)

            elif angle > 0:
                command = "ccw " + str(abs(angle))
                commands.append(command)

            command = "forward " + str(distance)
            commands.append(command)

        elif direction == "l":
            command = "left " + str(distance)
            commands.append(command)

        elif direction == "r":
            command = "right " + str(distance)
            commands.append(command)

        elif direction == "b":
            if angle < 0:
                command = "cw " + str(abs(angle))
                command = command.strip()
                commands.append(command)

            elif angle > 0:
                command = "ccw " + str(abs(angle))
                commands.append(command)

            command = "back " + str(distance)
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
    commands = generate_commands(route)

    #add takeoff and land
    commands.insert(0, "takeoff")
    commands.append("land")
    #print commands
    print("-------Command sequence-------")
    for command in commands: print(command)
    #plot path
    plot_path(occupancy_grid, path)

    # Manual check before sending commands
    proceed = input("Start sending commands (y/n) : ")

    if proceed == "y":
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
                elif command[0] == "forward":
                    drone.fly_forward(int(command[1]), 'cm')
                elif command[0] == "back":
                    drone.fly_backward(int(command[1]), 'cm')
                elif command[0] == "left":
                    drone.fly_left(int(command[1]), 'cm')
                elif command[0] == "right":
                    drone.fly_right(int(command[1]), 'cm')
                elif command[0] == "cw":
                    drone.yaw_right(int(command[1]))
                elif command[0] == "ccw":
                    drone.yaw_left(int(command[1]))

    else:

        print("Existed without execution....")

if __name__ == "__main__":
    main()
