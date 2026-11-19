from drone_control.tello import Tello
from datetime import datetime
import time
import os
import time
import threading
import math
import numpy as np
import matplotlib.pyplot as plt

from path_planner_xy import plan_path
# Replace with path occupancy grid output
occupancy_grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
start = (0, 0) #(0x,0y)
destination = (11, 9) #(11x,9y)

path = plan_path(occupancy_grid, start, destination, cell_size_cm=20.0)
cell_size = 20
starting_angle = 90

def generate_commands(path, cell_size, starting_angle):

    commands = []
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
        
        commands.append({"distance": distance, "angle": turn, "direction": direction})

    return commands

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

    print("Connecting to Tello...")
    # Initialize Tello
    tello = Tello()
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Connection check, enter SDK mode
    # response = tello.send_command("command")
    # print("Connection response:", response)

    # Generate route
    route = generate_commands(path, cell_size, starting_angle)

    #tello.send_command("takeoff")

    for segment in route:

        angle = segment["angle"]
        distance = segment["distance"]
        direction = segment["direction"]

        if direction == "f":
            if angle < 0:
                command = "cw " + str(abs(angle))
                command = command.strip()
                print(command)
                #tello.send_command(command)

            elif angle > 0:
                command = "ccw " + str(abs(angle))
                print(command)
                #tello.send_command(command)

            command = "forward " + str(distance)
            print(command)
            #tello.send_command(command)

        elif direction == "l":
            command = "left " + str(distance)
            print(command)
            #tello.send_command(command)

        elif direction == "r":
            command = "right " + str(distance)
            print(command)
            #tello.send_command(command)

        elif direction == "b":
            if angle < 0:
                command = "cw " + str(abs(angle))
                command = command.strip()
                print(command)
                #tello.send_command(command)

            elif angle > 0:
                command = "ccw " + str(abs(angle))
                print(command)
                #tello.send_command(command)

            command = "back " + str(distance)
            print(command)
            #tello.send_command(command)

    #tello.send_command("land")

    # # Save log
    # log = tello.get_log()
    # os.makedirs("log", exist_ok=True)
    # log_filename = f"log/{start_time}.txt"

    # with open(log_filename, "w") as out:
    #     for stat in log:
    #         stat.print_stats()
    #         out.write(stat.return_stats())

    # print(f"Log saved to {log_filename}")

    #plaot the grid
    plt.imshow(occupancy_grid[::-1], cmap='binary', vmin=0, vmax=1)

    # Plot the path
    x = [point[0] for point in path]#Here should be point[1]
    y = [point[1] for point in path]#and here point[0]

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

if __name__ == "__main__":
    main()
