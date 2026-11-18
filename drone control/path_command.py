from tello import Tello
from datetime import datetime
import time
import os
import time
import threading
import math

# Replace with path planner output
path = [(0, 0),(1, 1),(1, 2),(2, 1),(2, 2),(3, 3)]
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
        heading = angle

        # find the best turn direction (clockwise -, counterclockwise +)
        if turn > 180:
            turn = turn - 360
        elif turn < -180:
            turn = turn + 360

        commands.append({"distance": distance, "angle": turn})

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


# for forward navigation only
def main():

    print("Connecting to Tello...")
    # Initialize Tello
    tello = Tello()
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # Connection check, enter SDK mode
    # response = tello.send_command("command")
    # print("Connection response:", response)

    # Generate route
    route = generate_commands(path, cell_size, starting_angle)

    #tello.send_command("takeoff")

    for segment in route:

        angle = segment["angle"]
        distance = segment["distance"]

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

if __name__ == "__main__":
    main()