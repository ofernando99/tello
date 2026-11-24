from mapping.mapping import load_map
from path_planning.path_planner import plan_path
from drone_control.path_command import navigation
#===============================================#
# import tracking
#===============================================#

def main():

    starting_angle = 0
    cell_size = 20
    speed = 20

    map_path = input("Enter map path : ")
    # load map
    occupancy_grid, start, destination = load_map(map_path)
    # plan path
    path = plan_path(occupancy_grid, start, destination, cell_size)
    # send commands to drone
    navigation(cell_size, starting_angle, speed, occupancy_grid, path)
    # sending commands from takeoff to end point (without landing command)
    
    # video streaming and tracking
    #=====================================#

    # Ganbat #

    #=====================================#

if __name__ == "__main__":
    main()
