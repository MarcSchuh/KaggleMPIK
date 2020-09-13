from kaggle_environments.envs.halite.helpers import *
import numpy as np

# Model and Global Parameters ##########################################################################################

size_ = 21

max_ships = 35
num_max_ships_per_shipyard = 10

shipyards_min_distance = 4
starting_position = np.nan
shipyard_pos_1 = np.nan
shipyard_pos_2 = np.nan

directions_dict = {'NORTH': [-1, 0], 'EAST': [0, 1], 'SOUTH': [1, 0], 'WEST': [0, -1], 'None': [0, 0]}


# Board ################################################################################################################

# WARNING: The numpy boards use by default the x-axis going down and the y-axis going right, which is unexpected
#          for a game board. Accessing a single element therefore needs to be done by, e.g. board_halite[1, 2],
#          which means the y=1 and x=2 square.

def board_halite_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_halite = np.array(obs['halite']).reshape(config['size'], -1)
    return board_halite


def board_ships_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_ships = np.empty((config['size'], config['size']))
    board_ships[:] = np.nan

    for player in range(len(obs['players'])):
        ships_dict = obs['players'][player][2]
        for ship in ships_dict:
            x_pos = ships_dict[ship][0] % config['size']
            y_pos = ships_dict[ship][0] // config['size']
            board_ships[y_pos, x_pos] = player

    return board_ships


def board_shipyards_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_shipyards = np.empty((config['size'], config['size']))
    board_shipyards[:] = np.nan

    for player in range(len(obs['players'])):
        shipyards_dict = obs['players'][player][1]
        for shipyard in shipyards_dict:
            x_pos = shipyards_dict[shipyard] % config['size']
            y_pos = shipyards_dict[shipyard] // config['size']
            board_shipyards[y_pos, x_pos] = player

    return board_shipyards


# Distance #############################################################################################################

def get_coordinates(pos: Union[int, np.ndarray], size: int) -> np.ndarray:
    # Coordinates are given by [y, x]
    if type(pos) == int:
        return np.array([pos // size, pos % size])
    else:
        return np.transpose(np.array([pos // size, pos % size]))


def get_position(coordinates: np.ndarray, size: int) -> np.ndarray:
    # Coordinates are given by [y, x]
    if len(coordinates.shape) == 1:
        coordinates = coordinates[np.newaxis, :]
    return coordinates[:, 0] * size + coordinates[:, 1]


def distance_1d(val_1: Union[int, np.ndarray], val_2: Union[int, np.ndarray], size: int) -> Union[int, np.ndarray]:
    min_val = np.fmin(val_1, val_2)
    max_val = np.fmax(val_1, val_2)
    return np.fmin(max_val - min_val, min_val + size - max_val)


# This function only has to be used once to create the look-up table in the beginning
def create_distance_matrix(size: int) -> np.ndarray:
    # Maximal position is size**2 - 1
    position_range = np.arange(size ** 2)
    positions_1 = np.repeat(position_range, size ** 2)
    positions_2 = np.tile(position_range, size ** 2)

    # positions_1 and positions_2 combined contain all possible combinations of positions
    x_distances = distance_1d(positions_1 % size, positions_2 % size, size)
    y_distances = distance_1d(positions_1 // size, positions_2 // size, size)

    distance_matrix_ = (x_distances + y_distances).reshape(size ** 2, -1)

    return distance_matrix_


distance_matrix = create_distance_matrix(size_)


def get_squares_within_radius(position: int, radius: int) -> np.ndarray:
    global distance_matrix
    return np.where(distance_matrix[position] < radius)[0]


# Shipyard Placement ###################################################################################################

def compare_shipyard_positions(points_of_interest: np.ndarray) -> int:
    global distance_matrix
    global starting_position

    best_position = np.nan
    min_distance_to_start_pos = np.inf
    for point in points_of_interest:
        if distance_matrix[starting_position, point] < min_distance_to_start_pos:
            min_distance_to_start_pos = distance_matrix[starting_position, point]
            best_position = point

    return best_position


def determine_shipyard_positions(board_halite: np.ndarray, size: int) -> None:
    global shipyard_pos_1
    global shipyard_pos_2
    global shipyards_min_distance
    global starting_position

    board_of_interest = (board_halite != 0)
    board_of_interest = board_of_interest.astype(int)

    board_points_of_interest = (
                np.roll(board_of_interest, shift=1, axis=0) + np.roll(board_of_interest, shift=-1, axis=0)
                + np.roll(board_of_interest, shift=1, axis=1) + np.roll(board_of_interest, shift=-1, axis=1)
                + 0.5 * (np.roll(board_of_interest, shift=2, axis=0) + np.roll(board_of_interest, shift=-2, axis=0)
                         + np.roll(board_of_interest, shift=2, axis=1) + np.roll(board_of_interest, shift=-2, axis=1)
                         + np.roll(np.roll(board_of_interest, shift=1, axis=0), shift=1, axis=1)
                         + np.roll(np.roll(board_of_interest, shift=1, axis=0), shift=-1, axis=1)
                         + np.roll(np.roll(board_of_interest, shift=-1, axis=0), shift=1, axis=1)
                         + np.roll(np.roll(board_of_interest, shift=-1, axis=0), shift=-1, axis=1)))

    squares_close_to_start = get_squares_within_radius(starting_position, shipyards_min_distance)
    for coordinates in get_coordinates(squares_close_to_start, size):
        board_points_of_interest[coordinates[0], coordinates[1]] = 0

    points_of_interest_1 = get_position(np.column_stack(
                                            np.where(board_points_of_interest == np.max(board_points_of_interest))),
                                        size)
    shipyard_pos_1 = compare_shipyard_positions(points_of_interest_1)

    squares_close_to_pos_1 = get_squares_within_radius(shipyard_pos_1, shipyards_min_distance)
    for coordinates in get_coordinates(squares_close_to_pos_1, size):
        board_points_of_interest[coordinates[0], coordinates[1]] = 0

    points_of_interest_2 = get_position(np.column_stack(
                                            np.where(board_points_of_interest == np.max(board_points_of_interest))),
                                        size)
    shipyard_pos_2 = compare_shipyard_positions(points_of_interest_2)

    return None


# Pathfinder ###########################################################################################################

def pathfinder():
    pass


# Find Task ########################################################################################################

def find_task():
    pass


# Task - Gather Halite #############################################################################################

def task_gather_halite():
    pass


# Task - Attack Enemy ##############################################################################################

def task_attack_enemy():
    pass


# Task - Build Shipyard ############################################################################################

def task_build_shipyard():
    pass


# Task - Return Halite #############################################################################################

def task_return_halite(ship_position: int, shipyards_dict: Dict[str, int]) -> str:
    pass


# Task - Protect Shipyard ###########################################################################################

def task_protect_shipyard():
    pass


# Task - Shipyard Action ###########################################################################################

def task_shipyard_action():
    pass


# Agent ################################################################################################################

def agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    global max_ships
    global num_max_ships_per_shipyard

    global starting_position
    global shipyard_pos_1
    global shipyard_pos_2

    global directions_dict

    # print('-----------------------------------------------------------------------')
    # print(obs['step'])
    player_id = obs['player']
    actions = {}

    if obs['step'] == 1:
        # Initialize global variables that depend on the board
        ships_dict = obs['players'][player_id][2]
        for ship in ships_dict:
            starting_position = ships_dict[ship][0]

        board_halite = board_halite_(obs, config)
        determine_shipyard_positions(board_halite, config['size'])

    return actions
