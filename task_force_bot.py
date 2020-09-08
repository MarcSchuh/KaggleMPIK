from kaggle_environments.envs.halite.helpers import *
import numpy as np


# Boards ###############################################################################################################

# WARNING: The numpy boards use by default the x-axis going down and the y-axis going right, which is unexpected
#          for a game board. Accessing a single element therefore needs to be done by, e.g. board_halite[1, 2],
#          which means the y=1 and x=2 square.

def board_halite_(obs: Dict[str, Any], size: int) -> np.ndarray:
    board_halite = np.array(obs['halite']).reshape(size, -1)
    return board_halite


def board_ships_(obs: Dict[str, Any], size: int) -> np.ndarray:
    board_ships = np.empty((size, size))
    board_ships[:] = np.nan

    for player in range(len(obs['players'])):
        ships_dict = obs['players'][player][2]
        for ship in ships_dict:
            x_pos = ships_dict[ship][0] % size
            y_pos = ships_dict[ship][0] // size
            board_ships[y_pos, x_pos] = player

    return board_ships


def board_shipyards_(obs: Dict[str, Any], size: int) -> np.ndarray:
    board_shipyards = np.empty((size, size))
    board_shipyards[:] = np.nan

    for player in range(len(obs['players'])):
        shipyards_dict = obs['players'][player][1]
        for shipyard in shipyards_dict:
            x_pos = shipyards_dict[shipyard] % size
            y_pos = shipyards_dict[shipyard] // size
            board_shipyards[y_pos, x_pos] = player

    return board_shipyards


# Distance Functions ###################################################################################################

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


class TaskForceBot:

    def __init__(self, size: int, player_id: int, starting_position: int):
        self.size = size
        self.player_id = player_id
        self.starting_position = starting_position

        self.directions_dict = {'NORTH': [-1, 0], 'EAST': [0, 1], 'SOUTH': [1, 0], 'WEST': [0, -1], 'None': [0, 0]}
        self.distance_matrix = self.create_distance_matrix()

        self.max_ships = 35
        self.num_max_ships_per_shipyard = 10

        self.shipyards_min_distance = 4
        self.shipyard_pos_1 = np.nan
        self.shipyard_pos_2 = np.nan

    # Distance Functions ###############################################################################################

    # This function only has to be used once to create the look-up table in the beginning
    def create_distance_matrix(self) -> np.ndarray:
        # Maximal position is self.size**2 - 1
        position_range = np.arange(self.size ** 2)
        positions_1 = np.repeat(position_range, self.size ** 2)
        positions_2 = np.tile(position_range, self.size ** 2)

        # positions_1 and positions_2 combined contain all possible combinations of positions
        x_distances = distance_1d(positions_1 % self.size, positions_2 % self.size, self.size)
        y_distances = distance_1d(positions_1 // self.size, positions_2 // self.size, self.size)

        distance_matrix = (x_distances + y_distances).reshape(self.size ** 2, -1)

        return distance_matrix

    def get_squares_within_radius(self, position: int, radius: int) -> np.ndarray:
        return np.where(self.distance_matrix[position] < radius)[0]

    # Shipyard Placement ###############################################################################################

    def compare_shipyard_positions(self, points_of_interest: np.ndarray) -> int:
        best_position = np.nan
        min_distance_to_start_pos = np.inf
        for point in points_of_interest:
            if self.distance_matrix[self.starting_position, point] < min_distance_to_start_pos:
                min_distance_to_start_pos = self.distance_matrix[self.starting_position, point]
                best_position = point

        return best_position

    def determine_shipyard_positions(self, board_halite: np.ndarray) -> None:
        board_of_interest = (board_halite != 0)
        board_of_interest = board_of_interest.astype(int)

        board_points_of_interest = np.zeros((self.size, self.size))
        # Add direct neighbors with weight 1
        for shift in [1, -1]:
            for axis in [0, 1]:
                board_points_of_interest += np.roll(board_of_interest, shift=shift, axis=axis)

        # Add distance two neighbors with weight 0.5
        for shift in [2, -2]:
            for axis in [0, 1]:
                board_points_of_interest += 0.5 * np.roll(board_of_interest, shift=shift, axis=axis)
        for shift_1 in [-1, 1]:
            for shift_2 in [-1, 1]:
                board_points_of_interest += 0.5 * np.roll(np.roll(board_of_interest, shift=shift_1, axis=0),
                                                          shift=shift_2, axis=1)

        squares_close_to_start = self.get_squares_within_radius(self.starting_position, self.shipyards_min_distance)
        for coordinates in get_coordinates(squares_close_to_start, self.size):
            board_points_of_interest[coordinates[0], coordinates[1]] = 0

        points_of_interest_1 = get_position(
                                   np.column_stack(
                                       np.where(board_points_of_interest == np.max(board_points_of_interest))),
                                   self.size)
        self.shipyard_pos_1 = self.compare_shipyard_positions(points_of_interest_1)

        squares_close_to_pos_1 = self.get_squares_within_radius(self.shipyard_pos_1, self.shipyards_min_distance)
        for coordinates in get_coordinates(squares_close_to_pos_1, self.size):
            board_points_of_interest[coordinates[0], coordinates[1]] = 0

        points_of_interest_2 = get_position(
                                   np.column_stack(
                                       np.where(board_points_of_interest == np.max(board_points_of_interest))),
                                   self.size)
        self.shipyard_pos_2 = self.compare_shipyard_positions(points_of_interest_2)

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
    # print('-----------------------------------------------------------------------')
    # print(obs['step'])
    player_id = obs['player']
    actions = {}

    if obs['step'] == 0:
        # Initialize the bot
        ships_dict = obs['players'][player_id][2]
        starting_position = np.nan
        for ship in ships_dict:
            starting_position = ships_dict[ship][0]

            # Turn starting ship into shipyard
            actions[ship] = 'CONVERT'

        task_force_bot = TaskForceBot(config['size'], player_id, starting_position)

        board_halite = board_halite_(obs, config['size'])
        task_force_bot.determine_shipyard_positions(board_halite)

    return actions
