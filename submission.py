from kaggle_environments.envs.halite.helpers import *
from random import choice
import numpy as np


# Boards #######################################################################################################

def board_halite(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    return np.array(obs['halite']).reshape(config['size'], -1)


def board_ships(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_ships_ = np.empty((config['size'], config['size']))
    board_ships_[:] = np.nan

    for player in range(len(obs['players'])):
        ships_dict = obs['players'][player][2]
        for ship in ships_dict:
            x_pos = ships_dict[ship][0] % config['size']
            y_pos = ships_dict[ship][0] // config['size']
            board_ships_[x_pos, y_pos] = player

    return board_ships_


def board_shipyards(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_shipyards_ = np.empty((config['size'], config['size']))
    board_shipyards_[:] = np.nan

    for player in range(len(obs['players'])):
        shipyards_dict = obs['players'][player][1]
        for shipyard in shipyards_dict:
            x_pos = shipyards_dict[shipyard][0] % config['size']
            y_pos = shipyards_dict[shipyard][0] // config['size']
            board_shipyards_[x_pos, y_pos] = player

    return board_shipyards_


# Distance #####################################################################################################

def get_coordinates(pos: int, size: int) -> np.ndarray:
    return np.array([pos % size, pos // size])


def get_position(coordinates: np.ndarray, size: int) -> int:
    return coordinates[0] + coordinates[1] * size


def distance_1d(val_1: Union[int, np.ndarray], val_2: Union[int, np.ndarray], size: int) -> Union[int, np.ndarray]:
    min_val = np.fmin(val_1, val_2)
    max_val = np.fmax(val_1, val_2)
    return np.fmin(max_val - min_val, min_val + size - max_val)


def manhatten_distance(pos_1: int, pos_2: int, size: int) -> int:
    coordinates_1 = get_coordinates(pos_1, size)
    coordinates_2 = get_coordinates(pos_2, size)

    dx = distance_1d(coordinates_1[0], coordinates_2[0], size)
    dy = distance_1d(coordinates_1[1], coordinates_2[1], size)

    return dx + dy


# This function only has to be used once to create the look-up table in the beginning
def create_distance_matrix(size: int) -> np.ndarray:
    # Maximal position is size**2 - 1
    position_range = np.arange(size ** 2)
    positions_1 = np.repeat(position_range, size ** 2)
    positions_2 = np.tile(position_range, size ** 2)

    # positions_1 and positions_2 combined contain all possible combinations of positions
    x_distances = distance_1d(positions_1 % size, positions_2 % size, size)
    y_distances = distance_1d(positions_1 // size, positions_2 // size, size)

    distance_matrix = (x_distances + y_distances).reshape(size ** 2, -1)

    return distance_matrix


# Score ########################################################################################################

def score(board_halite_: np.ndarray) -> np.ndarray:
    return board_halite_


# Scheduler ####################################################################################################

def scheduler():
    pass


# Pathfinder ###################################################################################################

def pathfinder():
    pass


# Agent ########################################################################################################

def agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    board = Board(obs, config)
    me = board.current_player

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = choice([ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None])

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = None

    return me.next_actions
