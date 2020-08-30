from kaggle_environments.envs.halite.helpers import *
import numpy as np


# Boards #######################################################################################################

def board_halite_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    return np.array(obs['halite']).reshape(config['size'], -1)


def board_ships_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_ships = np.empty((config['size'], config['size']))
    board_ships[:] = np.nan

    for player in range(len(obs['players'])):
        ships_dict = obs['players'][player][2]
        for ship in ships_dict:
            x_pos = ships_dict[ship][0] % config['size']
            y_pos = ships_dict[ship][0] // config['size']
            board_ships[x_pos, y_pos] = player

    return board_ships


def board_shipyards_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_shipyards = np.empty((config['size'], config['size']))
    board_shipyards[:] = np.nan

    for player in range(len(obs['players'])):
        shipyards_dict = obs['players'][player][1]
        for shipyard in shipyards_dict:
            x_pos = shipyards_dict[shipyard] % config['size']
            y_pos = shipyards_dict[shipyard] // config['size']
            board_shipyards[x_pos, y_pos] = player

    return board_shipyards


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

def score(board_halite: np.ndarray, ship_position: int, size: int, distance_matrix: np.ndarray) -> np.ndarray:
    # Halite per turn with ship starting from shipyard
    return board_halite / (1 + 2 * distance_matrix[ship_position].reshape(size, -1))


# Scheduler ####################################################################################################

def scheduler():
    pass


# Pathfinder ###################################################################################################

def pathfinder(current_position: int, target_position: int, size: int) -> str:
    current_coordinates = get_coordinates(current_position, size)
    target_coordinates = get_coordinates(target_position, size)

    return 'NORTH'


# Agent ########################################################################################################

def agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    player_id = obs['player']
    actions = {}

    distance_matrix = create_distance_matrix(config['size'])

    board_halite = board_halite_(obs, config)
    board_ships = board_ships_(obs, config)
    board_shipyards = board_shipyards_(obs, config)

    shipyards_dict = obs['players'][player_id][1]
    ships_dict = obs['players'][player_id][2]

    if obs['step'] == 0:
        # Build a shipyard in the first turn
        for ship in ships_dict:
            actions[ship] = 'CONVERT'

    # Spawn ships in the first 10 rounds
    elif obs['step'] <= 10:
        for shipyard in shipyards_dict:
            actions[shipyard] = 'SPAWN'

        for ship in ships_dict:
            ship_position = ships_dict[ship][0]
            ship_score = score(board_halite, ship_position, config['size'], distance_matrix)
            target_coordinates = np.array(np.unravel_index(np.argmax(ship_score), ship_score.shape))
            target_position = get_position(target_coordinates, config['size'])
            actions[ship] = pathfinder(ship_position, target_position, config['size'])

    else:
        # for shipyard in shipyards_dict:
        #     actions[shipyard] = None

        for ship in ships_dict:
            ship_position = ships_dict[ship][0]
            ship_score = score(board_halite, ship_position, config['size'], distance_matrix)
            target_coordinates = np.array(np.unravel_index(np.argmax(ship_score), ship_score.shape))
            target_position = get_position(target_coordinates, config['size'])
            actions[ship] = pathfinder(ship_position, target_position, config['size'])

    return actions
