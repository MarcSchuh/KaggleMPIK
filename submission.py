from kaggle_environments.envs.halite.helpers import *
import numpy as np


# Boards #######################################################################################################

# WARNING: The numpy boards use by default the x-axis going down and the y-axis going right, which is unexpected
#          for a game board. Accessing a single element therefore needs to be done by, e.g. board_halite[1, 2],
#          which means the y=1 and x=2 square.

def board_halite_(obs: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    board_halite = np.array(obs['halite']).reshape(config['size'], -1)
    return np.transpose(board_halite)


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


# Distance #####################################################################################################

def get_coordinates(pos: int, size: int) -> np.ndarray:
    # Coordinates are given by [y, x]
    return np.array([pos // size, pos % size])


def get_position(coordinates: np.ndarray, size: int) -> int:
    # Coordinates are given by [y, x]
    return coordinates[0] * size + coordinates[1]


def distance_1d(val_1: Union[int, np.ndarray], val_2: Union[int, np.ndarray], size: int) -> Union[int, np.ndarray]:
    min_val = np.fmin(val_1, val_2)
    max_val = np.fmax(val_1, val_2)
    return np.fmin(max_val - min_val, min_val + size - max_val)


def manhatten_distance(pos_1: int, pos_2: int, size: int) -> int:
    coordinates_1 = get_coordinates(pos_1, size)
    coordinates_2 = get_coordinates(pos_2, size)

    dy = distance_1d(coordinates_1[0], coordinates_2[0], size)
    dx = distance_1d(coordinates_1[1], coordinates_2[1], size)

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

def score(board_halite: np.ndarray, board_shipyards: np.ndarray, ship: list,
          player_id: int, size: int, distance_matrix: np.ndarray) -> np.ndarray:
    halite_per_turn = np.zeros((size, size))

    ship_position = ship[0]
    ship_halite = ship[1]

    shipyard_counts = np.column_stack(np.unique(board_shipyards[~np.isnan(board_shipyards)], return_counts=True))
    shipyard_coordinates_array = np.column_stack(np.where(board_shipyards == player_id))

    if shipyard_counts[shipyard_counts[:, 0] == player_id].size == 0:  # Player has no shipyard
        halite_per_turn += 0.25 * board_halite / (1 + 2 * distance_matrix[ship_position].reshape(size, -1))

    else:  # Player has at least one shipyard
        distance_matrix_to_closest_shipyard = np.empty((size, size))
        distance_matrix_to_closest_shipyard[:] = np.nan

        for shipyard_coordinates in shipyard_coordinates_array:
            shipyard_position = get_position(shipyard_coordinates, size)
            distance_matrix_to_closest_shipyard = np.fmin(distance_matrix_to_closest_shipyard,
                                                          distance_matrix[shipyard_position].reshape(size, -1))

            halite_per_turn[shipyard_coordinates[0], shipyard_coordinates[1]] = \
                ship_halite / (distance_matrix[ship_position, shipyard_position] + 1)

        halite_per_turn += 0.25 * board_halite / (1 + distance_matrix[ship_position].reshape(size, -1)
                                                  + distance_matrix_to_closest_shipyard)
    return halite_per_turn


# Scheduler ####################################################################################################

def scheduler():
    pass


# Pathfinder ###################################################################################################

def pathfinder(current_position: int, target_position: int, blocked_squares: np.ndarray,
               distance_matrix: np.ndarray, size: int) -> Union[str, None]:
    current_coordinates = get_coordinates(current_position, size)

    directions_dict = {'NORTH': [-1, 0], 'EAST': [0, 1], 'SOUTH': [1, 0], 'WEST': [0, -1], 'None': [0, 0]}

    directions_values = {}
    for direction in directions_dict:
        directions_values[direction] = (distance_matrix[target_position].reshape(size, -1)
                                        .item(tuple((current_coordinates + directions_dict[direction]) % size))
                                        - distance_matrix[target_position].reshape(size, -1)
                                        .item(tuple(current_coordinates)))

    for direction in directions_dict:
        if blocked_squares.item(tuple(tuple((current_coordinates + directions_dict[direction]) % size))):
            directions_values.pop(direction, None)

    directions_values_sorted = {key: value for key, value in
                                sorted(directions_values.items(), key=lambda item: item[1])}

    print(current_position)
    print(target_position)
    print(list(directions_values_sorted)[0])

    # For the rare occasion that all field are blocked use try and stay put if nothing is possible
    try:
        return list(directions_values_sorted)[0]
    except IndexError:
        return 'None'


# Agent ########################################################################################################

def agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    print('-----------------------------------------------------------------------')
    print(obs['step'])
    player_id = obs['player']
    actions = {}
    blocked_squares = np.zeros((config['size'], config['size']), dtype=bool)
    directions_dict = {'NORTH': [-1, 0], 'EAST': [0, 1], 'SOUTH': [1, 0], 'WEST': [0, -1], 'None': [0, 0]}

    distance_matrix = create_distance_matrix(config['size'])

    board_halite = board_halite_(obs, config)
    # board_ships = board_ships_(obs, config)
    board_shipyards = board_shipyards_(obs, config)

    shipyards_dict = obs['players'][player_id][1]
    ships_dict = obs['players'][player_id][2]
    # Sort ships by halite content to give ship with most halite highest priority for its action
    ordered_ships_dict = {key: value for key, value in sorted(ships_dict.items(), key=lambda item: item[1][1],
                                                              reverse=True)}

    if obs['step'] == 0:
        # Build a shipyard in the first turn
        for ship in ordered_ships_dict:
            actions[ship] = 'CONVERT'

    # Spawn ships in the first 10 rounds
    elif obs['step'] <= 10:
        for shipyard in shipyards_dict:
            actions[shipyard] = 'SPAWN'

        for ship in ordered_ships_dict:
            ship_position = ordered_ships_dict[ship][0]
            ship_score = score(board_halite, board_shipyards, ordered_ships_dict[ship], obs['player'],
                               config['size'], distance_matrix)
            print(ship_score.round(1))
            target_coordinates = np.array(np.unravel_index(np.argmax(ship_score), ship_score.shape))
            target_position = get_position(target_coordinates, config['size'])

            ship_action = pathfinder(ship_position, target_position, blocked_squares, distance_matrix, config['size'])

            ship_coordinates = get_coordinates(ship_position, config['size'])
            action_target_coordinates = (ship_coordinates + directions_dict[ship_action]) % config['size']
            blocked_squares[action_target_coordinates[0], action_target_coordinates[1]] = True

            if ship_action == 'None':
                ship_action = None

            if ship_action is not None:
                actions[ship] = ship_action

    else:
        # for shipyard in shipyards_dict:
        #     actions[shipyard] = None

        for ship in ordered_ships_dict:
            ship_position = ordered_ships_dict[ship][0]
            ship_score = score(board_halite, board_shipyards, ordered_ships_dict[ship], obs['player'],
                               config['size'], distance_matrix)
            target_coordinates = np.array(np.unravel_index(np.argmax(ship_score), ship_score.shape))
            target_position = get_position(target_coordinates, config['size'])

            ship_action = pathfinder(ship_position, target_position, blocked_squares, distance_matrix, config['size'])

            ship_coordinates = get_coordinates(ship_position, config['size'])
            action_target_coordinates = (ship_coordinates + directions_dict[ship_action]) % config['size']
            blocked_squares[action_target_coordinates[0], action_target_coordinates[1]] = True

            if ship_action == 'None':
                ship_action = None

            if ship_action is not None:
                actions[ship] = ship_action

    return actions
