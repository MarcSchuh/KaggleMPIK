from kaggle_environments.envs.halite.helpers import *
import numpy as np


rng = np.random.default_rng(seed=None)


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

    def __init__(self, size: int = np.nan, player_id: int = np.nan, starting_position: int = np.nan):
        self.size = size
        self.player_id = player_id
        self.starting_position = starting_position

        self.directions_dict = {'NORTH': [-1, 0], 'EAST': [0, 1], 'SOUTH': [1, 0], 'WEST': [0, -1], 'None': [0, 0]}
        self.distance_matrix = None

        self.max_ships = 35
        self.num_max_ships_per_shipyard = 10

        self.shipyards_min_distance = 4
        self.shipyard_pos_1 = np.nan
        self.shipyard_pos_2 = np.nan

        self.gather_halite_target_squares = None

        self.blocked_squares = None

        self.board_threatened_squares = None  # For current ship

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

    def get_squares_with_distance(self, position: int, distance: int) -> np.ndarray:
        return np.where(self.distance_matrix[position] == distance)[0]

    # Boards ###########################################################################################################

    def board_enemy_ships_influence_zone_(self, obs: Dict[str, Any]) -> np.ndarray:
        board_enemy_ships_influence_zone = np.zeros((self.size, self.size))
        for player in range(len(obs['players'])):
            if player != self.player_id:
                ships_dict = obs['players'][player][2]
                for ship in ships_dict:
                    for distance_to_ship in range(3):
                        influence_squares = self.get_squares_with_distance(ships_dict[ship][0], distance_to_ship)
                        for influence_square in influence_squares:
                            influence_square_coordinates = get_coordinates(influence_square, self.size)
                            board_enemy_ships_influence_zone[influence_square_coordinates[0],
                                                             influence_square_coordinates[1]] \
                                += 0.5 ** (distance_to_ship + 1)

        return board_enemy_ships_influence_zone

    def board_threatened_squares_(self, obs: Dict[str, Any], my_ship_halite: int) -> np.ndarray:
        board_threatened_squares = np.zeros((self.size, self.size), dtype=bool)

        for player in range(len(obs['players'])):
            if player != self.player_id:
                ships_dict = obs['players'][player][2]
                for ship in ships_dict:
                    ship_halite = ships_dict[ship][1]
                    if my_ship_halite > ship_halite:
                        ship_position = ships_dict[ship][0]
                        threatened_positions = self.get_squares_within_radius(ship_position, 2)
                        for threatened_position in threatened_positions:
                            threatened_coordinates = get_coordinates(threatened_position, self.size)
                            board_threatened_squares[threatened_coordinates[0], threatened_coordinates[1]] = True

        return board_threatened_squares

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

    # Find Task ########################################################################################################

    def threatened(self, ship_position: int, board_threatened_squares: np.ndarray) -> bool:
        ship_coordinates = get_coordinates(ship_position, self.size)
        return board_threatened_squares[ship_coordinates[0], ship_coordinates[1]]

    def find_task(self, obs: Dict[str, Any], ship_position: int, ship_halite: int) -> str:
        board_threatened_squares = self.board_threatened_squares_(obs, ship_halite)
        if self.threatened(ship_position, board_threatened_squares):
            return 'evade_enemy'
        return 'gather_halite'

    # Task Functions ###################################################################################################

    def task_gather_halite(self, board_halite: np.ndarray, board_enemy_ships_influence_zone: np.ndarray,
                           board_shipyards: np.ndarray, ship_position: int) -> int:
        square_attractiveness = np.zeros((self.size, self.size))

        shipyard_counts = np.column_stack(np.unique(board_shipyards[~np.isnan(board_shipyards)], return_counts=True))
        shipyard_coordinates_array = np.column_stack(np.where(board_shipyards == self.player_id))

        if shipyard_counts[shipyard_counts[:, 0] == self.player_id].size == 0:  # Player has no shipyard
            square_attractiveness += (0.25 * board_halite
                                      / (1 + 2 * self.distance_matrix[ship_position].reshape(self.size, -1)))

        else:  # Player has at least one shipyard
            distance_matrix_to_closest_shipyard = np.empty((self.size, self.size))
            distance_matrix_to_closest_shipyard[:] = np.nan

            for shipyard_coordinates in shipyard_coordinates_array:
                shipyard_position = get_position(shipyard_coordinates, self.size)
                distance_matrix_to_closest_shipyard = \
                    np.fmin(distance_matrix_to_closest_shipyard,
                            self.distance_matrix[shipyard_position].reshape(self.size, -1))

            square_attractiveness += (0.25 * board_halite
                                      / (1 + self.distance_matrix[ship_position].reshape(self.size, -1)
                                         + distance_matrix_to_closest_shipyard))

        # Take into account proximity of enemy ships
        square_attractiveness *= 1 - board_enemy_ships_influence_zone

        # Take into account squares that are already being targeted (evil hack)
        square_attractiveness *= 1 - self.gather_halite_target_squares.astype(int)

        gather_halite_target_position = np.argmax(square_attractiveness.flatten())
        gather_halite_target_coordinates = get_coordinates(gather_halite_target_position, self.size)
        self.gather_halite_target_squares[gather_halite_target_coordinates[0],
                                          gather_halite_target_coordinates[1]] = True

        return int(gather_halite_target_position)

    def direction_is_free(self, ship_position: int, direction: str) -> bool:
        ship_coordinates = get_coordinates(ship_position, self.size)
        new_coordinates = (ship_coordinates + self.directions_dict[direction]) % self.size
        return ~self.board_threatened_squares[new_coordinates[0], new_coordinates[1]]

    def task_evade_enemy(self, ship_position: int) -> int:
        free_directions = []
        for direction in self.directions_dict:
            if self.direction_is_free(ship_position, direction):
                free_directions.append(direction)

        try:
            return rng.choice(free_directions)
        except ValueError:  # There is no unthreatened square to retreat to
            # TODO: improve on simple random choice if there is no retreat
            return rng.choice(np.array(['NORTH', 'EAST', 'SOUTH', 'WEST']))

    # Pathfinder #######################################################################################################

    def pathfinder(self, current_position: int, target_position: int) -> Union[str, None]:
        current_coordinates = get_coordinates(current_position, self.size)

        directions_values = {}
        for direction in self.directions_dict:
            directions_values[direction] = (self.distance_matrix[target_position].reshape(self.size, -1)
                                            .item(tuple((current_coordinates
                                                         + self.directions_dict[direction]) % self.size))
                                            - self.distance_matrix[target_position].reshape(self.size, -1)
                                            .item(tuple(current_coordinates)))

        for direction in self.directions_dict:
            if self.blocked_squares.item(
                    tuple((current_coordinates + self.directions_dict[direction]) % self.size)):
                directions_values.pop(direction, None)
            if self.board_threatened_squares.item(
                    tuple((current_coordinates + self.directions_dict[direction]) % self.size)):
                directions_values.pop(direction, None)

        # For the rare occasion that all field are blocked use try and stay put if nothing is possible
        try:
            return min(directions_values, key=directions_values.get)
        except IndexError:
            return None


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


task_force_bot = TaskForceBot()


# Agent ################################################################################################################

def agent(obs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    # print('-----------------------------------------------------------------------')
    # print(obs['step'] + 1)  # +1 to match counter of replay
    player_id = obs['player']
    actions = {}
    player_halite = obs['players'][player_id][0]

    if obs['step'] == 0:
        # Initialize the bot
        ships_dict = obs['players'][player_id][2]
        starting_position = np.nan
        for ship in ships_dict:
            starting_position = ships_dict[ship][0]

            # Turn starting ship into shipyard
            actions[ship] = 'CONVERT'

        task_force_bot.size = config['size']
        task_force_bot.player_id = player_id
        task_force_bot.starting_position = starting_position

        task_force_bot.distance_matrix = task_force_bot.create_distance_matrix()
        task_force_bot.gather_halite_target_squares = np.zeros((task_force_bot.size, task_force_bot.size), dtype=bool)
        task_force_bot.blocked_squares = np.zeros((task_force_bot.size, task_force_bot.size), dtype=bool)

        board_halite = board_halite_(obs, config['size'])
        task_force_bot.determine_shipyard_positions(board_halite)

        return actions

    task_force_bot.gather_halite_target_squares = np.zeros((task_force_bot.size, task_force_bot.size), dtype=bool)
    task_force_bot.blocked_squares = np.zeros((task_force_bot.size, task_force_bot.size), dtype=bool)

    shipyards_dict = obs['players'][player_id][1]
    ships_dict = obs['players'][player_id][2]

    board_halite = board_halite_(obs, task_force_bot.size)
    board_shipyards = board_shipyards_(obs, task_force_bot.size)

    ordered_ships_dict = {key: value for key, value in sorted(ships_dict.items(), key=lambda item: item[1][1],
                                                              reverse=True)}

    for shipyard in shipyards_dict:
        shipyard_position = shipyards_dict[shipyard]
        shipyard_coordinates = get_coordinates(shipyard_position, config['size'])
        if ((not task_force_bot.blocked_squares.item(tuple(shipyard_coordinates)))
                & (player_halite > 500)):
            actions[shipyard] = 'SPAWN'
            task_force_bot.blocked_squares[shipyard_coordinates[0], shipyard_coordinates[1]] = True

    for ship in ordered_ships_dict:
        ship_position = ordered_ships_dict[ship][0]
        ship_halite = ordered_ships_dict[ship][1]
        task_force_bot.board_threatened_squares = task_force_bot.board_threatened_squares_(obs, ship_halite)

        task = task_force_bot.find_task(obs, ship_position, ship_halite)

        if task == 'gather_halite':
            target_position = task_force_bot.task_gather_halite(board_halite,
                                                                task_force_bot.board_enemy_ships_influence_zone_(obs),
                                                                board_shipyards, ship_position)

            ship_action = task_force_bot.pathfinder(ship_position, target_position)

        else:  # task == 'evade_enemy'
            ship_action = task_force_bot.task_evade_enemy(ship_position)

        ship_coordinates = get_coordinates(ship_position, config['size'])
        action_target_coordinates = (ship_coordinates + task_force_bot.directions_dict[ship_action]) % config['size']
        task_force_bot.blocked_squares[action_target_coordinates[0], action_target_coordinates[1]] = True

        if ship_action == 'None':
            ship_action = None

        if ship_action is not None:
            actions[ship] = ship_action

    return actions
