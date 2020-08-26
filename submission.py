from kaggle_environments.envs.halite.helpers import *
from random import choice

# Variables
###################################################################################################
max_ships = 20
max_shipyards = 5


# Score ###################################################################################################


# Scheduler ###################################################################################################

#Priosierung der Schiffe
def shipyard_plan(turn, halite, ships): # decide if a shipyard spawns a new ship
    if halite > 500 & ships <= max_ships:
        return "SPAWN"
    else:
        return "NONE"


# Pathfinder ###################################################################################################


# Agent ###################################################################################################

def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player
    player = obs.player
    size = config.size
    board_halite = obs.halite
    current_halite, shipyards, ship_items = obs.players[player]
    shipyard_ids = list(shipyards.keys())
    shipyards = list(shipyards.values())
    ship_number = len(ship_items)
    turn = board.step

    # Set actions for each ship
    for ship in me.ships:
        ship.next_action = choice([ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None])

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard.next_action = shipyard_plan(turn, current_halite, ship_number)

    return me.next_actions

