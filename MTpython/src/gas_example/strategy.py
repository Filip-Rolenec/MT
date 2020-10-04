from gas_example.enum_types import Action
from gas_example.state import State


def heuristic_strategy_0(state: State, epoch: int):
    # In the first time epoch we build the powerplant.
    if epoch == 0:
        return Action.RUN_AND_BUILD
    # Else we just run it
    else:
        return Action.RUN


def heuristic_strategy_1(state: State, epoch: int):
    # In the first time epoch we build the powerplant.
    if epoch == 0:
        return Action.RUN_AND_BUILD
    # If this epoch would be profitable we will run, we might end up in a loss.
    if state.power_price - (state.gas_price + state.co2_price) > 0:
        return Action.RUN
    # If the run is not profitable, we are not running the plant
    else:
        return Action.DO_NOTHING


class Strategy:
    def __init__(self, strategy_function):
        self.strategy = strategy_function

    def get_action(self, state, epoch: int):
        return self.strategy(state, epoch)
