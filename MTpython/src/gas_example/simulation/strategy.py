from gas_example.enum_types import Action, MothballedState
from gas_example.simulation.state import State


def heuristic_strategy_function_0(state: State, epoch: int):
    # In the first time epoch we build the powerplant.
    if epoch == 0:
        return Action.IDLE_AND_BUILD
    # Else we just run it
    else:
        return Action.RUN


def heuristic_strategy_function_1(state: State, epoch: int):
    # In the first time epoch we build the powerplant.
    if epoch == 0:
        return Action.IDLE_AND_BUILD
    # If this epoch is profitable run it.
    if state.power_price - (state.gas_price + state.co2_price) > 0:
        return Action.RUN
    # If the run is not profitable, we are not running the plant
    else:
        return Action.DO_NOTHING


def heuristic_strategy_function_2(state: State, epoch: int):
    # In the first time epoch we build the powerplant.
    if epoch == 0:
        return Action.IDLE_AND_BUILD
    # If this epoch is profitable run it.
    if state.power_price - (state.gas_price + state.co2_price) > 0:
        return Action.RUN
    # If there is a sequence of negative spark prices mothball, if sequence of positive unmothball

    # Mothball with a hysteresis effect
    if state.get_spark_price() < -5 and state.mothballed_state == MothballedState.NORMAL:
        return Action.MOTHBALL_CHANGE

    if state.get_spark_price() > 5 and state.mothballed_state == MothballedState.MOTHBALLED:
        return Action.MOTHBALL_CHANGE

    else:
        return Action.DO_NOTHING


class Strategy:
    def __init__(self, strategy_function):
        self.strategy = strategy_function

    def get_action(self, state, epoch: int):
        return self.strategy(state, epoch)
