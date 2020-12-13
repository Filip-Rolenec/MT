from gas_example.enum_types import Action, MothballedState
from gas_example.optimization.optimization import get_best_action, INTEGRAL_SAMPLE_SIZE
from gas_example.optimization.vf import Vf
from gas_example.simulation.state import State

import pandas as pd


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


class HeuristicStrategy:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def get_action(self, state, epoch: int):
        return self.strategy_function(state, epoch)


class OptimalStrategy:

    def __init__(self, path):
        self.vfs = get_vfs_from_path(path)

    def get_action(self, state, epoch):
        return get_best_action(state, INTEGRAL_SAMPLE_SIZE, epoch, self.vfs[epoch + 1])


def get_vfs_from_path(path):
    df_vfs = pd.read_csv(path, index_col=[0])
    vfs = []
    for i in range(len(df_vfs.columns)):
        params = df_vfs[str(i)]
        vfs.append(Vf(params))

    return vfs
