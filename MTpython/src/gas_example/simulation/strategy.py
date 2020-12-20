from gas_example.enum_types import Action, MothballedState
from gas_example.optimization_2.vf import Vf
from gas_example.optimization_2.optimization import get_best_action
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



class HeuristicStrategy:
    def __init__(self, strategy_function):
        self.strategy_function = strategy_function

    def get_action(self, state, epoch: int):
        return self.strategy_function(state, epoch)


class OptimalStrategy:

    def __init__(self, path):
        self.vfs = get_vfs_from_path(path)

    def get_action(self, state, epoch):
        return get_best_action(state, epoch, self.vfs[epoch + 1])



def get_vfs_from_path(path):
    df_vfs = pd.read_pickle(path)

    vfs = []
    for column in df_vfs:
        vf = Vf()
        vf.set_params(df_vfs[column])
        vfs.append(vf)

    return vfs
