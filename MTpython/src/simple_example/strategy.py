import numpy as np


def get_action_from_strategy(strategy, state, time_epoch):
    return strategy(state, time_epoch)


def heuristic_strategy_1(state, time_epoch):
    return 0


def heuristic_strategy_0(state, time_epoch):
    return 1


def heuristic_strategy_2(state, time_epoch):
    return np.random.choice(np.arange(0, 2), p=[0.5, 0.5])


def optimal_strategy(state, time_epoch):  # Obtained from the algorithm in the middle of this file
    if state == 1:
        return 0
    else:
        return 1
