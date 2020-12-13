import operator
from typing import List

import numpy as np

from gas_example.enum_types import Action, PowerplantState
from gas_example.optimization_2.basis_function import uf_2, uf_2_inv
from gas_example.setup import BORROW_RATE, RISK_FREE_RATE
from gas_example.simulation.state import State, get_valid_actions

INTEGRAL_SAMPLE_SIZE = 10
UTILITY_FUNCTION = uf_2
INVERSE_UTILITY_FUNCTION = uf_2_inv

PRINT_DETAILS_GLOBAL = False


def get_state_utility_pairs(
        sampled_states: List[State],
        epoch: int,
        future_vf):
    state_utility_pairs = {}

    for i, state in enumerate(sampled_states):
        if PRINT_DETAILS_GLOBAL:
            print("Getting best action")
        action = get_best_action(state, epoch, future_vf)
        if PRINT_DETAILS_GLOBAL:
            print("Getting utility realization")
        utility_realization = get_utility_realization(state, action, epoch, future_vf, False)

        state_utility_pairs[state] = utility_realization

    return state_utility_pairs


def get_best_action(state: State, epoch: int, future_vf):
    valid_actions = get_valid_actions(state, epoch)

    exp_utility_per_action = {}
    for action in valid_actions:
        # We would like to compute expected value, we approximate by average of samples.
        sample_integral_values = []
        for i in range(INTEGRAL_SAMPLE_SIZE):
            utility_realization = get_utility_realization(state, action, epoch, future_vf)
            sample_integral_values.append(utility_realization)
        if PRINT_DETAILS_GLOBAL:
            print(sample_integral_values)
        exp_utility_per_action[action] = np.mean(sample_integral_values)

    if PRINT_DETAILS_GLOBAL:
        print(state.to_dict())
        print(f"Spark: {state.get_spark_price()}")
        print(exp_utility_per_action)
        print("\n")

    return max(exp_utility_per_action.items(), key=operator.itemgetter(1))[0]


def get_utility_realization(state: State, action: Action, epoch: int, future_vf, print_details=False):
    new_state, fcf = state.get_new_state_and_fcf(action, epoch)

    future_vf_utility = future_vf.compute_value(new_state)
    future_vf_money_equivalent = INVERSE_UTILITY_FUNCTION(future_vf_utility)
    pce_realization = pce([state.balance + fcf, future_vf_money_equivalent]) - state.balance

    utility_realization = round(UTILITY_FUNCTION(pce_realization), 2)
    if print_details:
        print(f"fcf: {fcf}")
        print(f"future_vf_utility: {future_vf_utility}")
        print(f"future_vf_money_equivalent: {future_vf_money_equivalent}")
        print(f"pce_realization: {pce_realization}")
        print(f"utility_realization: {utility_realization}")
        print("\n")

    return utility_realization


# General pce function, used when the entity that is being optimized is not expected cumulative FCF.
def pce(fcfs):
    balance = 0

    r_b = BORROW_RATE
    r_r = RISK_FREE_RATE

    for fcf in fcfs:
        balance += fcf
        if balance < 0:
            balance = balance * r_b
        else:
            balance = balance * r_r

    if balance < 0:
        return balance / r_b ** (len(fcfs))
    else:
        return balance / r_r ** (len(fcfs))
