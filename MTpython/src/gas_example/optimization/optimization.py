import operator
from typing import List

import numpy as np

from gas_example.enum_types import Action, PowerplantState
from gas_example.optimization.basis_function import uf_2, uf_2_inv
from gas_example.setup import INTEGRAL_SAMPLE_SIZE, BORROW_RATE_EPOCH, RISK_FREE_RATE_EPOCH
from gas_example.simulation.state import State, get_valid_actions

import seaborn as sns

sns.set()

UTILITY_FUNCTION = uf_2
INVERSE_UTILITY_FUNCTION = uf_2_inv

PRINT_DETAILS_GLOBAL = False


def get_state_utility_pairs(
        sampled_states: List[State],
        future_vf):
    state_utility_pairs = {}

    for i, state in enumerate(sampled_states):
        if PRINT_DETAILS_GLOBAL:
            print("Getting best action")
        action, exp_utility = get_best_action(state, future_vf)


        state_utility_pairs[state] = exp_utility

    return state_utility_pairs


def get_best_action(state: State, future_vf, print_details=False):
    valid_actions = get_valid_actions(state)

    exp_utility_per_action = {}
    for action in valid_actions:
        # We would like to compute expected value, we approximate by average of samples.
        sample_integral_dict = {}
        for i in range(INTEGRAL_SAMPLE_SIZE):
            utility_realization = get_utility_realization(state, action, future_vf)
            sample_integral_dict[i] = utility_realization

        # plt.hist(sample_integral_dict.values())
        # plt.show()
        exp_utility_per_action[action] = np.mean(list(sample_integral_dict.values()))

    if PRINT_DETAILS_GLOBAL or print_details:
        print(state.to_dict())
        print(f"Spark: {state.get_spark_price()}")
        print(exp_utility_per_action)
        print("\n")

    best_action = max(exp_utility_per_action.items(), key=operator.itemgetter(1))[0]

    return best_action, exp_utility_per_action[best_action]


def get_utility_realization(state: State, action: Action, future_vf, print_details=False):
    new_state, fcf = state.get_new_state_and_fcf(action)
    future_vf_utility = future_vf.compute_value(new_state)
    future_vf_money_equivalent = INVERSE_UTILITY_FUNCTION(future_vf_utility)
    pce_realization = pce([state.balance + fcf, future_vf_money_equivalent]) - state.balance

    utility_realization = round(UTILITY_FUNCTION(pce_realization), 2)

    if False & (state.plant_state == PowerplantState.NOT_BUILT) & (utility_realization < 0) & (action != Action.IDLE_AND_BUILD):
        print("Negative value spotted")
        print(state.to_dict())
        print(f"Action: {action}")
        print(new_state.to_dict())
        print(f"Fcf: {fcf}")
        print(f"Util realization: {utility_realization}")
        print(f"future_vf_utility: {future_vf_utility}")

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

    r_b = BORROW_RATE_EPOCH
    r_r = RISK_FREE_RATE_EPOCH

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
