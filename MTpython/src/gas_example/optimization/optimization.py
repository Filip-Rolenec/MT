import operator
from typing import List

import numpy as np

from gas_example.enum_types import Action, PowerplantState
from gas_example.optimization.basis_function import uf_2, uf_2_inv
from gas_example.setup import INTEGRAL_SAMPLE_SIZE, BORROW_RATE_EPOCH, RISK_FREE_RATE_EPOCH
from gas_example.simulation.state import State, get_valid_actions, get_next_plant_state

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
        utility_realizations = get_utility_realizations(state, action, future_vf)

        exp_utility_per_action[action] = np.mean(utility_realizations)

    if print_details:
        print(f"Spark: {state.get_spark_price()}")
        print(exp_utility_per_action)
        print("\n")

    best_action = max(exp_utility_per_action.items(), key=operator.itemgetter(1))[0]

    return best_action, exp_utility_per_action[best_action]


UTILITY_FUNCTION_V = np.vectorize(uf_2)
INVERSE_UTILITY_FUNCTION_V = np.vectorize(uf_2_inv)


def get_utility_realizations(state: State, action: Action, future_vf):
    spark_prices, fcfs = state.get_spark_prices_and_fcfs(action)
    new_powerplant_state = get_next_plant_state(state, action)

    future_vf_utilities = future_vf.compute_values(new_powerplant_state, spark_prices)

    future_vf_money_equivalents = INVERSE_UTILITY_FUNCTION_V(future_vf_utilities)

    updated_balances = [fcf + state.balance for fcf in fcfs]

    balance_future_vf_pairs = [[a, b] for a, b in zip(updated_balances, future_vf_money_equivalents)]

    pce_realizations = [pce_value - state.balance for pce_value in pce_v(balance_future_vf_pairs)]

    utility_realizations = np.round(UTILITY_FUNCTION_V(pce_realizations), 2)

    return utility_realizations


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


def pce_v(fcfs_v):
    pces = []
    for fcfs in fcfs_v:
        pces.append(pce(fcfs))
    return pces