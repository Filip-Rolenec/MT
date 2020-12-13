import operator
from typing import List

from gas_example.enum_types import Action
from gas_example.optimization.basis_function import uf_1, uf_2, uf_2_inv
from gas_example.setup import RISK_FREE_RATE, BORROW_RATE
from gas_example.simulation.state import State, get_valid_actions

import numpy as np

INTEGRAL_SAMPLE_SIZE = 10
UTILITY_FUNCTION = uf_2
INVERSE_UTILITY_FUNCTION = uf_2_inv


def get_state_utility_pairs(
        sampled_states: List[State],
        epoch: int,
        future_vf

):
    # print(epoch)

    state_utility_pairs = {}

    print_details = False

    for i, state in enumerate(sampled_states):
        # When creating a random path, we need to also consider the various actions.
        action = get_best_action(state, INTEGRAL_SAMPLE_SIZE, epoch, future_vf)

        # if epoch < 298:
        #print(state.to_dict())
        #print(action)
        #print_details = True

        utility_realization = get_utility_realization(state, action, epoch, future_vf, print_details)

        state_utility_pairs[i] = [state, utility_realization]

    return list(state_utility_pairs.values())


def get_best_action(state: State, integral_sample_size: int, epoch: int, future_vf):
    valid_actions = get_valid_actions(state, epoch)

    exp_utility_per_action = {}
    for action in valid_actions:
        # Because we are unable to compute the expected fcf conditioned on this action properly,
        # we need to compute the integral numerically, with selected sample size
        # TODO optimize this, for better precision.
        sample_integral_values = []
        for i in range(integral_sample_size):
            utility_realization = get_utility_realization(state, action, epoch, future_vf)

            sample_integral_values.append(utility_realization)

        exp_utility_per_action[action] = np.mean(sample_integral_values)

    #print(exp_utility_per_action)

    return max(exp_utility_per_action.items(), key=operator.itemgetter(1))[0]


def get_utility_realization(state: State, action: Action, epoch: int, future_vf, print_details=False):
    new_state, fcf = state.get_new_state_and_fcf(action, epoch)

    # This is utility, we need to make a money equivalent first.

    future_vf_utility = future_vf.compute_value(new_state)

    future_vf_money_equivalent = INVERSE_UTILITY_FUNCTION(future_vf_utility)

    # To know the pce of the reward fcf now and expected value represented by vf, we adjust by current balance.
    pce_realization = pce([state.balance + fcf, future_vf_money_equivalent]) - state.balance

    # After this, we subtract the balance and get the amount we need.

    utility_realization = round(UTILITY_FUNCTION(pce_realization), 2)

    if print_details:
        print(f"Future_vf: {future_vf.params}")
        print(f"future_vf_util: {future_vf_utility}")
        print(f"future_vf_money: {future_vf_money_equivalent}")

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
