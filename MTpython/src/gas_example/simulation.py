import operator
from typing import List

import numpy as np

from gas_example.setup import TIME_EPOCHS, GasProblemSetup
from gas_example.strategy import Strategy
from gas_example.state import State, get_valid_actions


def run_simulation(strategy: Strategy, initial_state: State):
    state = initial_state
    cumulative_reward = 0
    for epoch in range(TIME_EPOCHS):
        action = strategy.get_action(state, epoch)
        epochs_left = TIME_EPOCHS - epoch
        state, reward = state.get_new_state_and_reward(action, epochs_left, epoch)
        cumulative_reward += reward

    return cumulative_reward


def get_state_reward_pairs(
        sampled_states: List[State],
        epochs_left: int,
        epoch: int,
        future_vf

):
    state_reward_pairs = []
    for state in sampled_states:
        action = get_best_action(state, 10, epochs_left, epoch, future_vf)
        new_state, reward = state.get_new_state_and_reward(action, epochs_left, epoch)
        future_vf.compute_value(new_state)

        state_reward_pairs.append([state, reward + future_vf.compute_value(new_state)])
    return state_reward_pairs


def get_best_action(state: State, integral_sample_size: int, epochs_left: int, epochs: int, future_vf):
    valid_actions = get_valid_actions(state)

    exp_rew_per_action = {}
    for action in valid_actions:
        sample_integral_values = []
        for i in range(integral_sample_size):
            new_state, reward = state.get_new_state_and_reward(action, epochs_left, epochs)
            vf_value = future_vf.compute_value(new_state)
            sample_integral_values.append(reward + vf_value)

        exp_rew_per_action[action] = np.mean(sample_integral_values)

    return max(exp_rew_per_action.items(), key=operator.itemgetter(1))[0]
