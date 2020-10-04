import numpy as np

from simple_example.helpers import vector_mult
from simple_example.setup import ProblemSetup
from simple_example.state import get_new_state
from simple_example.strategy import get_action_from_strategy


def run_simulation(strategy, initial_state, problem_setup: ProblemSetup):
    state = initial_state
    reward = 0
    for epoch in problem_setup.time_epochs:
        action = get_action_from_strategy(strategy, state, epoch)
        new_state = get_new_state(state, action, problem_setup.prob_matrix)
        reward += problem_setup.reward_matrix[state][action][new_state]

        state = new_state

    return reward


def get_state_reward_pairs(sample_size, future_vf, problem_setup):
    state_reward_pairs = []
    for i in range(sample_size):
        state = np.random.choice(problem_setup.states, p=[0.34, 0.33, 0.33])

        action = get_best_action(state, future_vf, problem_setup)
        new_state = get_new_state(state, action, problem_setup.prob_matrix)

        reward = problem_setup.reward_matrix[state][action][new_state] + future_vf.compute_value(new_state)
        state_reward_pairs.append([state, reward])
    return state_reward_pairs


def get_best_action(state, future_vf, problem_setup: ProblemSetup):
    exp_rewards = []
    for action in problem_setup.actions:
        exp_rewards.append(get_exp_value(state, action, problem_setup, future_vf))
    return np.argmax(exp_rewards)


def get_exp_value(state, action, problem_setup: ProblemSetup, future_vf):
    rewards = problem_setup.reward_matrix[state][action]
    future_vf_values = future_vf.compute_all_values()
    total_rewards = [i + j for i, j in zip(rewards, future_vf_values)]
    return vector_mult(problem_setup.prob_matrix[state][action], total_rewards)
