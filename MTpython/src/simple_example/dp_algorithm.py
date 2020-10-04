import numpy as np

from simple_example.setup import ProblemSetup


def classic_dp(horizon_vf, problem_setup: ProblemSetup):
    vf = {}
    vf_prev_epoch = horizon_vf

    time_epochs_count = len(problem_setup.time_epochs)
    strategy = {}

    exp_vector = [0, 0]

    for epoch in problem_setup.time_epochs:
        vf_epoch = []
        epoch_strategy = {}
        for state in problem_setup.states:
            # print(state)
            for action in problem_setup.actions:
                exp_vector[action] = sum(
                    [i * j for i, j in zip(problem_setup.prob_matrix[state][action], [i + j for i, j in zip(
                        problem_setup.reward_matrix[state][action], vf_prev_epoch)])])

            epoch_strategy[state + 1] = np.argmax([exp_vector]) + 1
            vf_epoch.append(np.max([exp_vector]))
        vf_prev_epoch = vf_epoch
        vf[time_epochs_count - epoch - 1] = vf_epoch
        strategy[time_epochs_count - epoch - 1] = epoch_strategy

    return strategy, vf
