import numpy as np


def get_new_state(state, action, prob_matrix):
    return np.random.choice(np.arange(0, 3), p=prob_matrix[state][action])
