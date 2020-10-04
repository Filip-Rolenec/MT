import numpy as np


def get_next_price(current_price, volatility) -> float:
    return np.random.lognormal(np.log(current_price), volatility)


def get_next_gov_policy(gov_policy, negative_prob, positive_prob):
    zero_prob = 1 - positive_prob - negative_prob

    movement = np.random.choice(np.arange(1, 4), p=[negative_prob, zero_prob, positive_prob, ]) - 2

    if gov_policy + movement in range(1, 6):
        gov_policy = gov_policy + movement
    return gov_policy
