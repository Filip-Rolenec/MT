import numpy as np

from gas_example.setup import EPOCHS_IN_YEAR


def get_next_price(price_now, sigma):
    dt = 1/EPOCHS_IN_YEAR
    return price_now * np.exp((0 - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), 1))[0]


def get_next_gov_policy(gov_policy, negative_prob, positive_prob):
    zero_prob = 1 - positive_prob - negative_prob

    movement = np.random.choice(np.arange(1, 4), p=[negative_prob, zero_prob, positive_prob, ]) - 2

    if gov_policy + movement in range(1, 6):
        gov_policy = gov_policy + movement
    return gov_policy


def get_discount_rate(epoch_rate, epoch):
    return 1 + epoch_rate ** epoch
