import numpy as np

from gas_example.setup import GasProblemSetup


def get_next_price(current_price, volatility) -> float:
    return np.random.lognormal(np.log(current_price), volatility)


def get_next_gov_policy(gov_policy, negative_prob, positive_prob):
    zero_prob = 1 - positive_prob - negative_prob

    movement = np.random.choice(np.arange(1, 4), p=[negative_prob, zero_prob, positive_prob, ]) - 2

    if gov_policy + movement in range(1, 6):
        gov_policy = gov_policy + movement
    return gov_policy


def get_discount_rate(epoch_rate, epoch):
    return (1 + epoch_rate ** epoch)


def get_reward_from_fcf(balance, fcf, epoch):
    setup = GasProblemSetup()
    b_rate = setup.epoch_b_rate
    rf_rate = setup.epoch_rf_rate
    # I owe money
    if balance < 0:
        # All fcf that I have earned goes towards the debt
        if balance + fcf < 0:
            return fcf / get_discount_rate(b_rate, epoch)
        # Or I made more than I owe. Thus negative balance is discounted as debt, while the positive gains by risk free
        else:
            reward_1 = -balance / get_discount_rate(b_rate, epoch)
            reward_2 = (balance + fcf) / get_discount_rate(rf_rate, epoch)
            return reward_1 + reward_2
    # I did not owe money
    else:
        # All goes into cash surplus
        if balance + fcf > 0:
            return fcf / get_discount_rate(rf_rate, epoch)
        # I have lost the balance and went into debt. Each of them have different discount rates
        else:
            reward_1 = -balance / get_discount_rate(rf_rate, epoch)
            reward_2 = (fcf + balance) / get_discount_rate(b_rate, epoch)
            return reward_1 + reward_2
