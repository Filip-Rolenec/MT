import numpy as np

from gas_example.enum_types import PowerplantState, MothballedState
from gas_example.setup import GasProblemSetup
from gas_example.simulation.state import State, state_is_invalid


def get_lognormal_prices(start_price: float, time_epoch: int, sigma: float, sample_size: int):
    return [np.exp(np.log(start_price) + i) for i in np.random.normal(0, sigma * np.sqrt(time_epoch), sample_size)]


def get_exp_gov_policy_in_t(prob_up, prob_down, epoch):
    return min(1 + epoch * (prob_up - prob_down), 5)


def get_avg_exp_policies(prob_up, prob_down, epochs):
    return [get_exp_gov_policy_in_t(prob_up, prob_down, i) for i in range(epochs)]


def get_avg_sigma_for_price(prob_up, prob_down, epoch, sigma):
    if epoch == 0:
        return sigma

    exp_policies_by_epoch = get_avg_exp_policies(prob_up, prob_down, epoch)
    sigmas = [sigma * (1 + (epoch_policy - 1) * 0.2) for epoch_policy in exp_policies_by_epoch]
    sigmas_squared = [sigma * sigma for sigma in sigmas]
    sigma_average = np.sqrt(sum(sigmas_squared) / len(sigmas))

    return sigma_average


def get_power_price_sample(prob_up, prob_down, epoch, power_volatility, power_price, sample_size):
    avg_sigma = get_avg_sigma_for_price(prob_up, prob_down, epoch, power_volatility)
    return get_lognormal_prices(power_price, epoch, avg_sigma, sample_size)


def clean_policy_distribution(raw_random_policies):
    clean_policies = []
    for policy in raw_random_policies:
        if policy < 1:
            clean_policies.append(1)
        elif policy > 5:
            clean_policies.append(5)
        else:
            clean_policies.append(round(policy))
    return clean_policies


def get_gov_samples(epoch, prob_up, prob_down, sample_size):
    exp_gov_policy = 1 + (prob_up - prob_down) * epoch
    raw_random_policies = np.random.normal(exp_gov_policy, 1, sample_size)
    return clean_policy_distribution(raw_random_policies)


def get_powerplant_state_sample(sample_size):
    states = []
    for i in range(sample_size):
        states.append(np.random.choice([PowerplantState.NOT_BUILT,
                                        PowerplantState.STAGE_1,
                                        PowerplantState.STAGE_2],
                                       p=[0.2, 0.4, 0.4]))
    return states


def get_mothball_state_sample(sample_size):
    states = []
    for i in range(sample_size):
        states.append(np.random.choice([MothballedState.NORMAL,
                                        MothballedState.MOTHBALLED],
                                       p=[0.75, 0.25]))
    return states


def get_balance_sample(sample_size):
    return np.random.uniform(-130_000_000, 130_000_000, sample_size)


def get_individual_samples(ps: GasProblemSetup,
                           epoch: int,
                           individual_sample_size: int):
    gas_price_sample = get_lognormal_prices(ps.init_gas_price, epoch, ps.gas_vol, individual_sample_size)
    co2_price_sample = get_lognormal_prices(ps.init_co2_price, epoch, ps.co2_vol, individual_sample_size)
    power_sample = get_power_price_sample(ps.gov_prob_up, ps.gov_prob_up, epoch, ps.power_vol, ps.init_power_price,
                                          individual_sample_size,
                                          )
    clean_policies = get_gov_samples(epoch, ps.gov_prob_up, ps.gov_prob_down, individual_sample_size)
    powerplant_state = get_powerplant_state_sample(individual_sample_size)
    mothball_state = get_mothball_state_sample(individual_sample_size)
    balance_sample = get_balance_sample(individual_sample_size)

    return [gas_price_sample,
            co2_price_sample,
            power_sample,
            clean_policies,
            powerplant_state,
            mothball_state,
            balance_sample]


def get_state_sample(sample_size_global, sample_size_individual, epoch):
    individual_samples = get_individual_samples(GasProblemSetup(), epoch, sample_size_individual)

    states = []
    while len(states) < sample_size_global:
        state_by_element = []
        for sample in individual_samples:
            state_by_element.append(sample[np.random.choice(range(sample_size_individual))])
        generated_state = State(state_by_element[0],
                                state_by_element[1],
                                state_by_element[2],
                                state_by_element[3],
                                state_by_element[4],
                                state_by_element[5],
                                state_by_element[6])
        if not state_is_invalid(generated_state, print_warning=False):
            states.append(generated_state)

    return states
