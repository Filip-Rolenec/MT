import numpy as np

from gas_example.enum_types import PowerplantState, MothballedState
from gas_example.setup import GAS_PRICE, GAS_VOL, CO2_PRICE, CO2_VOL, POWER_VOL, POWER_PRICE
from gas_example.simulation.state import State, state_is_invalid

SAMPLE_SIZE_INDIVIDUAL = 200
SAMPLE_SIZE_GLOBAL = 2000


def get_lognormal_prices(start_price: float, time_epoch: int, sigma: float, sample_size: int):
    return [np.exp(np.log(start_price) + i) for i in np.random.normal(0, sigma * np.sqrt(time_epoch), sample_size)]


def get_powerplant_state_sample(sample_size):
    states = []
    for i in range(sample_size):
        states.append(np.random.choice([PowerplantState.NOT_BUILT,
                                        PowerplantState.STAGE_1,
                                        PowerplantState.STAGE_2],
                                       p=[0.3, 0.35, 0.35]))
    return states


def get_balance_sample(sample_size):
    return np.random.uniform(-130_000_000, 130_000_000, sample_size)


def get_individual_samples(epoch: int,
                           individual_sample_size: int):
    gas_price_sample = get_lognormal_prices(GAS_PRICE, epoch, GAS_VOL, SAMPLE_SIZE_INDIVIDUAL)
    co2_price_sample = get_lognormal_prices(CO2_PRICE, epoch, CO2_VOL, SAMPLE_SIZE_INDIVIDUAL)
    power_sample = get_lognormal_prices(POWER_PRICE, epoch, POWER_VOL, SAMPLE_SIZE_INDIVIDUAL)
    powerplant_state = get_powerplant_state_sample(individual_sample_size)
    balance_sample = get_balance_sample(individual_sample_size)

    return [gas_price_sample,
            co2_price_sample,
            power_sample,
            powerplant_state,
            balance_sample]


def get_state_sample(epoch):
    individual_samples = get_individual_samples(epoch, SAMPLE_SIZE_INDIVIDUAL)
    states = []
    while len(states) < SAMPLE_SIZE_GLOBAL:
        state_by_element = []
        for sample in individual_samples:
            state_by_element.append(sample[np.random.choice(range(SAMPLE_SIZE_INDIVIDUAL))])
        generated_state = State(state_by_element[0],
                                state_by_element[1],
                                state_by_element[2],
                                state_by_element[3],
                                state_by_element[4])
        if not state_is_invalid(generated_state, print_warning=False):
            states.append(generated_state)

    return states
