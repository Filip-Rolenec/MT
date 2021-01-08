import numpy as np

from gas_example.enum_types import PowerplantState
from gas_example.setup import SAMPLE_SIZE_INDIVIDUAL, SAMPLE_SIZE_GLOBAL, TIME_EPOCHS, CO2_VOL, POWER_VOL, \
    EPOCHS_IN_YEAR, GAS_VOL
from gas_example.simulation.state import State, state_is_invalid


def get_price_sample(sample_size, from_value=0, to_value=200):
    return np.random.uniform(from_value, to_value, sample_size)





def get_powerplant_state_sample(sample_size):
    states = []
    for i in range(sample_size):
        states.append(np.random.choice([PowerplantState.NOT_BUILT,
                                        PowerplantState.STAGE_1,
                                        PowerplantState.STAGE_2],
                                       p=[0.3, 0.35, 0.35]))
    return states


def get_balance_sample(sample_size):
    return [0] * sample_size


def get_individual_samples(individual_sample_size: int):
    gas_price_sample = get_price_sample(SAMPLE_SIZE_INDIVIDUAL, 0,  30)
    co2_price_sample = get_price_sample(SAMPLE_SIZE_INDIVIDUAL, 5, 40)
    power_sample = get_price_sample(SAMPLE_SIZE_INDIVIDUAL, 10, 80)
    powerplant_state = get_powerplant_state_sample(individual_sample_size)
    balance_sample = get_balance_sample(individual_sample_size)

    return [gas_price_sample,
            co2_price_sample,
            power_sample,
            powerplant_state,
            balance_sample]


def get_state_sample():
    individual_samples = get_individual_samples(SAMPLE_SIZE_INDIVIDUAL)
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
