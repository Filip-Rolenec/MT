from gas_example.enum_types import PowerplantState, RunningState
from gas_example.state import State


# Indicator function of sold plant
def bf_1(state: State):
    return 0 if state.plant_state == PowerplantState.SOLD else 1


# Price difference of inputs and outputs per MW
def bf_2(state: State):
    return state.power_price - state.co2_price - state.gas_price


# Indicator of positive and negative balance
def bf_3(state: State):
    return 0 if state.balance < 0 else 1


# How does the state favor renewables
def bf_4(state: State):
    return state.gov_state


# Plant stage, or installed capacity:
def bf_5(state: State):
    return 0 if state.plant_state == PowerplantState.SOLD else state.plant_state.value * 200


def bf_6(state: State):
    return state.running_state == RunningState.MOTHBALLED


def get_basis_functions():
    return [bf_1, bf_2, bf_3, bf_4, bf_5, bf_6]
