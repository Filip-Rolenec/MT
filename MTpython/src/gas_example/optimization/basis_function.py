from gas_example.enum_types import PowerplantState, MothballedState
from gas_example.simulation.state import State


# Price difference of inputs and outputs per MW
def bf_1(state: State):
    return state.power_price - state.co2_price - state.gas_price


# Installed capacity, computed in blocks
def bf_2(state: State):
    return state.plant_state.value


# Indicator of positive and negative balance
def bf_3(state: State):
    return 0 if state.balance < 0 else 1


# How does the state favor renewables
def bf_4(state: State):
    return state.gov_state


# Indentificator of mothballed plant
def bf_5(state: State):
    return 1 if state.mothballed_state == MothballedState.MOTHBALLED else 0


def get_basis_functions():
    return [bf_1, bf_2, bf_3, bf_4, bf_5]
