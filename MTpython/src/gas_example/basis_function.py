from gas_example.enum_types import PowerplantState
from gas_example.state import State


# Indicator function of sold plant
def bf_1(state: State):
    return 0 if state.running_state == PowerplantState.SOLD else 1


# Price difference of inputs and outputs per MW
def bf_2(state: State):
    return state.power_price - state.co2_price - state.gas_price


# Indicator of positive and negative balance
def bf_3(state: State):
    return 0 if state.balance < 0 else 1
