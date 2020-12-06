from gas_example.setup import TIME_EPOCHS, RISK_FREE_RATE, get_epoch_rate, BORROW_RATE
from gas_example.simulation.strategy import Strategy

from gas_example.simulation.state import State


def run_simulation(strategy: Strategy, initial_state: State):
    state = initial_state
    # balances = []
    for epoch in range(TIME_EPOCHS):
        action = strategy.get_action(state, epoch)
        state, fcf = state.get_new_state_and_fcf(action, epoch)
    # balances.append(round(state.balance) / 1000000.0)
    return round(balance_to_pce(state.balance) / 1000000.0)


def balance_to_pce(balance):
    if balance > 0:
        pce = balance / (get_epoch_rate(RISK_FREE_RATE) ** TIME_EPOCHS)
    else:
        pce = balance / (get_epoch_rate(BORROW_RATE) ** TIME_EPOCHS)

    return pce



