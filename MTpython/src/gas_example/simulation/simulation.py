from gas_example.setup import TIME_EPOCHS, RISK_FREE_RATE_EPOCH, BORROW_RATE_EPOCH, YEARS

from gas_example.simulation.state import State


def run_simulation(strategy, initial_state: State):
    state = initial_state
    for epoch in range(TIME_EPOCHS - 1):
        action, _ = strategy.get_action(state, epoch)
        state, fcf = state.get_new_state_and_fcf(action)
    return round(balance_to_pce(state.balance) / 1000000.0, 5)


def balance_to_pce(balance):
    if balance > 0:
        pce = balance / (RISK_FREE_RATE_EPOCH ** TIME_EPOCHS)
    else:
        pce = balance / (BORROW_RATE_EPOCH ** TIME_EPOCHS)

    return pce
