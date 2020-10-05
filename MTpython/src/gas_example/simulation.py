from gas_example.setup import TIME_EPOCHS
from gas_example.strategy import Strategy
from gas_example.state import State


def run_simulation(strategy: Strategy, initial_state: State):
    state = initial_state
    profit = 0
    #print(f"state {state.to_dict()}")

    for epoch in range(TIME_EPOCHS):
        action = strategy.get_action(state, epoch)
        #print(f"action {action}")

        epochs_left = TIME_EPOCHS - epoch
        state, balance= state.get_new_state_and_reward(action, epochs_left)
        #print(f"balance {balance}")
        #print(f"state {state.to_dict()}")

    return balance
