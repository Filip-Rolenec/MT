from strategy import Strategy
from state import State

TIME_EPOCHS = 300


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
