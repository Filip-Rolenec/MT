import warnings

import numpy as np

from gas_example.enum_types import PowerplantState, RunningState, Action
from gas_example.fcf import compute_fcf
from gas_example.setup import BORROW_RATE, get_epoch_rate, RISK_FREE_RATE
from gas_example.transformation_functions import get_next_price, get_next_gov_policy, get_reward_from_fcf

CO2_VOLATILITY = 0.02
GAS_VOLATILITY = 0.04
POWER_VOLATILTY = 0.05
POWER_VOL_COEF = 0.02

GOV_PROB_UP = 0.07
GOV_PROB_DOWN = 0.03

IMPOSSIBLE_PLANT_STATES = [[PowerplantState.NOT_BUILT, RunningState.RUNNING],
                           [PowerplantState.NOT_BUILT, RunningState.MOTHBALLED],
                           [PowerplantState.SOLD, RunningState.RUNNING],
                           [PowerplantState.SOLD, RunningState.MOTHBALLED]
                           ]


def get_initial_state():
    return State(13, 9, 40, 1, PowerplantState.NOT_BUILT, RunningState.NOT_RUNNING, 0)


class State:
    def __init__(self, gas_price: float, co2_price: float, power_price: float, gov_state: int,
                 plant_state: PowerplantState, running_state: RunningState, balance: float):
        self.gas_price = gas_price
        self.co2_price = co2_price
        self.power_price = power_price
        self.gov_state = gov_state
        self.plant_state = plant_state
        self.running_state = running_state
        self.balance = balance

    def get_new_state_and_reward(self, action, epochs_left, epoch):
        gas_price = get_next_price(self.gas_price, GAS_VOLATILITY)
        co2_price = get_next_price(self.co2_price, CO2_VOLATILITY)
        power_price = get_next_price(self.power_price, get_pow_vol(self.gov_state))
        gov_state = get_next_gov_policy(self.gov_state, GOV_PROB_DOWN, GOV_PROB_UP)
        plant_state = get_next_plant_state(self, action)
        running_state = get_next_running_state(self, action)

        fcf_raw = compute_fcf(gas_price, co2_price, power_price, running_state, plant_state, action, epochs_left)
        reward = get_reward_from_fcf(self.balance, fcf_raw, epoch)
        balance = update_balance(self.balance, fcf_raw, epochs_left)

        return State(gas_price, co2_price, power_price, gov_state, plant_state, running_state, balance), reward

    def to_dict(self):
        return self.__dict__


def get_pow_vol(gov_state: int):
    return POWER_VOLATILTY * (1 + (gov_state - 1) * POWER_VOL_COEF)


def get_next_plant_state(state: State, action: Action):
    if action_is_invalid(action, state, print_warning=True):
        raise Exception(f"Action {action} in state {state} is invalid")

    # Building new stage
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        if state.plant_state == PowerplantState.NOT_BUILT:
            return PowerplantState.STAGE_1
        elif state.plant_state == PowerplantState.STAGE_1:
            return PowerplantState.STAGE_2
        else:
            raise Exception(f"New stage cannot be built, plant is in a state {state.plant_state}")
    # Selling the plant
    if action == Action.SELL:
        return PowerplantState.SOLD
    # Not changing the plant state
    else:
        return state.plant_state


def get_next_running_state(state: State, action: Action):
    if action_is_invalid(action, state, print_warning=True):
        raise Exception(f"Action {action} in state {state} is invalid")

    if action == Action.RUN or action == Action.RUN_AND_BUILD:
        return RunningState.RUNNING

    if action == Action.DO_NOTHING or action == Action.IDLE_AND_BUILD or action == Action.SELL:
        return RunningState.NOT_RUNNING

    if action == Action.MOTHBALL:
        return RunningState.MOTHBALLED

    else:
        raise Exception(f"The action {action} in state {state.to_dict()} is problematic")


def update_balance(balance, fcf_raw, epochs_left):
    # if there is no time left, I need to repay all the money
    if epochs_left == 0 and balance < 0:
        return 0, fcf_raw + balance

    if balance < 0:
        money_from_interest = get_epoch_rate(BORROW_RATE) * balance
    else:
        money_from_interest = get_epoch_rate(RISK_FREE_RATE) * balance
    fcf = fcf_raw + money_from_interest
    balance = balance + fcf

    return balance


def state_is_invalid(state: State, print_warning=True):
    # non-positive prices
    if sum(np.sign([state.power_price,
                    state.gas_price,
                    state.co2_price])) < 3:
        if print_warning:
            warnings.warn(f"Prices are invalid {state.power_price}, {state.gas_price}, {state.co2_price}")

    # Gov_policy in range
    if state.gov_state not in range(1, 6):
        if print_warning:
            warnings.warn(f"Government policy {state.gov_state} is not in range")
        return True

    if [state.plant_state, state.running_state] in IMPOSSIBLE_PLANT_STATES:
        if print_warning:
            warnings.warn(f"Plant state {[state.plant_state, state.running_state]} is not in allowed states")
        return True

    return False


def action_is_invalid(action: Action, state: State, print_warning: bool):
    if state_is_invalid(state):
        if print_warning:
            warnings.warn(f"State {state.to_dict()} is invalid")
        return True

    if state.plant_state == PowerplantState.SOLD and action != Action.DO_NOTHING:
        if print_warning:
            warnings.warn(f"Action {action} is not possible, powerplant was sold.")
        return True

    if action == Action.RUN and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("Powerplant cannot run, it was not built yet.")
        return True

    if action == Action.RUN_AND_BUILD and state.plant_state == PowerplantState.STAGE_2:
        if print_warning:
            warnings.warn("Powerplant cannot be extended in stage 2")
        return True

    if action == Action.IDLE_AND_BUILD and state.plant_state == PowerplantState.STAGE_2:
        if print_warning:
            warnings.warn("Cannot build new capacity at stage 2.")
        return True

    if action == Action.MOTHBALL and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("There is nothing to mothball")
        return True

    if action == Action.SELL and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("There is nothing to sell")
        return True

    return False


def get_valid_actions(state: State):
    return [action for action in Action if not action_is_invalid(action, state, print_warning=False)]
