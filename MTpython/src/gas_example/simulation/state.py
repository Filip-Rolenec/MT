import warnings

import numpy as np

from gas_example.enum_types import PowerplantState, Action, MothballedState
from gas_example.setup import BORROW_RATE, get_epoch_rate, RISK_FREE_RATE, GAS_VOL, CO2_VOL, \
    POWER_VOL_COEF, POWER_VOL, TIME_EPOCH_BUILDING_LIMIT, POWERPLANT_COST, MOTHBALLING_COST, MAINTENANCE_COST_PER_MW, \
    HOURS_IN_MONTH
from gas_example.simulation.evolution_models import get_next_price


def get_initial_state():
    return State(14, 10, 40, PowerplantState.NOT_BUILT, 0)


class State:
    def __init__(self, gas_price: float, co2_price: float, power_price: float,
                 plant_state: PowerplantState, balance: float):
        self.gas_price = gas_price
        self.co2_price = co2_price
        self.power_price = power_price
        self.plant_state = plant_state
        self.balance = balance

    def get_new_state_and_fcf(self, action, epoch):
        gas_price = get_next_price(self.gas_price, GAS_VOL)
        co2_price = get_next_price(self.co2_price, CO2_VOL)
        power_price = get_next_price(self.power_price, POWER_VOL)
        plant_state = get_next_plant_state(self, action, epoch)

        fcf = compute_fcf(self,
                          self.plant_state,
                          action)
        balance = round(update_balance(self.balance, fcf))

        return State(gas_price, co2_price, power_price, plant_state, balance), fcf

    def get_spark_price(self):
        return self.power_price - self.co2_price - self.gas_price - MAINTENANCE_COST_PER_MW

    def to_dict(self):
        return self.__dict__


def get_pow_vol(gov_state: int):
    return POWER_VOL * (1 + (gov_state - 1) * POWER_VOL_COEF)


def get_next_plant_state(state: State, action: Action, time_epoch: int):
    if action_is_invalid(action, state, time_epoch, print_warning=True):
        raise Exception(f"Action {action} in state {state} is invalid")

    # Building new stage
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        if state.plant_state == PowerplantState.NOT_BUILT:
            return PowerplantState.STAGE_1
        elif state.plant_state == PowerplantState.STAGE_1:
            return PowerplantState.STAGE_2
        else:
            raise Exception(f"New stage cannot be built, plant is in a state {state.plant_state}")

    # Other actions do not change the plant state
    else:
        return state.plant_state


def update_balance(balance, fcf_raw):

    new_balance = balance + fcf_raw

    if new_balance < 0:
        new_balance_updated = get_epoch_rate(BORROW_RATE) * new_balance
    else:
        new_balance_updated = get_epoch_rate(RISK_FREE_RATE) * new_balance

    return new_balance_updated


def state_is_invalid(state: State, print_warning=True):
    # non-positive prices
    if sum(np.sign([state.power_price,
                    state.gas_price,
                    state.co2_price])) < 3:
        if print_warning:
            warnings.warn(f"Prices are invalid {state.power_price}, {state.gas_price}, {state.co2_price}")

    return False


def action_is_invalid(action: Action, state: State, time_epoch: int, print_warning: bool):
    if state_is_invalid(state):
        if print_warning:
            warnings.warn(f"State {state.to_dict()} is invalid")
        return True

    if action == Action.RUN and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("Powerplant cannot run, it was not built yet.")
        return True

    if action == Action.RUN_AND_BUILD and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("Cannot run what is not built")
        return True

    if action == Action.RUN_AND_BUILD and state.plant_state == PowerplantState.STAGE_2:
        if print_warning:
            warnings.warn("Powerplant cannot be extended in stage 2")
        return True

    if action == Action.IDLE_AND_BUILD and state.plant_state == PowerplantState.STAGE_2:
        if print_warning:
            warnings.warn("Cannot build new capacity at stage 2.")
        return True

    if action == Action.MOTHBALL_CHANGE and state.plant_state == PowerplantState.NOT_BUILT:
        if print_warning:
            warnings.warn("There is nothing to mothball")
        return True

    return False


def get_valid_actions(state: State, time_epoch: int):
    return [action for action in Action if not action_is_invalid(action, state, time_epoch, print_warning=False)]


def compute_fcf(state: State,
                plant_state: PowerplantState,
                action: Action):
    profit = 0

    # Building new capacity costs money
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        profit = - POWERPLANT_COST

    installed_mw = get_installed_mw(plant_state)

    if action == Action.MOTHBALL_CHANGE:
        profit -= MOTHBALLING_COST * plant_state.value

    profit -= installed_mw * MAINTENANCE_COST_PER_MW * HOURS_IN_MONTH

    # Making profit if action is to run:
    if action == Action.RUN_AND_BUILD or action == Action.RUN:
        profit += (state.get_spark_price()) * installed_mw * HOURS_IN_MONTH

    return profit


def get_installed_mw(p_state: PowerplantState):
    if p_state == PowerplantState.STAGE_1:
        return 200
    elif p_state == PowerplantState.STAGE_2:
        return 400
    else:
        return 0
