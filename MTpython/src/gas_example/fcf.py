from gas_example.enum_types import RunningState, PowerplantState, Action

POWERPLANT_COST = 650_000_000
MAINTENANCE_COST_PER_MW = 3
MOTHBALLED_COST_PER_MW = 1
SALVAGE_VALUE_MONTH_PER_MW = 200
HOURS_IN_MONTH = 30*24

def compute_fcf(gas_price: float,
                co2_price: float,
                power_price: float,
                running_state: RunningState,
                plant_state: PowerplantState,
                action: Action, epochs_left):
    profit = 0

    # Building new capacity
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        profit = - POWERPLANT_COST

    installed_mw = get_installed_mw(plant_state)

    # Does not matter if new state or action, it is deterministic
    if running_state == RunningState.RUNNING:
        profit += (power_price - co2_price - gas_price - MAINTENANCE_COST_PER_MW) * installed_mw * HOURS_IN_MONTH
    elif running_state == RunningState.NOT_RUNNING:
        profit += -MAINTENANCE_COST_PER_MW * installed_mw * HOURS_IN_MONTH
    elif running_state == RunningState.MOTHBALLED:
        profit += -MOTHBALLED_COST_PER_MW * installed_mw * HOURS_IN_MONTH
    # This makes the r a funciton of time, not sure if we want that.
    elif action == Action.SELL:
        profit += installed_mw * SALVAGE_VALUE_MONTH_PER_MW * epochs_left

    return profit


def get_installed_mw(p_state: PowerplantState):
    if p_state == PowerplantState.STAGE_1:
        return 200
    elif p_state == PowerplantState.STAGE_2:
        return 400
    else:
        return 0
