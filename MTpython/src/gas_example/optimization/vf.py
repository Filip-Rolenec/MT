from gas_example.enum_types import PowerplantState
from gas_example.optimization.adp_model import get_zero_model, get_new_models
from gas_example.optimization.optimization import get_state_utility_pairs
from gas_example.optimization.sampling import get_state_sample
from gas_example.setup import TIME_EPOCHS
import numpy as np


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def create_vfs_time_list():
    vfs = []
    for i in range(TIME_EPOCHS):
        vfs.append(Vf())
    return vfs


class Vf:

    def __init__(self):
        self.models = {PowerplantState.NOT_BUILT: get_zero_model(),
                       PowerplantState.STAGE_1: get_zero_model(),
                       PowerplantState.STAGE_2: get_zero_model()
                       }

    def compute_value(self, state):
        model = self.models[state.plant_state]  # Choose the correct piecewise parameters
        return model.eval(x=[state.get_spark_price()])[0]

    def set_models(self, models):
        self.models = models


def update_vf_models(current_vf: Vf, next_vf: Vf):
    sampled_states = get_state_sample()
    state_utility_pairs = get_state_utility_pairs(sampled_states, next_vf)

    new_models = get_new_models(state_utility_pairs)

    current_vf.set_models(new_models)
