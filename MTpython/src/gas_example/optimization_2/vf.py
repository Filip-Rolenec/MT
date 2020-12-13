from gas_example.optimization_2.adp_model import get_new_params
from gas_example.optimization_2.basis_function import get_basis_functions
from gas_example.optimization_2.optimization import get_state_utility_pairs
from gas_example.optimization_2.sampling import get_state_sample
from gas_example.setup import TIME_EPOCHS
import numpy as np


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def create_vfs_time_list():
    return [Vf()] * TIME_EPOCHS


class Vf:

    def __init__(self):
        self.basis_functions = get_basis_functions()
        # Parameters, which define the pw linear function. Three time, one for each state.
        self.pw_params = [[0, 0, 0, 0]] * 3

    def compute_value(self, state):
        pw_params = self.pw_params[state.plant_state.value]  # Choose the correct piecewise parameters
        return piecewise_linear(state.get_spark_price(), *pw_params)

    def set_params(self, params):
        self.pw_params = params


def update_vf_coef(current_vf: Vf, next_vf: Vf, time_epoch: int):
    sampled_states = get_state_sample(time_epoch)

    state_utility_pairs = get_state_utility_pairs(sampled_states, time_epoch, next_vf)

    new_pw_params = get_new_params(state_utility_pairs)

    current_vf.set_params(new_pw_params)
