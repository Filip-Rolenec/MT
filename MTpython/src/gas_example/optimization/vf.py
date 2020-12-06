from datetime import datetime

from sklearn.linear_model import LinearRegression

from gas_example.optimization.model import AdpModel
from gas_example.optimization.sampling import get_state_sample
from gas_example.simulation import get_state_reward_pairs

# Number of state samples for each update of value function
from simple_example.helpers import prepare_regression_variables

VF_SAMPLE_GLOBAL = 50
# Number of samples for each element of the state vector
VF_SAMPLE_IND = 5


class Vf:

    def __init__(self, model: AdpModel):
        self.params = model.parameters
        self.base_functions = model.basis_functions

    # Here we can see the interaction of parameters and basis functions. For now we have classical linear model.
    def compute_value(self, state):
        value_sum = 0
        for i in range(len(self.params)):
            value_sum += self.params[i] * self.base_functions[i](state)

        return value_sum

    def set_params(self, params):
        self.params = params


def create_vfs(model: AdpModel, time_epochs: range):
    vfs = []
    for i in time_epochs:
        vfs.append(Vf(model))
    return vfs


def update_vf_coef(current_vf: Vf, next_vf: Vf, basis_functions, epoch: int, epochs_left: int):
    sampled_states = get_state_sample(VF_SAMPLE_GLOBAL, VF_SAMPLE_IND, epoch)
    print(f"{datetime.now()} Got sample_states")
    state_reward_pairs_raw = get_state_reward_pairs(sampled_states, epochs_left, epoch, next_vf)
    print(f"{datetime.now()} Got sstate_reward_pairs, size {len(state_reward_pairs_raw)}")

    x, y = prepare_regression_variables(state_reward_pairs_raw, basis_functions)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    current_vf.set_params(model.coef_)
