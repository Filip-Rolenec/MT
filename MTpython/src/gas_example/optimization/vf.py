
from sklearn.linear_model import LinearRegression

from gas_example.optimization.basis_function import get_basis_functions
from gas_example.optimization.optimization import get_state_utility_pairs
from gas_example.optimization.sampling import get_state_sample

# Number of state samples for each update of value function
from simple_example.helpers import prepare_regression_variables

# Global number of new state sample
VF_SAMPLE_GLOBAL = 30
# Number of samples for each element of the state vector
VF_SAMPLE_IND = 10


class Vf:

    def __init__(self, parameters=None):
        #
        if parameters is None:
            self.base_functions = get_basis_functions()
            self.params = [0] * len(self.base_functions)
        else:
            self.params = parameters
            self.base_functions = get_basis_functions()

    # Here we can see the interaction of parameters and basis functions. For now we have classical linear model.
    def compute_value(self, state):
        value_sum = 0
        for i in range(len(self.params)):
            value_sum += self.params[i] * self.base_functions[i](state)

        return value_sum

    def set_params(self, params):
        self.params = params


# Value functions
def create_vfs_time_list( time_epochs: range):
    vfs = []
    for _ in time_epochs:
        vfs.append(Vf())
    return vfs


def update_vf_coef(current_vf: Vf, next_vf: Vf, basis_functions, time_epoch: int):
    sampled_states = get_state_sample(VF_SAMPLE_GLOBAL, VF_SAMPLE_IND, time_epoch)

    # print(f"{datetime.now()} Got sample_states, time epoch {time_epoch}")

    state_reward_pairs_raw = get_state_utility_pairs(sampled_states, time_epoch, next_vf)

    # print(f"{datetime.now()} Got state_reward_pairs, size {len(state_reward_pairs_raw)}")

    x, y = prepare_regression_variables(state_reward_pairs_raw, basis_functions)
    model = LinearRegression(fit_intercept=False).fit(x, y)

    #print(model.coef_)
    # Based on the linear fit of the model, update the vf_coefficients of the current epoch.
    current_vf.set_params(model.coef_)
