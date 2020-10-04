from sklearn.linear_model import LinearRegression

from simple_example.basis_function import get_basis_functions
from simple_example.helpers import prepare_regression_variables
from simple_example.simulation import get_state_reward_pairs


class Vf:

    def __init__(self, params):
        self.params = params
        self.base_functions = get_basis_functions()

    def compute_value(self, state):
        return self.params[0] * self.base_functions[0](state) + \
               self.params[1] * self.base_functions[1](state) + \
               self.params[2] * self.base_functions[2](state)

    def compute_all_values(self):
        return [self.compute_value(0),
                self.compute_value(1),
                self.compute_value(2)]

    def set_params(self, params):
        self.params = params


def create_vfs(time_epochs, theta_initial):
    vfs = []
    for i in time_epochs:
        vfs.append(Vf(theta_initial))
    return vfs


def update_vf_coef(current_vf, next_vf, problem_setup, sample_size, basis_functions):
    state_reward_pairs_raw = get_state_reward_pairs(sample_size, next_vf, problem_setup)

    x, y = prepare_regression_variables(state_reward_pairs_raw, basis_functions)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    current_vf.set_params(model.coef_)
