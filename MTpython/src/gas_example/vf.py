from gas_example.model import AdpModel


class Vf:

    def __init__(self, model: AdpModel):
        self.params = model.parameters
        self.base_functions = model.basis_functions

    # Here we can see the interaction of parameters and basis functions. For now we have classical linear model.
    def compute_value(self, state):

        value_sum = 0
        for i in range(len(self.params)):
            value_sum += self.params[i]*self.base_functions[i](state)

        return value_sum

    def compute_all_values(self, states):
        return [self.compute_value(i) for i in states]

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
