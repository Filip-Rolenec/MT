def vector_mult(a, b):
    return sum([i * j for i, j in zip(a, b)])


def prepare_regression_variables(state_reward_pairs, basis_functions):
    x = []
    y = []
    for pair in state_reward_pairs:
        x.append([basis_functions[0](pair[0]),
                  basis_functions[1](pair[0]),
                  basis_functions[2](pair[0])])
        y.append(pair[1])

    return x, y
