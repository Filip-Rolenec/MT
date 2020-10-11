def vector_mult(a, b):
    return sum([i * j for i, j in zip(a, b)])


def prepare_regression_variables(state_reward_pairs, basis_functions):
    x = []
    y = []
    for pair in state_reward_pairs:
        one_x = []
        for basis_function in basis_functions:
            one_x.append(basis_function(pair[0]))
        x.append(one_x)
        y.append(pair[1])

    return x, y
