def vector_mult(a, b):
    return sum([i * j for i, j in zip(a, b)])


def prepare_regression_variables(state_reward_pairs, basis_functions, intercept=False):
    # x vector of regression values.
    # y vector of responses
    # pair consists of State and sampled utility.
    x = []
    y = []
    print(basis_functions)
    if intercept:
        basis_functions = basis_functions[1:len(basis_functions)]

    print(basis_functions)
    for pair in state_reward_pairs:
        one_x = []
        for basis_function in basis_functions:
            one_x.append(basis_function(pair[0]))
        x.append(one_x)
        y.append(pair[1])

    return x, y
