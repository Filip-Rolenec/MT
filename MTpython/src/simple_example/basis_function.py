def bf_1(s):
    return 1 if s == 0 else 0


def bf_2(s):
    return 1 if s == 1 else 0


def bf_3(s):
    return 1 if s == 2 else 0


def get_basis_functions():
    return [bf_1, bf_2, bf_3]
