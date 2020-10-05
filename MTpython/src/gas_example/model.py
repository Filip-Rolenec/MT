from typing import List


class AdpModel:

    def __init__(self, parameters: List[float]):
        self.basis_functions = get_basis_functions()
        self.parameters = parameters


def get_basis_functions():

