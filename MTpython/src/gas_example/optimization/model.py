from typing import List

from gas_example.optimization.basis_function import get_basis_functions


class AdpModel:

    def __init__(self, parameters: List[float] = (0, 0, 0, 0, 0, 0,)):
        self.basis_functions = get_basis_functions()
        self.parameters = parameters
