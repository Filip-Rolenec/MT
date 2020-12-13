from typing import Dict

from gas_example.enum_types import PowerplantState
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from gas_example.optimization_2.optimization import PRINT_DETAILS_GLOBAL


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def get_new_params(state_utility_pairs: Dict):  # Dict where state is key and value is utility

    # Fitting for each of the states separately
    params_all = []
    for i in PowerplantState:
        spark_prices = [key.get_spark_price() for key in state_utility_pairs.keys() if key.plant_state == i]
        utilities = [state_utility_pairs[key] for key in state_utility_pairs.keys() if key.plant_state == i]
        params_pw, _ = optimize.curve_fit(piecewise_linear, spark_prices, utilities)

        xd = np.linspace(min(spark_prices), max(spark_prices), 500)
        if PRINT_DETAILS_GLOBAL:
            plt.plot(spark_prices, utilities, "o")
            plt.plot(xd, piecewise_linear(xd, *params_pw))
            plt.show()

        params_all.append(params_pw)

    return params_all
