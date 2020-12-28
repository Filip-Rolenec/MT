from typing import Dict

from lmfit import Model
from lmfit.models import StepModel

from gas_example.enum_types import PowerplantState
import numpy as np
import matplotlib.pyplot as plt


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0])


def get_new_models(state_utility_pairs: Dict, print_graphs_local=True):  # Dict where state is key and value is utility

    # Fitting for each of the states separately
    models = {}
    for plant_state in PowerplantState:
        spark_prices = [key.get_spark_price() for key in state_utility_pairs.keys() if key.plant_state == plant_state]
        utilities = [state_utility_pairs[key] for key in state_utility_pairs.keys() if key.plant_state == plant_state]

        pw_model = Model(piecewise_linear)
        slope_2 = max(utilities) / max(spark_prices)
        params = pw_model.make_params(x0=0, y0=0, k1=0, k2=slope_2)

        fitted_model = pw_model.fit(utilities, params, x=spark_prices)

        xd = np.linspace(min(spark_prices), max(spark_prices), 500)
        predicted_values = fitted_model.eval(x=xd)

        if print_graphs_local:
            plt.plot(spark_prices, utilities, "o")
            plt.plot(xd, predicted_values)
            plt.show()

        models[plant_state] = fitted_model

    print(models[PowerplantState.NOT_BUILT].params)
    return models


def get_zero_model():
    xdata = np.linspace(-100, 200, 301)
    ydata = np.zeros(301)

    model = StepModel(form='linear', prefix='step_')
    zero_model = model.fit(ydata, x=xdata)

    return zero_model
