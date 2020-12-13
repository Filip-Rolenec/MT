from gas_example.enum_types import MothballedState
from gas_example.simulation.state import State

import numpy as np


# Spark price time installed capacity - should be positive and allow for building as a best action
def bf_1(state: State):
    return state.get_spark_price() * state.plant_state.value


# Spark price only, reflects the value of spark, even with 0 capacity
def bf_2(state: State):
    return state.get_spark_price()


# Indicator of positive balance
def bf_3(state: State):
    return 0 if state.balance <= 0 else 1


# How does the state favor renewables
def bf_4(state: State):
    return state.gov_state

# Mothballed plant* spark price. This should cover the fact that mothballed plant has value in negative spark.
# It saves money, when the spark is negative, however it is bad when spark is positive.
def bf_5(state: State):
    is_mothballed = 1 if state.mothballed_state == MothballedState.MOTHBALLED else 0

    return state.get_spark_price() * is_mothballed

# Only value of the mothballed state, should be the around the price of chaning it.
def bf_6(state: State):
    return 1 if state.mothballed_state == MothballedState.MOTHBALLED else 0


def get_basis_functions():
    return [bf_1, bf_2, bf_3, bf_4, bf_5, bf_6]


def uf_1(x):
    x = x / 5000

    if x <= 0:
        return -((-x) ** (1 / 1.7))
    if x < 100:
        return np.log(2 * x)
    else:
        return x ** (1 / 2) - 4.7


def uf_2(x):
    x = x / 1000

    if x <= 0:
        return -((-x) ** (1 / 1.2))
    else:
        return x ** (1 / 1.25)


def uf_2_inv(y):
    if y < 0:
        thousands = -((-y) ** 1.2)
    else:
        thousands = y ** 1.25

    return thousands * 1000
