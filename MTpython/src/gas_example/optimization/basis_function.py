from gas_example.simulation.state import State


# Spark price, representing the value of having a plant
def bf_1(state: State):
    return state.get_spark_price()


# State of a powerplant, representing the value of having already built the powerplant
def bf_2(state: State):
    return state.plant_state.value


def get_basis_functions():
    return [bf_1, bf_2]


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
