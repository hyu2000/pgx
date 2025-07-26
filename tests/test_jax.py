"""

"""

import jax
import jax.numpy as jnp
import equinox as eqx


global_list = []

@jax.jit
#@eqx.filter_jit
def evaluate(model, x):
    # side effect. compile-time only
    global_list.append(model)

    y = model(x)
    return y


class SimpleAdder:
    def __init__(self, increment = 1):
        self.increment = increment

    def __call__(self, x):
        return x + self.increment


class Adder(eqx.Module):
    increment: jnp.array

    def __init__(self, increment = 1):
        self.increment = increment

    def __call__(self, x):
        return x + self.increment


class Multiplier(eqx.Module):
    multiplier: jnp.array

    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def __call__(self, x):
        return x * self.multiplier


def test_jit_object():
    adder1 = SimpleAdder()
    adder2 = SimpleAdder(2)

    # TypeError:
    print(evaluate(adder1, jnp.arange(3)))


def test_polymorphism():
    modela = Adder()
    modela2 = Adder(2)
    modelm = Multiplier(2)

    print(evaluate(modela, jnp.arange(3)))
    assert(len(global_list) == 1)
    print(evaluate(modela, jnp.arange(3) + 4))
    assert(len(global_list) == 1)

    # same function, different parameter: does not trigger a re-jit
    print(evaluate(modela2, jnp.arange(3)))
    assert(len(global_list) == 1)

    # different function: re-jit
    print(evaluate(modelm, jnp.arange(3)))
    assert(len(global_list) == 2)

    # shape change: re-jit
    print(evaluate(modela, jnp.arange(4)))
    assert(len(global_list) == 3)
    print(global_list)
