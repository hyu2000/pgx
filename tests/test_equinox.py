"""

"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from pprint import pprint
import equinox as eqx


global_list = []

@jax.jit
#@eqx.filter_jit
def evaluate(model, x):
    """ model encapsulates both function and param

    model is a eqx.Module, converted to a pytree
    How does jit know to recompile when a different model is used: different pytree node type
    """
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


class NamedAdder(eqx.Module):
    increment: jnp.array
    name: str = eqx.field(static=True)

    def __init__(self, increment = 1, name='unnamed'):
        self.increment = increment
        self.name = name

    def __call__(self, x):
        return x + self.increment


def test_ptree():
    modela = Adder(5)
    # notice CustomNode
    print(jax.tree.structure(modela))
    print(jax.tree.leaves(modela))
    print(jax.tree.leaves_with_path(modela))


def test_jit_object():
    adder1 = SimpleAdder()
    adder2 = SimpleAdder(2)

    # TypeError: adder is an object, not a pytree
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


def test_device_put_str_fields():
    devices = jax.local_devices()

    # this won't work: x = {'name': 'x'}
    x = {'val': jnp.arange(3)}
    y = jax.device_put_replicated(x, devices)
    print(y['val'].shape)

    modela = NamedAdder(name='a')
    assert modela.name == 'a'
    print(modela.increment)
    # modela.name marked as static
    modely = jax.device_put_replicated(modela, devices)
    print(modely.increment)
    assert modely.name == modela.name


def test_device_put_aznet():
    from examples.alphazero.network import AZNet, BlockV2
    nn = BlockV2(16, jax.random.PRNGKey(0))
    nn_structure = jax.tree.structure(nn)
    print(nn_structure)
    # TypeError: Argument 'batch' of type <class 'str'> is not a valid JAX type
    # why isn't BatchNorm properties static?!
    devices = jax.local_devices()
    params, static = eqx.partition(nn, eqx.is_array)
    y = jax.device_put_replicated(params, devices)
    print(y)
    nn_device = eqx.combine(y, static)
    print(nn_device)


def test_device_put_bn():
    nn = eqx.nn.BatchNorm(16, axis_name='batch')

    devices = jax.local_devices()
    params, static = eqx.partition(nn, eqx.is_array)
    y = jax.device_put_replicated(nn, devices)
    # assert y.axis_name is None
    # nn_device = eqx.combine(y, static)
    # print(nn_device)
    assert y.axis_name == 'batch'


def test_device_get():
    x = jax.numpy.array([1., 2., 3.])
    x_host = jax.device_get(x)
    print(x_host)


def test_nn_state():
    def create_model(num_channels):
        from examples.alphazero.network import BlockV2
        return BlockV2(num_channels, jax.random.PRNGKey(0))
    model, bn_state = eqx.nn.make_with_state(create_model)(3)
    pprint(jax.tree.structure(model))
