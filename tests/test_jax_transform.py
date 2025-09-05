from functools import partial

import jax
import jax.random as jrandom
from jax import numpy as jnp
from jax import pmap, vmap


""" e.g. eval() how's the result agg'ed

could i have messed it up, even when we have only 1 device?
"""

def test_pmap_basic():
    out = pmap(lambda x: x ** 2)(jnp.arange(1))
    print(out)
    print(out.shape)


def test_pmap_out():
    """ this is a bit surprising:
    # Note: pmap returns values mapped over their leading axis, equivalent to using out_axes=0 in vmap
    # It's treated in a pytree style!
    """

    x, y = jnp.arange(1.), 4.
    out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)
    print(out, type(out[0]))
    assert isinstance(out, tuple)
    out = pmap(lambda x, y: ((x + y, 1), y * 2.), in_axes=(0, None))(x, y)
    print(out, type(out[0]))
    assert isinstance(out, tuple) and isinstance(out[0], tuple)


@partial(pmap, axis_name='i')
def selfplay(key):
    data = jnp.ones((2, 3))
    return data


def test_pmap_selfplay():
    key = jrandom.PRNGKey(0)
    assert len(jax.local_devices()) == 1
    keys = jrandom.split(key, 1)
    print(keys.shape, keys)
    data = selfplay(keys)
    print(data.shape)
    assert data.shape[0] == 1


@partial(pmap, axis_name="i", in_axes=(None, None, 0), out_axes=(None, None, 0, 0))
def train(model, data):
    new_model = model
    loss = 1
    return new_model, loss


def test_pmap_train():
    """ would pmap add extra axis in output: out_axes=None
    train() uses: grads = jax.lax.pmean(grads, axis_name="i")
    so model can be unmapped thruout
    """


def test_vmap_out():
    # out_axes=None: Only for unmapped results we can specify out_axes to be None (to keep it unmapped)
    data_in = jnp.arange(1.), 4.
    print(data_in)
    out = vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(*data_in)
    print(out)


def test_vmap_collective():
    """
    axis_name: identify the mapped axis so that parallel collectives can be applied.
    """
    xs = jnp.arange(3. * 4.).reshape(3, 4)
    print('\n', xs)
    out = vmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
    print(out)
    assert xs.shape == out.shape


def test_replicate():
    xs = jnp.arange(6).reshape((2, 3))
    ds = jax.device_put_replicated(xs, jax.local_devices())
    print(ds.shape)
    ys = jax.device_get(ds[0])
    print(ys.shape)
    assert (xs == ys).all()