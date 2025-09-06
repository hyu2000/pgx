from typing import NamedTuple
import jax
import jax.numpy as jnp


@jax.jit
def pure_uses_internal_state(x):
  state = dict(even=0, odd=0)
  for i in range(10):
    state['even' if i % 2 == 0 else 'odd'] += x
  return state['even'] + state['odd']


def test_pure1():
    assert pure_uses_internal_state(5.) == 50
    assert pure_uses_internal_state(6.) == 60


def test_sample_vectorized():
    rng_key = jax.random.PRNGKey(0)
    logits = jnp.log(jnp.array([0.2, 0.7, 0.1]))
    N = 50
    vlogits = jnp.tile(logits, N).reshape((N, 3))
    print(vlogits.shape)
    print(vlogits[[0, 3], :])
    actions = jax.random.categorical(rng_key, logits=vlogits, axis=-1)
    print(actions)
    probs = jnp.bincount(actions) / N
    print(probs)


class OpTrace(NamedTuple):
    state: jnp.array
    x: jnp.array
    result: jnp.array


def test_scan():
    """ scan(f)  f can output a pytree """
    def cumsum_scan(carry, x):
        new_carry = carry + x
        return new_carry, OpTrace(carry, x, new_carry)

    # Initial state is 0, scan over [1, 2, 3, 4, 5]
    xs = jnp.array([1, 2, 3, 4, 5])
    # final_carry, outputs = jax.lax.scan(cumsum_scan, 0, xs)
    # assert final_carry == xs.sum()

    inits = jnp.arange(3)
    final_carry, outputs = jax.lax.scan(cumsum_scan, inits, xs)
    assert isinstance(outputs, OpTrace)
    assert(outputs.result.shape == (len(xs), len(inits)))
    print(final_carry, outputs.result)


def test_vmap():
    def cumsum_scan(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    print('orig')
    print(jax.make_jaxpr(cumsum_scan)(0., 0.))
    xs = jnp.zeros(2)
    cumsum_v = jax.vmap(cumsum_scan)
    print(jax.make_jaxpr(cumsum_v)(xs, xs))


def test_device_put_replicated():
    """ device_put_replicated() adds a leading device dimension """
    x = jnp.ones(5)
    dx = jax.device_put_replicated(x, jax.devices())
    print(dx.shape)
    assert x.ndim + 1 == dx.ndim
    print(dx.sharding)


class DataWithMask(NamedTuple):
    x: jnp.array
    mask: jnp.array


def test_filter_pytree():
    """ """
    x = jnp.arange(10)
    samples = DataWithMask(x=x, mask=x % 2 == 0)
    x_filtered = jax.tree_map(lambda x: x[samples.mask], samples)
    print(x_filtered)


def test_jnp_indexing():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 2))
    print(x[:3])


def test_cumsum():
    """ value mask in compute_loss_input """
    # selfplay: scan on two simultaneous games
    terminated = jnp.array([
        [0, 0, 0, 0, True, 0, 0, 0],  # first game ended in 4 moves; 2nd game didn't end in 4 moves
        [0, 0, 0, True, 0, 0, True, 0]
    ]).T
    value_mask = jnp.cumsum(terminated[::-1, :], axis=0)[::-1, :] >= 1
    print()
    print(value_mask)
