from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch
from typing import Tuple
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping


class CNN(eqx.Module):
    layers: list

    def __init__(self, key, bn_mode='ema'):
        keys = jax.random.split(key, 5)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=3, padding='SAME', key=keys[0]),
            eqx.nn.BatchNorm(3, axis_name='batch', mode=bn_mode),
            jax.nn.relu,
            eqx.nn.Conv2d(3, 3, kernel_size=4, key=keys[1]),  # 4x4, padding=valid: 28x28 -> 25x25
            eqx.nn.MaxPool2d(kernel_size=2),  # 2x2, valid: H-1, W-1
            eqx.nn.BatchNorm(3, axis_name='batch', mode=bn_mode),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=keys[2]),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=keys[4]),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"], state: eqx.nn.State) -> Tuple[Float[Array, "10"], eqx.nn.State]:
        """ single sample """
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state

    def batch_call(self, x_batch: Float[Array, "batch 1 28 28"], state: eqx.nn.State) -> Tuple[Float[Array, "batch 10"], eqx.nn.State]:
        """ batch eval """
        return jax.vmap(self.__call__, axis_name='batch', in_axes=(0, None), out_axes=(0, None))(x_batch, state)


@eqx.filter_jit
def loss_fn(
    model: CNN, state: eqx.nn.State, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Tuple[Float[Array, ""], eqx.nn.State]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y, state = model.batch_call(x, state)
    return cross_entropy(y, pred_y), state


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


# Evaluation

@eqx.filter_jit
def compute_accuracy(
    inference_model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y, _ = jax.vmap(inference_model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(model: CNN, state: eqx.nn.State, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)
    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        batch_loss, _ = loss_fn(model, state, x, y)
        avg_loss += batch_loss
        avg_acc += compute_accuracy(inference_model, x, y)
    return avg_loss / len(testloader), avg_acc / len(testloader)


def train_loop(
    model: CNN,
    state: eqx.nn.State,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> Tuple[CNN, eqx.nn.State]:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        state: eqx.nn.State,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        (loss_value, state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, state, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, state, opt_state, train_loss = make_step(model, state, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, state, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model, state


def test_cnn():
    key = jax.random.PRNGKey(0)
    cnn = CNN(key, bn_mode='ema')
    bn_layers = [l for l in cnn.layers if isinstance(l, eqx.nn.BatchNorm)]
    print(bn_layers)