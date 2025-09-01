""" copied from a0jax, pax -> eqx """
import chex
import jax
import jax.numpy as jnp
import equinox as eqx


class ResidualBlock(eqx.Module):
    """A residual block of conv layers

    y(x) = x + F(x),

    where:
        F = BatchNorm >> relu >> Conv1 >> BatchNorm >> relu >> Conv2
    """
    layers: list

    def __init__(self, dim: int) -> None:
        self.layers = [
            eqx.nn.BatchNorm(),
            jax.nn.relu,
            eqx.nn.Conv2d(),
            eqx.nn.BatchNorm(),
            jax.nn.relu,
            eqx.nn.Conv2d(),
        ]
        self.batchnorm1 = eqx.nn.BatchNorm2D(dim, True, True)
        self.batchnorm2 = eqx.BatchNorm2D(dim, True, True)
        self.conv1 = eqx.Conv2D(dim, dim, 3)
        self.conv2 = eqx.Conv2D(dim, dim, 3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        t = x
        t = jax.nn.relu(self.batchnorm1(t))
        t = self.conv1(t)
        t = jax.nn.relu(self.batchnorm2(t))
        t = self.conv2(t)
        return x + t


class ResnetPolicyValueNet(eqx.Module):
    """Residual Conv Policy-Value network.

    Two-head network:
                        ┌─> action head
    input ─> Backbone  ─┤
                        └─> value head
    """

    def __init__(
        self, input_dims, num_actions: int, dim: int = 64, num_resblock: int = 5
    ) -> None:
        super().__init__()
        if len(input_dims) == 3:
            num_input_channels = input_dims[-1]
            input_dims = input_dims[:-1]
            self.has_channel_dim = True
        else:
            num_input_channels = 1
            self.has_channel_dim = False

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = eqx.Sequential(
            eqx.nn.Conv2d(num_input_channels, dim, 1), eqx.BatchNorm2D(dim)
        )
        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = eqx.Sequential(
            eqx.Conv2D(dim, dim, 1),
            eqx.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            eqx.Conv2D(dim, self.num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = eqx.Sequential(
            eqx.Conv2D(dim, dim, 1),
            eqx.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            eqx.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            eqx.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            eqx.Conv2D(dim, 1, kernel_shape=1, padding="VALID"),
            jnp.tanh,
        )

    def __call__(self, x: chex.Array, batched: bool = False):
        """Compute the action logits and value.

        Support both batched and unbatched states.
        """
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]  # add the batch dimension
        if not self.has_channel_dim:
            x = x[..., None]  # add channel dimension
        x = self.backbone(x)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        if batched:
            return action_logits[:, 0, 0, :], value[:, 0, 0, 0]
        else:
            return action_logits[0, 0, 0, :], value[0, 0, 0, 0]


class ResnetPolicyValueNet128(ResnetPolicyValueNet):
    """Create a resnet of 128 channels, 5 blocks"""

    def __init__(
        self, input_dims, num_actions: int, dim: int = 128, num_resblock: int = 5
    ):
        super().__init__(input_dims, num_actions, dim, num_resblock)


class ResnetPolicyValueNet256(ResnetPolicyValueNet):
    """Create a resnet of 256 channels, 6 blocks"""

    def __init__(
        self, input_dims, num_actions: int, dim: int = 256, num_resblock: int = 6
    ):
        super().__init__(input_dims, num_actions, dim, num_resblock)
