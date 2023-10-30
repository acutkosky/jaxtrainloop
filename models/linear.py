import equinox as eqx
from equinox import nn
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from typing import Optional
from omegaconf import DictConfig

class LinearModel(eqx.Module):
    lin: nn.Linear

    def __init__(self, feature_dim, label_dim, use_bias=True, zero_init=True, *, key: jax.Array):
        self.lin = nn.Linear(feature_dim, label_dim, use_bias, key=key)

        if zero_init:
            self.lin = jtu.tree_map(jnp.zeros_like, self.lin)


    def __call__(self, x: jax.Array , state: Optional[nn.State]=None, *, key: Optional[jax.Array]=None):
        return self.lin(x), state



