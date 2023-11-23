import optax
from jaxtyping import PyTree
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import numpy
from typing import NamedTuple, Any, Optional, List, Tuple
import util


class OptAdamState(NamedTuple):
    sum_squared_grad_diff: optax.Updates
    momentum: optax.Updates
    previous_grad: optax.Updates


def opt_adam(
    beta1: float=0.9,
    beta2: float=0.95,
    weight_decay: float=0.01,
    epsilon: float = 1e-8,
    use_max: bool = False
) -> optax.GradientTransformation:
    
    def init_fn(params: optax.Params):
        state  = OptAdamState(
            sum_squared_grad_diff = jtu.tree_map(jnp.zeros_like, params),
            momentum = jtu.tree_map(jnp.zeros_like,  params),
            previous_grad = jtu.tree_map(jnp.zeros_like, params),
        )
        return state

    def update_fn(
        updates: optax.Updates, state: OptAdamState, params: optax.Params
    ):
        
        sum_squared_grad_diff = state.sum_squared_grad_diff
        momentum = state.momentum
        previous_grad = state.previous_grad

        def update_v(v_i, g_i, prev_g_i):
            result = beta2 * v_i + (1-beta2)*(g_i-prev_g_i)**2
            if use_max:
                result = jnp.maximum(result, (1-beta2)*g_i**2)
            return result

        next_sum_squared_grad_diff = jtu.tree_map(
            update_fn,
            state.sum_squared_grad_diff,
            updates,
            state.previous_grad
        )

        def update_m(m_i, v_i, prev_v_i, g_i, prev_g_i):
            result = beta1 * m_i + (2* g_i/(jnp.sqrt(v_i) + epsilon) - prev_g_i/(jnp.sqrt(v_i) + epsilon))
            return result

        next_momentum = jtu.tree_map(
            update_m,
            state.momentum,
            next_sum_squared_grad_diff,
            state.sum_squared_grad_diff,
            updates,
            state.previous_grad
        )

        def add_weight_decay(m_i, x_i):
            return -m_i - x_i * weight_decay

        next_updates = jtu.tree_map(
            next_momentum,
            params,
        )

        next_state = OptAdamState(
            sum_squared_grad_diff=next_sum_squared_grad_diff,
            momentum=next_momentum,
            previous_grad=updates
        )

        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)
