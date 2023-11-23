import optax
from jaxtyping import PyTree
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import numpy
from typing import NamedTuple, Any, Optional, List, Tuple
import util
import chex


def update_opt_moment_per_elem_norm(
    updates, prev_updates, moments, decay, order
):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g**order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return jnp.square(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, g_p, t: (1 - decay) * orderth_norm(g - g_p) + decay * t,
      updates,
      prev_updates,
      moments,
  )

def update_moment(
    updates, moments, decay, order
):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g**order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return jnp.square(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: -(1 - decay) * orderth_norm(g) + decay * t,
      updates,
      moments,
  )

class ScaleByOptLAProp(NamedTuple):
  """State for the Adam algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates
  g: optax.Updates


def scale_by_opt_laprop(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> optax.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = optax._src.utils.utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
    )
    g = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
    )
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByOptLAProp(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        nu=nu,
        g=g,
    )

  def update_fn(updates, state, params=None):
    del params
    g = updates

    nu = update_opt_moment_per_elem_norm(updates, state.g, state.nu, b2, 2)

    def get_opt_update(g, gp, v, vp):
      return (2 * g / (jnp.sqrt(v + eps_root) + eps)) - (
          gp / (jnp.sqrt(vp + eps_root) + eps)
      )
    opt_updates = jax.tree_util.tree_map(
        get_opt_update, updates, state.g, nu, state.nu
    )

    mu = update_moment(opt_updates, state.mu, b1, 1)
    count_inc = optax._src.numerics.safe_int32_increment(state.count)
    mu = optax._src.utils.cast_tree(mu, mu_dtype)
    return mu, ScaleByOptLAProp(
        count=count_inc,
        mu=mu,
        nu=nu,
        g=g,
    )

  return optax.GradientTransformation(init_fn, update_fn)
