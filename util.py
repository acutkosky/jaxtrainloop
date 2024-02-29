import jax
from jax import numpy as jnp
from jax import tree_util as jtu
import optax
from optax import GradientTransformation
import equinox as eqx
from jax import Array
from jaxtyping import PyTree
from typing import Any
import functools
import torch
import logstate


def tree_reduce_sum(t):
    return jnp.sum(
        jnp.array(
            jtu.tree_leaves(jtu.tree_map(jnp.sum, t))
        )
    )

def map_over_lists(func, *ls):
    return jtu.tree_map(func, *ls, is_leaf=lambda x: x not in ls)


def tree_sq_norm(t):
    return jtu.tree_reduce(
        lambda x, y: x + y, jtu.tree_map(lambda p: jnp.sum(p**2), t)
    )


def tree_norm(t):
    return jnp.sqrt(tree_sq_norm(t))


def tree_copy(t):
    return jtu.tree_map(jnp.copy, t)


def tree_add(a, b):
    return jtu.tree_map(lambda a_i, b_i: a_i + b_i, a, b)


def tree_reduce_mean(t):
    total = jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(jnp.sum, t))
    count = len(jtu.tree_leaves(jtu.tree_map(lambda x: x.size, t)))
    return total / count


def tree_subtract(a, b):
    return jtu.tree_map(lambda a_i, b_i: a_i - b_i, a, b)


def tree_dot_per_layer(v, w):
    return jtu.tree_map(lambda vi, wi: jnp.sum(vi * wi), v, w)


def tree_scalar_mul(t, m):
    return jtu.tree_map(lambda t_i: t_i * m, t)


def tree_dot(v, w):
    return jtu.tree_reduce(lambda x, y: x + y, tree_dot_per_layer(v, w))


def same_shape_leaf(l1, l2):
    if eqx.is_array(l1) != eqx.is_array(l2):
        return False
    if not eqx.is_array(l1):
        return l1 == l2
    if l1.shape != l2.shape:
        return False
    if l1.dtype != l2.dtype:
        return False
    return True


def same_shape_pytree(tree1, tree2):
    try:
        return jtu.tree_all(jtu.tree_map(same_shape_leaf, tree1, tree2))
    except ValueError:
        return False


def key_tree(key, tree):
    leaves, treedef = jtu.tree_flatten(tree)
    key_leaves = jax.random.split(key, len(leaves))
    return jtu.tree_unflatten(treedef, key_leaves)


def count_tokens(input_ids: Array, ignore_index=0):
    return jnp.sum(input_ids != ignore_index)


def pytorch_to_np(batch):
    return jtu.tree_map(lambda x: x.numpy(), batch)


def reduce_state(state, new_state, reduce_fn=lambda x: jnp.mean(x, axis=0)):
    """
    if a new axis as been added to state via vmap, reduce it back out
    """
    return jtu.tree_map(
        lambda s, n_s: n_s if len(s.shape) == len(n_s.shape) else reduce_fn(n_s),
        state,
        new_state,
    )


def zeros_like(tree: PyTree) -> PyTree:
    return jtu.tree_map(jnp.zeros_like, tree)


def tree_dot_per_layer(v, w):
    return jtu.tree_map(lambda vi, wi: jnp.sum(vi * wi), v, w)


def tree_dot(v, w):
    return jtu.tree_reduce(lambda x, y: x + y, tree_dot_per_layer(v, w))


def tree_norm(tree, ord=2):
    if ord == "infty":
        return jtu.tree_reduce(
            lambda x, y: jnp.maximum(x, y),
            jtu.tree_map(lambda x: jnp.max(jnp.abs(x)), eqx.filter(tree, eqx.is_array)),
        )
    return (
        jtu.tree_reduce(
            lambda x, y: x + y,
            jtu.tree_map(
                lambda x: jnp.sum(jnp.abs(x) ** ord), eqx.filter(tree, eqx.is_array)
            ),
        )
    ) ** (1.0 / ord)


def merge_dicts(*to_merge):
    result = {}
    for d in to_merge:
        result.update(d)
    return result


# TODO: This is hella slow. Needs better solution
def log_optax(base_optimizer, log_fn, pruned=False):
    def init_fn(params):
        state = base_optimizer.init(params)
        if pruned:
            return state
        log_data = log_fn(zeros_like(params), state, params)
        return (state, logstate.Log(log_data))

    def update_fn(updates, state, params):
        if not pruned:
            state, _ = state
        log_data = log_fn(updates, state, params)
        updates, state = base_optimizer.update(updates, state, params)
        state = (state, logstate.Log(log_data))
        return updates, state

    return GradientTransformation(init_fn, update_fn)


# basically the same as the pytorch function cross_entropy
@jax.named_scope("softmax_cross_entropy")
def softmax_cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    axis=None,
):
    """Computes softmax cross entropy between sets of logits and integer labels.
    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.
    References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
    Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
        `[...]`.
    Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
    """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.

    if axis is None:
        axis = input.ndim - 1
    if axis < 0:
        axis = input.ndim + axis

    C = input.shape[axis]

    if weight is not None:
        weight_shape = (
            (1,) * axis + (input.shape[axis],) + (1,) * (input.ndim - axis - 1)
        )
        weight = weight.reshape(weight_shape)

    if isinstance(target, int) or target.ndim != input.ndim:
        no_ignore = jax.lax.stop_gradient(target != ignore_index)
        logits_max = jnp.max(
            input, axis=axis, keepdims=True
        )  # , where=no_ignore, initial=-jnp.inf)
        logits = input - jax.lax.stop_gradient(logits_max)

        broadcast_shape = logits.shape[:axis] + (1,) + logits.shape[axis + 1 :]

        log_normalizers = jnp.log(
            jnp.sum(
                jnp.exp(logits), axis=axis, where=no_ignore.reshape(broadcast_shape)
            )
        )

        labels_no_ignore = jnp.where(no_ignore, target, 0)

        label_logits = jnp.take_along_axis(
            logits, labels_no_ignore[..., None], axis=axis
        )[..., 0]

        if label_smoothing != 0 or weight is not None:
            one_hot_labels = jax.nn.one_hot(labels_no_ignore, num_classes=C, axis=axis)
            target_probs = (
                one_hot_labels * (1.0 - label_smoothing)
                + jnp.ones_like(one_hot_labels) / C * label_smoothing
            )

            if weight is not None:
                target_probs = target_probs * weight
                log_normalizers = log_normalizers * jnp.sum(target_probs, axis=axis)

            losses = -(
                jnp.sum(
                    target_probs * logits,
                    where=no_ignore.reshape(broadcast_shape),
                    axis=axis,
                )
                - log_normalizers
            )
        else:
            label_logits = jnp.take_along_axis(
                logits, labels_no_ignore[..., None], axis=axis
            )[..., 0]
            losses = log_normalizers - label_logits

        losses = jnp.where(no_ignore, losses, 0.0)
    else:
        target_probs = (
            target * (1.0 - label_smoothing)
            + jnp.ones_like(target) / C * label_smoothing
        )

        logits_max = jnp.max(input, axis=axis, keepdims=True)
        logits = input - jax.lax.stop_gradient(logits_max)

        log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=axis))

        if weight is not None:
            target_probs = target_probs * weight
            log_normalizers = log_normalizers * jnp.sum(
                target_probs * weight, axis=axis
            )

        losses = -(jnp.sum(target_probs * logits, axis=axis) - log_normalizers)

        no_ignore = None

    if reduction == "none":
        return losses
    if reduction == "mean":
        return jnp.mean(losses, where=no_ignore)
    if reduction == "sum":
        return jnp.sum(losses, where=no_ignore)
    else:
        raise ValueError(f"unknown reduction type: {reduction}")
