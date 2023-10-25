import jax
from jax import numpy as jnp
from jax import tree_util as jtu
import optax
from optax import GradientTransformation
import equinox as eqx
from jax import Array
from typing import Any
import functools
import torch

def count_tokens(input_ids: Array, ignore_index=0):
    return jnp.sum(input_ids != ignore_index)

def pytorch_to_np(batch):
    return jtu.tree_map(lambda x: x.numpy(), batch)

def reduce_state(state, new_state, reduce_fn=lambda x: jnp.mean(x, axis=0)):
    '''
    if a new axis as been added to state via vmap, reduce it back out
    '''
    return jtu.tree_map(
        lambda s, n_s: n_s if len(s.shape) == len(n_s.shape) else reduce_fn(n_s),
        state,
        new_state
    )


def tree_norm(tree):
    return jnp.sqrt(
        jtu.tree_reduce(
            lambda x, y: x + y,
            jtu.tree_map(lambda x: jnp.sum(x * x), eqx.filter(tree, eqx.is_array)),
        )
    )

# TODO: This is hella slow. Needs better solution
def log_optax(base_optimizer, log_fn):
    def init_fn(params):
        return base_optimizer.init(params)

    def update_fn(updates, state, params):
        log_fn(updates, state, params)
        return base_optimizer.update(updates, state, params)

    return GradientTransformation(init_fn, update_fn)


# basically the same as the pytorch function cross_entropy
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
