import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Tuple, Dict, Callable
from jaxtyping import Array
from jax import tree_util as jtu
import util
import equinox as eqx
from omegaconf import DictConfig, OmegaConf


def get_accuracy(
    scores: Array,
    target: Array,
    axis: int = -1,
    reduce: str = "mean",
    ignore_index: int = -100,
):
    """**Arguments:**
    - `scores`:  Scores for each class. Should have at least one dimension of size `C`
        where `C` is the number of classes. The "prediction" is the class with the
        highest score.
    - `target`: target class indices. The shape should match `scores` except for the
        dimension of size `C`, which should be of size 1.
    - `axis`: which dimension of `scores` is the size `C` dimension to argmax over when finding predictions.
    - `reduce`: post-processing on the errors. Can be one of `'mean'`, `'sum'` or `'none'`.
        In the case of `'none'`, all the prediction errors are returned.
    - `ignore_index`: Do not compute accuracy for indices whose target is this value.

    **Returns:**
    The average or total number of errors, or a boolean Array specifying where the errors are.
    """
    predictions = jnp.argmax(scores, axis=axis)
    if reduce == "none":
        return predictions
    if reduce == "mean":
        return jnp.sum(predictions == target) / jnp.sum(target != ignore_index)
    if reduce == "sum":
        return jnp.sum(predictions == target)
    else:
        raise ValueError(f"Unknown reduce option: {reduce}")


@jtu.Partial
def regression_loss_fn(
    model: eqx.Module,
    state: eqx.nn.State,
    batch: Dict[str, Array],
    error_fn: Callable[Array, Array],
    target_key="target",
    input_key="input",
    *,
    key: Array,
):
    def single_example_loss_fn(input, target, state):
        predictions, state = model(input, state=state, key=key)
        loss = error_fn(predictions, target)

        return loss, predictions, state

    vmapped_loss_fn = jax.vmap(
        single_example_loss_fn,
        in_axes=(0, 0, None),
        out_axes=(0, 0, None),
        axis_name="batch",
    )
    input = batch[input_key]
    target = batch[target_key]
    loss, predictions, state = vmapped_loss_fn(input, target, state)
    loss = jnp.mean(loss)
    log_data = {}
    return jnp.mean(loss), (state, log_data)


def classification_loss_fn(
    model: eqx.Module,
    state: eqx.nn.State,
    batch: Dict[str, Array],
    input_key="input_ids",
    target_key="labels",
    *,
    key: Array,
):
    def single_example_loss_fn(input, target, state):
        logits, state = model(input, state=state, key=key)
        loss = util.softmax_cross_entropy(logits, target)
        return loss, logits, state

    vmapped_loss_fn = jax.vmap(
        single_example_loss_fn,
        in_axes=(0, 0, None),
        axis_name="batch",
        out_axes=(0, 0, None),
    )
    input = batch[input_key]
    target = batch[target_key]
    loss, logits, new_state = vmapped_loss_fn(input, target, state)

    accuracy = get_accuracy(logits, target)

    loss = jnp.mean(loss)
    log_data = {
        "accuracy": accuracy,
    }

    return loss, (new_state, log_data)


LOSS_FN_REGISTRY = {}
LOSS_FN_REGISTRY["lm_classification"] = jtu.Partial(classification_loss_fn)
LOSS_FN_REGISTRY["tuple_classification"] = jtu.Partial(
    classification_loss_fn, input_key=0, target_key=1
)
LOSS_FN_REGISTRY["mean_squared_loss"] = jtu.Partial(
    regression_loss_fn, error_fn=lambda p, y: jnp.mean((p - y) ** 2)
)
LOSS_FN_REGISTRY["mean_abs_loss"] = jtu.Partial(
    regression_loss_fn, error_fn=lambda p, y: jnp.mean(jnp.abs(p - y))
)
LOSS_FN_REGISTRY["mean_norm_loss"] = jtu.Partial(
    regression_loss_fn, error_fn=lambda p, y: jnp.sqrt(jnp.sum((p - y) ** 2))
)


def get_loss(config: DictConfig):
    loss_fn = LOSS_FN_REGISTRY[config.train.loss_fn]

    if config.train.get("loss_fn_args", None):
        print("container: ")
        print(OmegaConf.to_container(config.train.loss_fn_args))
        loss_fn = jtu.Partial(
            loss_fn, **OmegaConf.to_container(config.train.loss_fn_args)
        )
    return loss_fn
