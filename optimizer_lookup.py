import equinox as eqx
import jax
import optax
from jax import tree_util as jtu
from jax import numpy as jnp
from jaxtyping import PyTree
from omegaconf import DictConfig
from typing import Optional, Any, Callable
from util import log_optax, key_tree, zeros_like
from duration import Duration
from typing import NamedTuple
import time
import mechanic as new_mechanic
import duration
import logstate



class NoiseState(NamedTuple):
    key: jax.Array


def add_noise(sigma: float, key: jax.Array):
    def init_fn(params):
        state = NoiseState(key)
        return state

    def update_fn(updates, state, params):
        to_use, key = jax.random.split(state.key)
        state = NoiseState(key)

        to_use = key_tree(to_use, updates)
        updates = jtu.tree_map(
            lambda u_i, k_i: u_i + sigma * jax.random.normal(k_i, u_i.shape, u_i.dtype),
            updates,
            to_use,
        )

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def all_finite(tree: PyTree) -> jax.Array:
    tree = jtu.tree_map(lambda x: jnp.all(jnp.isfinite(x)), tree)
    leaves = jtu.tree_flatten(tree)[0]
    return jnp.all(jnp.array(leaves))


def skip_nonfinite(opt: optax.GradientTransformation) -> optax.GradientTransformation:
    def init_fn(params: optax.Params):
        return opt.init(params)

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None,
    ):
        next_updates, next_state = opt.update(updates, state, params)
        return jax.lax.cond(
            all_finite((next_updates, next_state)),
            lambda: (next_updates, next_state),
            lambda: (zeros_like(updates), state),
        )

    return optax.GradientTransformation(init_fn, update_fn)




def zero_if_nan(updates, params):
    return jax.lax.cond(all_finite(updates), lambda x: x, zeros_like, updates)


class AnytimeAvgState(NamedTuple):
    iteration: jax.Array
    momentum: PyTree


def anytime_avg():
    def init_fn(params):
        state = AnytimeAvgState(
            iteration=jnp.array(0), momentum=jtu.tree_map(jnp.zeros_like, params)
        )
        return state

    def update_fn(updates, state, params):
        iteration = state.iteration + 1
        beta = (iteration - 1) / (iteration + 1)
        momentum = jtu.tree_map(
            lambda m, u: m * beta + u / 2 * (1 - beta), state.momentum, updates
        )

        state = AnytimeAvgState(
            iteration=iteration,
            momentum=momentum,
        )
        return momentum, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_schedule_logged(schedule_fn):
    def init_fn(self):
        return logstate.LoggedState((jnp.array(0), duration.JaxTimeStamp()), log_data={"lr/schedule": jnp.array(0.0)})

    def update_fn(updates, state, params):
        (count, timestamp), log_data = state
        count = count + 1

        schedule = schedule_fn(count, timestamp)
        updates = jtu.tree_map(lambda x: schedule * x, updates)
        log_data = {"lr/schedule": schedule}
        return updates, logstate.LoggedState(
            state=(count, timestamp), log_data=log_data
        )

    return optax.GradientTransformation(init_fn, update_fn)


def schedule_fn(
    count: int,
    timestamp: duration.JaxTimeStamp,
    config: DictConfig,
    train_duration: Duration,
    loader: Any,
    peak: float,
    # logger: Optional[Callable] = None,
):
    if train_duration.minutes != float("inf"):
        train_elapsed = jnp.asarray(
            (timestamp.timestamp / 60 - train_duration.start_time)
            / train_duration.minutes
        )
    else:
        if train_duration.iterations != float("inf"):
            max_iter = train_duration.iterations
        if train_duration.epochs != float("inf"):
            max_iter = len(loader) * train_duration.epochs
        train_elapsed = count / max_iter

    warmup = config.lr_warmup

    fraction_remaining = (1 - train_elapsed) / (1 - warmup)
    if config.lr_decay == "linear":
        decay_value = fraction_remaining
    elif config.lr_decay == "cosine":
        decay_value = jnp.cos(fraction_remaining * jnp.pi) * 0.5 + 0.5
    else:
        decay_value = 1.0

    result = peak * jax.lax.select(
        train_elapsed < warmup,
        train_elapsed / warmup,
        decay_value,
    )
    # if logger is not None:
    #     jax.experimental.io_callback(
    #         logger, None, {"lr/schedule": result}, commit=False
    #     )
    return result


def get_optimizer(
    config: DictConfig,
    model: eqx.Module,
    train_duration: Duration,
    train_loader,
    logger: Optional[Callable] = None,
):
    if not config.log_callback_data:
        logger = None
    # total_steps = config.max_steps
    opt_config = config.optim
    schedule = jtu.Partial(
        schedule_fn,
        loader=train_loader,
        train_duration=train_duration,
        config=opt_config,
        peak=opt_config.lr,
        # logger=logger,
    )

    # set the learning rate to 1.0 here - we will scale
    # by the learning rate schedule later.
    if opt_config.name == "sgd":
        optimizer = optax.chain(
            optax.add_decayed_weights(opt_config.weight_decay),
            optax.sgd(learning_rate=1.0, momentum=opt_config.momentum),
        )
    elif opt_config.name == "adamw":
        optimizer = optax.adamw(learning_rate=1.0, weight_decay=opt_config.weight_decay)

    if opt_config.bake_schedule:
        optimizer = optax.chain(optimizer, scale_by_schedule_logged(schedule))

    if opt_config.mechanize:
        if opt_config.mechanize == "optax":
            optimizer = optax.contrib.mechanize(
                optimizer, weight_decay=opt_config.mech_lambda
            )
        elif opt_config.mechanize == "new":
            optimizer = new_mechanic.mechanize(optimizer)
        elif opt_config.mechanize == "nobeta":
            optimizer = new_mechanic.mechanize_no_beta(optimizer)
        else:
            raise ValueError(f"unknown mechanize option: {opt_config.mechanize}")
        if logger is not None:

            def log_fn(updates, state, params):
                return {"mechanic/s": jnp.sum(state.s)}

            optimizer = log_optax(optimizer, log_fn)

    # else:
    #     optimizer = optax.chain(optimizer, optax.scale(opt_config.lr))

    if not opt_config.bake_schedule:
        optimizer = optax.chain(optimizer, scale_by_schedule_logged(schedule))
        # if not opt_config.mechanize:
        #     optimizer = optax.chain(optimizer, optax.scale(opt_config.lr))

    # optimizer = optax.apply_if_finite(optimizer, 15)
    optimizer = skip_nonfinite(optimizer)
    # optimizer = optax.chain(optimizer, optax.stateless(zero_if_nan))

    # we do gradient clipping before anything else
    if opt_config.gradient_clip_val is not None:
        grad_clip = optax.clip_by_global_norm(opt_config.gradient_clip_val)
        optimizer = optax.chain(grad_clip, optimizer)

    if config.averaging == "anytime":
        optimizer = optax.chain(optimizer, anytime_avg())

    if config.get("gradient_noise", 0) != 0:
        optimizer = optax.chain(
            add_noise(config.gradient_noise, jax.random.PRNGKey(1231)), optimizer
        )

    opt_state = optimizer.init(jtu.tree_map(jnp.array, eqx.filter(model, eqx.is_array)))
    return optimizer, opt_state
