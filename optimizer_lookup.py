import equinox as eqx
import jax
import optax
from jax import tree_util as jtu
from jax import numpy as jnp
from omegaconf import DictConfig
from typing import Optional, Any, Callable
from util import log_optax
from duration import Duration
import time


def schedule_fn(
    count: int,
    config: DictConfig,
    train_duration: Duration,
    loader: Any,
    peak: float,
    logger: Optional[Callable] = None,
):
    if train_duration.minutes != float("inf"):
        train_elapsed = jax.experimental.io_callback(
            lambda: jnp.asarray(
                (time.time() / 60 - train_duration.start_time) / train_duration.minutes
            ),
            jnp.asarray(1.0),
        )
    else:
        if train_duration.iterations != float("inf"):
            max_iter = train_duration.iterations
        if train_duration.epochs != float("inf"):
            max_iter = len(loader) * train_duration.epochs
        train_elapsed = count / max_iter

    warmup = config.lr_warmup

    result = peak * jax.lax.select(
        train_elapsed < warmup,
        train_elapsed / warmup,
        (1 - train_elapsed) / (1 - warmup),
    )
    if logger is not None:
        jax.experimental.io_callback(
            logger, None, {"lr/schedule": result}, commit=False
        )
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
    total_steps = config.max_steps
    schedule = jtu.Partial(
        schedule_fn,
        loader=train_loader,
        train_duration=train_duration,
        config=config,
        peak=1.0,
        logger=logger,
    )
    if config.bake_schedule:
        base_schedule = schedule
    else:
        base_schedule = 1.0

    if config.optimizer == "sgd":
        optimizer = optax.chain(
            optax.add_decayed_weights(config.wd),
            optax.sgd(learning_rate=base_schedule, momentum=config.mom),
        )
    elif config.optimizer == "adamw":
        optimizer = optax.adamw(
            learning_rate=base_schedule, weight_decay=config.weight_decay
        )

    if config.mechanize:
        optimizer = optax.contrib.mechanize(optimizer, weight_decay=config.mech_lambda)
        if logger is not None:

            def log_fn(updates, state, params):
                jax.experimental.io_callback(
                    logger, None, {"mechanic/s": jnp.sum(state.s)}, commit=False
                )

            optimizer = log_optax(optimizer, log_fn)

    else:
        optimizer = optax.chain(optimizer, optax.scale(config.lr))

    if not config.bake_schedule:
        optimizer = optax.chain(optimizer, optax.scale_by_schedule(schedule))
        # if not config.mechanize:
        #     optimizer = optax.chain(optimizer, optax.scale(config.lr))

    optimizer = optax.apply_if_finite(optimizer, 15)

    # we do gradient clipping before anything else
    grad_clip = optax.clip_by_global_norm(config.gradient_clip_val)
    optimizer = optax.chain(grad_clip, optimizer)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    return optimizer, opt_state
