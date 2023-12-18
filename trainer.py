import jax
from jax import numpy as jnp
import equinox as eqx
from jax import random as jr
from jax.random import PRNGKey, PRNGKeyArray
from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable, Dict
import hydra
from omegaconf import OmegaConf, DictConfig
import equinox as eqx
import tqdm
from jax import tree_util as jtu
import optax
from util import softmax_cross_entropy, tree_norm, log_optax
import util
import logging
import transformers
import ml_dtypes
import wandb
from collections import defaultdict
import time
import duration
from ratelogger import RateLimitedLog
import checkpoint

import numpy as np

from timekeeper import TimeKeeper

from dataset_lookup import get_loader
from model_lookup import get_model
from optimizer_lookup import get_optimizer
from losses import get_loss
import logstate


class ModelAndState(NamedTuple):
    model: eqx.Module
    state: Optional[eqx.nn.State]

class OptimizerAndState(NamedTuple):
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState

def apply_model(model: ModelAndState, *args, **kwargs):
    module = model.module
    state = model.state
    state = model.module(*args, state=state, **kwargs)
    return ModelAndState(module=module, state=state)


class TrainState(NamedTuple):
    model: ModelAndState
    optimizer: OptimizerAndState
    # opt_state: Any
    # optimizer: optax.GradientTransformation
    dynamic_scaler_state: Optional[DynamicScalerState]
    time: Dict[str, duration.TrainTime]
    aux: Any


def get_dtype(dtype: str):
    registry = {
        "bfloat16": ml_dtypes.bfloat16,
        "float16": jnp.float16,
    }

    return registry[dtype.lower()]


def inference_step(
    train_state: TrainState,
    batch: Any,
    loss_fn: Callable,
    prng_key: PRNGKeyArray,
    config: Any,
):
    if isinstance(train_state, logstate.LoggedState):
        train_state = train_state.get_state()
    model = train_state.model.model
    state = train_state.model.state

    loss, (state, log_data) = loss_fn(model, state, batch, key=prng_key)

    logged_state = logstate.LoggedState(train_state, log_data)
    all_logs = util.merge_dicts(*logstate.list_of_logs(logged_state))
    return loss, logged_state, all_logs


def update_avg_tree(tree, avg_tree, count):
    array, static = eqx.partition(tree, eqx.is_array)
    avg_array, _ = eqx.partition(avg_tree, eqx.is_array)

    array = jtu.tree_map(lambda a, t: a + (t - a) / count, avg_array, array)

    return eqx.combine(array, static)


def update_average(
    model: eqx.Module,
    state: Optional[eqx.nn.State],
    average: ModelAndState,
    iteration: jax.Array,
):
    model = update_avg_tree(model, average.model, iteration)
    if state is not None:
        state = update_avg_tree(state, average.state, iteration)

    return ModelAndState(model=model, state=state)


def train_step(
    train_state: TrainState,
    batch: Any,
    loss_fn: Callable,
    prng_key: PRNGKeyArray,
    config: Any,
):
    print("\ncompiling train step!\n")

    if isinstance(train_state, logstate.LoggedState):
        train_state = train_state.get_state()
    model = train_state.model.model
    state = train_state.model.state
    opt_state = train_state.optimizer.opt_state
    dynamic_scaler_state = train_state.dynamic_scaler_state
    optimizer = train_state.optimizer.optimizer
    aux = train_state.aux
    time = train_state.time

    if config.use_amp:
        value_and_grad_fn = dynamic_scale_value_and_grad(
            loss_fn, filter=True, has_aux=True, redo_on_nan=0
        )
        dynamic_scaler_state, (
            (loss, (state, log_data)),
            grads,
        ) = value_and_grad_fn(
            model,
            state,
            batch,
            key=prng_key,
            dynamic_scaler_state=dynamic_scaler_state,
        )
    else:
        value_and_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, (state, log_data)), grads = value_and_grad_fn(
            model, state, batch, key=prng_key
        )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)

    if config.averaging == "polyak":
        aux['polyak'] = update_average(model, state, aux['polyak'], time["train"].it + 1)

    new_train_state = TrainState(
        model=ModelAndState(model=model, state=state),
        optimizer=OptimizerAndState(optimizer=optimizer,opt_state=opt_state),
        # opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        # optimizer=optimizer,
        time=time,
        aux=aux,
    )

    if config.log_norms:
        log_data["norms/grads"] = tree_norm(grads)
        log_data["norms/params"] = tree_norm(model)

    logged_state = logstate.LoggedState(new_train_state, log_data)
    all_logs = util.merge_dicts(*logstate.list_of_logs(logged_state))
    return loss, logged_state, all_logs



def train_prepare(train_state, config):
    return eqx.tree_at(
        lambda t: t.model.model,
        train_state,
        eqx.nn.inference_mode(train_state.model.model, value=False),
    )


def train_break(train_state, config):
    return train_state


def valid_prepare(train_state, config):
    train_state = eqx.tree_at(
        lambda t: t.model.model,
        train_state,
        eqx.nn.inference_mode(train_state.model.model, value=True),
    )
    aux = train_state.aux
    model = train_state.model
    if config.train.averaging == "polyak":
        train_state = eqx.tree_at(
            lambda t: (t.model, t.aux['polyak']), train_state, replace=(aux['polyak'], model)
        )
    return train_state


def valid_break(train_state, config):
    aux = train_state.aux
    model = train_state.model
    if config.train.averaging == "polyak":
        train_state = eqx.tree_at(
            lambda t: (t.model, t.aux['polyak']), train_state, replace=(aux['polyak'], model)
        )

    return train_state


def copy_arrays(tree):
    array, static = eqx.partition(tree, eqx.is_array)
    array = jtu.tree_map(jnp.array, array)
    return eqx.combine(array, static)


def run_epoch(
    train_state: TrainState,
    dataloader: Any,
    pbar: Any,
    loss_fn: Callable,
    mode_start: duration.TrainDuration,
    mode_duration: duration.TrainDuration,
    checkpoint_manager: checkpoint.CheckpointManager,
    total_duration: duration.TrainDuration,
    config: DictConfig,
    mode: str,
    time_keeper: TimeKeeper,
    logger: Callable,
    batch_size_fn: Callable,
    prng_key: Array,
):
    mode = mode.lower()

    total_tokens = 0

    if mode == "train":
        step_fn_jit = eqx.filter_jit(
            jtu.Partial(train_step, loss_fn=loss_fn, config=config.train),
            donate="all",
        )
    else:
        step_fn_jit = eqx.filter_jit(
            jtu.Partial(inference_step, loss_fn=loss_fn, config=config.train),
            donate="all",
        )

    exhausted_loader = True
    iteration_timing_events = ["iteration", "dataloader", "train_step"]
    time_keeper.mark(start_events=["dataloader", "iteration", "tokens", "examples"])
    it = -1
    summary_metrics = {"loss": None, "accuracy": None}
    time_updater = duration.TimeUpdater()
    time_updater.start(mode)
    while True:
        try:
            iter_in_epoch, batch = next(dataloader)
        except StopIteration:
            break
        it += 1
        batch = util.pytorch_to_np(batch)

        # tricky: we need to compute the size info BEFORE sending to the
        # step_fn because we are going to donate all the buffers.
        batch_size = batch_size_fn(batch)

        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])
        to_use, prng_key = jr.split(prng_key)
        loss, train_state, log_data = step_fn_jit(train_state, batch, prng_key=to_use)
        train_state = time_updater(
            train_state.time[mode], train_state, **batch_size.dict()
        )

        total_time = sum(train_state.time.values())




        time_keeper.mark(
            end_events={"train_step": 1},
        )

        #### Record metrics that should be tagged with the current mode ####
        log_data["loss"] = loss
        log_data.update(batch_size.loggable_dict("batch/"))

        log_data["iter_in_epoch"] = iter_in_epoch

        time_keeper.mark(
            start_events=["dataloader", "iteration", "tokens", "examples"],
            end_events={
                "iteration": batch_size.it,
                "tokens": batch_size.tok,
                "examples": batch_size.ex,
            },
        )
        intervals = time_keeper.get_intervals()
        proportions = time_keeper.get_proportions()
        log_data.update(
            {
                f"time/secs_per/{k}": intervals[k]
                for k in iteration_timing_events
                if k in intervals
            }
        )

        if train_state.dynamic_scaler_state is not None:
            log_data.update(
                {
                    f"dynamic_scaler/scaler": np.array(
                        train_state.dynamic_scaler_state.scaler
                    ),
                    f"dynamic_scaler/total_resets": np.array(
                        train_state.dynamic_scaler_state.total_resets
                    ),
                }
            )
        log_data.update(
            {
                f"time/fraction_spent/{k}": proportions[k]
                for k in iteration_timing_events
                if k in proportions
            }
        )

        if "iteration" in intervals:
            throughput = {
                "throughput/iteration_per_sec": 1.0 / intervals["iteration"],
                "throughput/examples_per_sec": 1.0 / intervals["examples"],
            }
            if "tokens" in log_data:
                throughput["throughput/tokens_per_sec"] = 1.0 / intervals["tokens"]
            log_data.update(throughput)

        for key in summary_metrics:
            if key in log_data:
                if summary_metrics[key] is None:
                    summary_metrics[key] = 0
                summary_metrics[key] += log_data[key]
                log_data["running_average/" + key] = summary_metrics[key] / (it + 1)

        #### Add mode tag to current metrics ####
        log_data = {f"{mode}/{k}": v for k, v in log_data.items()}

        #### Add time metrics for all modes ####
        for m in train_state.time:
            log_data.update(train_state.time[m].loggable_dict(f"{m}/time/"))

        #### Record metrics that should NOT be tagged with the current mode ####
        log_data.update(total_time.loggable_dict("total/"))
        log_data["total/remaining_hr"] = total_time.hr  * (total_duration / train_state.time['train']-1.0)

        if logger is not None:
            logger(
                log_data,
            )

        pbar_inclusions = ["total/iteration", "train/iter_in_epoch"]
        pbar_desc = ", ".join(
            [f"{mode} ep: {train_state.time[mode].ep}"]
            + [f"loss: {loss:.2f}"]
            + [f"{k}: {v:.2f}" for k, v in log_data.items() if k in pbar_inclusions]
        )
        pbar.set_description(pbar_desc)

        if checkpoint_manager is not None:
            checkpoint_manager.maybe_save(train_state, train_state.time[mode])
        if mode_duration < train_state.time[mode] - mode_start:
            exhausted_loader = False
            break
    if exhausted_loader:
        train_state = time_updater(train_state.time[mode], train_state, ep=1)

    if logger is not None:
        for key in summary_metrics:
            if summary_metrics[key] is None:
                summary_metrics[key] = 0
            log_data[f"{mode}/full_average/{key}"] = summary_metrics[key] / (
                it + 1
            )
        logger(log_data, force=True)


    return train_state, exhausted_loader


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def train(config: DictConfig, data_loaders=None, model_state=None, optimizer_state=None) -> None:
    logging.info(OmegaConf.to_yaml(config))

    total_duration = duration.TrainDuration(config.train.total_duration)
    valid_freq = duration.TrainDuration(config.train.valid_frequency)
    valid_duration = duration.TrainDuration(config.train.valid_duration)

    checkpoint_manager = checkpoint.CheckpointManager(config)

    if data_loaders is None:
        data_loaders = get_loader(config)

    if config.train.wandb_project is not None:
        wandb.init(project=config.train.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
        limited_log = RateLimitedLog(wandb.log, config.train.wandb_logs_per_sec)
    else:
        limited_log = None

    if model_state is None:
        model, state = get_model(config, data_loaders)
        model_state = ModelAndState(model, state)

    if optimizer_state is None:
        optimizer, opt_state = get_optimizer(
            config.train, model_state.model, total_duration, data_loaders["train"], limited_log
        )
        optimizer_state  = OptimizerAndState(optimizer=optimizer, opt_state=opt_state)

    if config.train.use_amp:
        dynamic_scaler_state = DynamicScalerState(
            scaler=jnp.array(2**10, dtype=jnp.float32)
        )
    else:
        dynamic_scaler_state = None

    if config.train.averaging is not None and config.train.averaging != "none":
        aux = {'polyak': copy_arrays(model_state)}
        # ModelAndState(model=model, state=state))
    else:
        aux = {}

    train_time = duration.TrainTime(name="train")
    valid_time = duration.TrainTime(name="valid")
    train_state = TrainState(
        model=model_state,
        optimizer=optimizer_state,
        # optimizer=optimizer,
        dynamic_scaler_state=dynamic_scaler_state,
        time={"train": train_time, "valid": valid_time},
        aux=aux,
    )

    if config.train.get("load_path", False):
        train_state = checkpoint_manager.load(train_state, name=config.train.load_path)

    prng_key = jr.PRNGKey(0)

    time_keeper = TimeKeeper()

    loss_fn = get_loss(config)
    if config.train.use_amp:
        loss_fn = amp(loss_fn, compute_dtype=get_dtype(config.train.precision))

    train_pbar = tqdm.tqdm(enumerate(data_loaders["train"]))
    train_loader = iter(train_pbar)
    if config.train.valid_key is not None:
        valid_pbar = tqdm.tqdm(enumerate(data_loaders[config.train.valid_key]))
        valid_loader = iter(valid_pbar)


    train_start = duration.TrainTime()
    valid_start = duration.TrainTime()

    batch_size_fn = data_loaders["batch_size_fn"]
    while True:
        # train...
        to_use, prng_key = jax.random.split(prng_key)
        train_state = train_prepare(train_state, config)
        train_state, exhausted_train_loader = run_epoch(
            train_state,
            train_loader,
            train_pbar,
            loss_fn,
            mode="train",
            config=config,
            logger=limited_log,
            time_keeper=time_keeper,
            prng_key=to_use,
            mode_start=train_start,
            mode_duration=valid_freq,
            checkpoint_manager=checkpoint_manager,
            batch_size_fn=batch_size_fn,
            total_duration=total_duration,
        )
        train_state = train_break(train_state, config)

        print("\n")

        if train_state.time["train"] - train_start > valid_freq:
            train_start = train_state.time["train"].copy()
            valid_start = train_state.time["valid"].copy()
            # it doesn't make sense to have valid_duration > 1ep, so
            # we don't need to consider this case.
            if config.train.valid_key is not None:
                # valid_duration.reset(train_state.epoch, train_state.iteration)
                to_use, prng_key = jax.random.split(prng_key)
                train_state = valid_prepare(train_state, config)
                train_state, exhausted_valid_loader = run_epoch(
                    train_state,
                    valid_loader,
                    valid_pbar,
                    loss_fn,
                    mode="valid",
                    mode_duration=valid_duration,
                    checkpoint_manager=None,
                    config=config,
                    logger=limited_log,
                    time_keeper=time_keeper,
                    prng_key=to_use,
                    mode_start=valid_start,
                    batch_size_fn=batch_size_fn,
                    total_duration=total_duration,
                )
                train_state = valid_break(train_state, config)
                if exhausted_valid_loader:
                    valid_pbar = tqdm.tqdm(
                        enumerate(data_loaders[config.train.valid_key])
                    )
                    valid_loader = iter(valid_pbar)

        # tricky: we do this reset *after* the validation loop in order to
        # postpone incrementing the epoch count until we've logged all the
        # validation data.
        if exhausted_train_loader:
            train_pbar = tqdm.tqdm(enumerate(data_loaders["train"]))
            train_loader = iter(train_pbar)
        # make a newline so that the progress bar has a new line too...
        print("\n")
        if train_state.time["train"] > total_duration:
            break


if __name__ == "__main__":
    train()
