import jax
from jax import numpy as jnp
import equinox as eqx
from jax import random as jr
from jax.random import PRNGKey, PRNGKeyArray
from jaxamp import amp, DynamicScalerState, dynamic_scale_value_and_grad
from jaxtyping import Array, PyTree
from typing import Tuple, Any, Optional, Sequence, Union, NamedTuple, Callable
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
from duration import Duration, set_timestamp

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


def apply_model(model: ModelAndState, *args, **kwargs):
    module = model.module
    state = model.state
    state = model.module(*args, state=state, **kwargs)
    return ModelAndState(module=module, state=state)


class TrainState(NamedTuple):
    model: ModelAndState
    opt_state: Any
    optimizer: optax.GradientTransformation
    dynamic_scaler_state: Optional[DynamicScalerState]
    iteration: Array
    epoch: Array
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

    # if config.use_amp:
    #     loss_fn = amp(loss_fn, compute_dtype=get_dtype(config.precision))

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
    opt_state = train_state.opt_state
    dynamic_scaler_state = train_state.dynamic_scaler_state
    optimizer = train_state.optimizer
    aux = train_state.aux
    epoch = train_state.epoch
    iteration = train_state.iteration

    if config.use_amp:
        # amp_loss_fn = amp(loss_fn, compute_dtype=get_dtype(config.precision))
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

    iteration = iteration + 1
    epoch = epoch
    if config.averaging == "polyak":
        aux = update_average(model, state, aux, iteration)

    new_train_state = TrainState(
        model=ModelAndState(model=model, state=state),
        opt_state=opt_state,
        dynamic_scaler_state=dynamic_scaler_state,
        optimizer=optimizer,
        iteration=iteration,
        epoch=epoch,
        aux=aux,
    )

    if config.log_norms:
        log_data["norms/grads"] = tree_norm(grads)
        log_data["norms/params"] = tree_norm(model)

    logged_state = logstate.LoggedState(new_train_state, log_data)
    all_logs = util.merge_dicts(*logstate.list_of_logs(logged_state))
    return loss, logged_state, all_logs


class RateLimitedWandbLog:
    def __init__(self, max_frequency=1.0):
        self.max_frequency = 1.0
        self.last_time = time.time() - 1.0 / self.max_frequency
        self.metrics = {}

    def __call__(self, metrics, *args, commit=True, force=False, **kwargs):
        self.metrics.update(metrics)
        if commit:
            self.commit(force=force, *args, **kwargs)

    def commit(self, force=False, *args, **kwargs):
        if len(self.metrics) != 0:
            cur_time = time.time()
            if force or cur_time >= self.last_time + 1.0 / self.max_frequency:
                wandb.log(self.metrics, *args, **kwargs)
                self.last_time = cur_time
                self.metrics = {}


def train_prepare(train_state, config):
    # return train_state
    return eqx.tree_at(
        lambda t: t.model.model,
        train_state,
        eqx.nn.inference_mode(train_state.model.model, value=False),
    )


def train_break(train_state, config):
    return train_state


def valid_prepare(train_state, config):
    # return train_state
    train_state = eqx.tree_at(
        lambda t: t.model.model,
        train_state,
        eqx.nn.inference_mode(train_state.model.model, value=True),
    )
    aux = train_state.aux
    model = train_state.model
    if config.train.averaging == "polyak":
        print("switching!")
        train_state = eqx.tree_at(
            lambda t: (t.model, t.aux), train_state, replace=(aux, model)
        )
    return train_state


def valid_break(train_state, config):
    aux = train_state.aux
    model = train_state.model
    if config.train.averaging == "polyak":
        print("switching back!")
        train_state = eqx.tree_at(
            lambda t: (t.model, t.aux), train_state, replace=(aux, model)
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
    stop_time: Duration,
    config: DictConfig,
    mode: str,
    time_keeper: TimeKeeper,
    logger: Callable,
    prng_key: Array,
):
    mode = mode.lower()

    total_tokens = 0

    if mode == "train":
        step_fn_jit = eqx.filter_jit(
            jtu.Partial(train_step, loss_fn=loss_fn, config=config.train),
            donate="all",
        )
        train_state = train_prepare(train_state, config)
        break_fn = train_break
    else:
        step_fn_jit = eqx.filter_jit(
            jtu.Partial(inference_step, loss_fn=loss_fn, config=config.train),
        )
        train_state = valid_prepare(train_state, config)
        break_fn = valid_break

    exhausted_loader = True
    iteration_timing_events = ["iteration", "dataloader", "train_step"]
    time_keeper.mark(start_events=["dataloader", "iteration", "tokens", "samples"])
    start_iter = np.array(train_state.iteration)
    it = -1
    summary_metrics = {"loss": None, "accuracy": None}
    while True:
        try:
            idt, batch = next(dataloader)
        except StopIteration:
            break
        it += 1
        batch = util.pytorch_to_np(batch)

        # bit of a hack:  some data type (like lists of numpy arrays)
        # will get made here, so we  just skip it.
        # tricky: we need to count the tokens BEFORE sending  to the step_fn
        # because we are going to donate all the buffers.
        try:
            if "attention_mask" in batch:
                tokens = util.count_tokens(batch["attention_mask"])
            else:
                tokens = None
        except ValueError:
            tokens = None
        time_keeper.mark(end_events={"dataloader": 1}, start_events=["train_step"])
        to_use, prng_key = jr.split(prng_key)
        train_state = set_timestamp(train_state)
        loss, train_state, log_data = step_fn_jit(train_state, batch, prng_key=to_use)
        # print("log_list: ",log_list)

        time_keeper.mark(
            end_events={"train_step": 1},
        )

        #### Record metrics that should be tagged with the current mode ####

        if tokens is not None:
            log_data["tokens"] = tokens
        if "tokens" in log_data:
            total_tokens += log_data["tokens"]
            log_data["total_tokens"] = total_tokens

        log_data["iter_in_epoch"] = it
        log_data["loss"] = loss

        pbar_desc = ", ".join(
            [f"{mode} ep: {train_state.epoch}"]
            + [f"{k}: {v:.2f}" for k, v in log_data.items()]
            + [f"total_it: {train_state.iteration}"]
        )
        pbar.set_description(pbar_desc)

        tokens = log_data.get("tokens", 0)
        samples = log_data["samples"]
        time_keeper.mark(
            start_events=["dataloader", "iteration", "tokens", "samples"],
            end_events={"iteration": 1, "tokens": tokens, "samples": samples},
        )
        durations = time_keeper.get_durations()
        proportions = time_keeper.get_proportions()
        log_data.update(
            {
                f"time/secs_per/{k}": durations[k]
                for k in iteration_timing_events
                if k in durations
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

        if "iteration" in durations:
            throughput = {
                "throughput/iteration_per_sec": 1.0 / durations["iteration"],
                "throughput/samples_per_sec": 1.0 / durations["samples"],
            }
            if "tokens" in log_data:
                throughput["throughput/tokens_per_sec"] = 1.0 / durations["tokens"]
            log_data.update(throughput)

        for key in summary_metrics:
            if key in log_data:
                if summary_metrics[key] is None:
                    summary_metrics[key] = 0
                summary_metrics[key] += log_data[key]
                log_data["running_average/" + key] = summary_metrics[key] / (it + 1)

        #### Add mode tag to current metrics ####
        log_data = {f"{mode}/{k}": v for k, v in log_data.items()}

        #### Record metrics that should NOT be tagged with the current mode ####
        log_data["epoch"] = np.array(train_state.epoch)
        log_data["total_iter"] = np.array(train_state.iteration)

        if config.train.wandb_project is not None:
            logger(
                log_data,
            )

        if stop_time.elapsed(np.array(train_state.epoch), it + start_iter + 1):
            exhausted_loader = False
            break

    if config.train.wandb_project is not None:
        if exhausted_loader:
            for key in summary_metrics:
                if summary_metrics[key] is None:
                    summary_metrics[key] = 0
                log_data[f"{mode}/epoch_average/{key}"] = summary_metrics[key] / (it + 1)
        logger.commit(force=True)

    train_state = break_fn(train_state, config)

    return train_state, prng_key, exhausted_loader


@hydra.main(version_base=None, config_path="conf", config_name="config_gpt2")
def train(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))

    total_duration = Duration(config.train.total_duration)
    valid_freq = Duration(config.train.total_duration, config.train.valid_frequency)
    valid_duration = Duration(config.train.total_duration, config.train.valid_duration)

    data_loaders = get_loader(config)

    if config.train.wandb_project is not None:
        limited_log = RateLimitedWandbLog(config.train.wandb_logs_per_sec)
        wandb.init(project=config.train.wandb_project)
        wandb.config.update(OmegaConf.to_container(config))
    else:
        limited_log = None

    model, state = get_model(config, data_loaders)

    optimizer, opt_state = get_optimizer(
        config.train, model, total_duration, data_loaders["train"], limited_log
    )

    if config.train.use_amp:
        dynamic_scaler_state = DynamicScalerState(
            scaler=jnp.array(2**10, dtype=jnp.float32)
        )
    else:
        dynamic_scaler_state = None

    if config.train.averaging is not None and config.train.averaging != "none":
        aux = copy_arrays(ModelAndState(model=model, state=state))
    else:
        aux = None

    train_state = TrainState(
        model=ModelAndState(model=model, state=state),
        opt_state=opt_state,
        optimizer=optimizer,
        dynamic_scaler_state=dynamic_scaler_state,
        iteration=jnp.array(0),
        epoch=jnp.array(0),
        aux=aux,
    )

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

    total_duration.reset()
    valid_duration.reset()
    valid_freq.reset()

    while True:
        # train...
        train_state, prng_key, exhausted_train_loader = run_epoch(
            train_state,
            train_loader,
            train_pbar,
            loss_fn,
            mode="train",
            config=config,
            stop_time=valid_freq,
            logger=limited_log,
            time_keeper=time_keeper,
            prng_key=prng_key,
        )

        next_epoch_number = train_state.epoch + exhausted_train_loader
        if not exhausted_train_loader:
            print("\n")

        if valid_freq.elapsed_and_reset(next_epoch_number, train_state.iteration):
            if config.train.valid_key is not None:
                valid_duration.reset(train_state.epoch, train_state.iteration)
                train_state, prng_key, exhausted_valid_loader = run_epoch(
                    train_state,
                    valid_loader,
                    valid_pbar,
                    loss_fn,
                    mode=config.train.valid_key,
                    stop_time=valid_duration,
                    config=config,
                    logger=limited_log,
                    time_keeper=time_keeper,
                    prng_key=prng_key,
                )
                if exhausted_valid_loader:
                    valid_pbar = tqdm.tqdm(
                        enumerate(data_loaders[config.train.valid_key])
                    )
                    valid_loader = iter(valid_pbar)

        # tricky: we do this reset *after* the validation loop in order to
        # postpone incrementing the epoch count until we've logged all the
        # validation data.
        if exhausted_train_loader:
            train_state = eqx.tree_at(
                lambda t: t.epoch, train_state, train_state.epoch + 1
            )
            train_pbar = tqdm.tqdm(enumerate(data_loaders["train"]))
            train_loader = iter(train_pbar)
        # make a newline so that the progress bar has a new line too...
        print("\n")
        if total_duration.elapsed(train_state.epoch, train_state.iteration):
            break


if __name__ == "__main__":
    train()
