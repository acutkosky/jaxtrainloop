import equinox as eqx
import jax
import optax
from jax import tree_util as jtu
from jax import numpy as jnp
from jaxtyping import PyTree
from omegaconf import DictConfig
from typing import Optional, Any, Callable
from util import log_optax, key_tree, zeros_like
from typing import NamedTuple
import time
import mechanic as new_mechanic
import duration
import logstate
import util
import optadam
import optadam_harsh
import otnc
import cocob
from precondition_opt import vector_preconditioned_momentum
import exponential_balancer
import simplified_mechanic


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
        return (opt.init(params), 0, logstate.Log({"optimizer/skipped_steps": 0}))

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: Optional[optax.Params] = None,
    ):
        inner_state, skip_count, logs = state
        next_updates, next_inner_state = opt.update(updates, inner_state, params)
        next_updates, next_inner_state, next_skip_count = jax.lax.cond(
            all_finite((next_updates, next_inner_state)),
            lambda: (next_updates, next_inner_state, skip_count),
            lambda: (zeros_like(updates), inner_state, skip_count + 1),
        )

        return next_updates, (
            next_inner_state,
            next_skip_count,
            logstate.Log({"optimizer/skipped_steps": next_skip_count}),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def zero_if_nan(updates, params):
    return jax.lax.cond(all_finite(updates), lambda x: x, zeros_like, updates)


class AnytimeAvgState(NamedTuple):
    iteration: jax.Array
    base_optimizer_state: optax.OptState
    base_iterate: PyTree
    total_weight: jax.Array
    prev_weight: jax.Array
    weight_state: PyTree
    av_iterate: jax.Array
    log_data: logstate.Log


def uniform_weight_fn(grads, state, params, weight_state):
    return 1.0, None


def inv_grad_sq_weight_fn(grads, state, params, weight_state):
    return 1.0 / (1e-6 + util.tree_norm(grads) ** 2), None


def exp_av_inv_grad_sq_weight_init(init, beta):
    return (init, beta)


def step_count_weight_fn(grads, state, params, weight_state):
    return weight_state + 1, weight_state + 1


def exp_av_inv_grad_sq_weight_fn(grads, state, params, weight_state):
    weight, beta = weight_state
    inv_weight = 1.0 / weight
    inv_weight = inv_weight * beta + (1 - beta) * util.tree_norm(grads) ** 2
    weight = 1.0 / inv_weight
    return weight, (weight, beta)


def exp_av_inv_grad_l1_weight_fn(grads, state, params, weight_state):
    weight, beta = weight_state
    inv_weight = 1.0 / weight
    inv_weight = inv_weight * beta + (1 - beta) * util.tree_norm(grads, ord=1)
    weight = 1.0 / inv_weight
    return weight, (weight, beta)


def anytime_avg(base_optimizer, weight="uniform", averaging_momentum=False):
    if eqx.is_array_like(weight):
        weight_fn = lambda grads, state, params: weight, None
        weight_state = None
    elif weight == "step_count":
        weight_fn = step_count_weight_fn
        weight_state = 0
    elif weight == "uniform":
        weight_fn = uniform_weight_fn
        weight_state = None
    elif weight == "inv_grad_sq":
        weight_fn = inv_grad_sq_weight_fn
        weight_state = None
    elif weight == "exp_av_inv_grad_sq":
        weight_fn = exp_av_inv_grad_sq_weight_fn
        weight_state = (1e-8, 0.999)
    elif weight == "exp_av_inv_grad_l1":
        weight_fn = exp_av_inv_grad_l1_weight_fn
        weight_state = (1e-8, 0.999)
    else:
        weight_fn = weight[0]
        weight_state = weight[1]

    def init_fn(params):
        state = AnytimeAvgState(
            iteration=jnp.array(0),
            base_optimizer_state=base_optimizer.init(params),
            base_iterate=util.tree_copy(params),
            av_iterate=util.tree_copy(params),
            total_weight=1e-8,
            prev_weight=1e-8,
            weight_state=weight_state,
            log_data=logstate.Log(
                {"averaging/total_weight": 0.0, "averaging/weight": 0.0}
            ),
        )
        return state

    def update_fn(grads, state, params):
        iteration = state.iteration + 1
        base_optimizer_state = state.base_optimizer_state
        base_iterate = state.base_iterate
        total_weight = state.total_weight
        prev_weight = state.prev_weight
        weight_state = state.weight_state
        av_iterate = state.av_iterate

        weight, weight_state = weight_fn(grads, state, params, weight_state)
        # beta = 0.999#jnp.sqrt(0.999)
        beta = 1.0
        total_weight = beta * total_weight + weight

        scaled_grads = util.tree_scalar_mul(grads, prev_weight)

        base_update, base_optimizer_state = base_optimizer.update(
            scaled_grads, base_optimizer_state, params
        )

        base_iterate = optax.apply_updates(base_iterate, base_update)

        av_iterate_update = util.tree_scalar_mul(
            util.tree_subtract(params, av_iterate), weight / total_weight
        )

        av_iterate = util.tree_add(av_iterate_update, av_iterate)

        # base_iterate = optax.apply_updates(base_iterate, av_iterate)
        if averaging_momentum:
            adjusted_base_iterate = optax.apply_updates(base_iterate, av_iterate)
        else:
            adjusted_base_iterate = base_iterate

        updates = util.tree_scalar_mul(
            util.tree_subtract(adjusted_base_iterate, params), weight / total_weight
        )

        print("in schedule!")
        state = AnytimeAvgState(
            iteration=iteration,
            base_optimizer_state=base_optimizer_state,
            base_iterate=base_iterate,
            total_weight=total_weight,
            prev_weight=weight,
            av_iterate=av_iterate,
            weight_state=weight_state,
            log_data=logstate.Log(
                {"averaging/total_weight": total_weight, "averaging/weight": weight}
            ),
        )

        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def flip_sign():
    def init_fn(params):
        return None

    def update_fn(updates, state, params):
        return jtu.tree_map(lambda x: -x, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


class RunningStdDevState(NamedTuple):
    param_mean: PyTree
    variance: float
    log_data: logstate.Log


def record_running_std_dev(beta):
    def init_fn(params):
        return RunningStdDevState(
            param_mean=util.tree_copy(params),
            variance=0.0,
            log_data=logstate.Log({f"param_variance_@{beta}": 0.0}),
        )

    def update_fn(updates, state, params):
        next_param_mean = jtu.tree_map(
            lambda m_i, p_i: beta * m_i + (1 - beta) * p_i, state.param_mean, params
        )

        next_variance = beta * state.variance + util.tree_sq_norm(
            util.tree_subtract(params, state.param_mean)
        )

        next_state = RunningStdDevState(
            param_mean=next_param_mean,
            variance=next_variance,
            log_data=logstate.Log({f"param_variance_@{beta}": jnp.sqrt(next_variance)}),
        )

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class ScheduleState(NamedTuple):
    count: jax.Array
    train_time: duration.TrainTime
    base_state: optax.OptState
    grad_stats: jax.Array
    rng_key: jax.Array
    log_data: logstate.Log
    prev_update: jax.Array


def scale_by_schedule_logged(
    schedule_fn: Callable,
    base_optimizer: optax.GradientTransformation,
    config: DictConfig,
):
    def init_fn(params):
        count = jnp.array(0)
        train_time = duration.TrainTime()
        base_state = base_optimizer.init(params)
        if config.log_inner_product:
            prev_update = util.zeros_like(params)
        else:
            prev_update = None

        state = ScheduleState(
            count=count,
            train_time=train_time,
            base_state=base_state,
            grad_stats=1.0,
            rng_key=jax.random.PRNGKey(124323),
            log_data=logstate.Log(
                {"lr/schedule": jnp.array(0.0), "lr/innerprod": jnp.array(1.0)}
            ),
            prev_update=prev_update,
        )
        return state

        # logstate.LoggedState((jnp.array(0), duration.JaxTimeStamp()), log_data={"lr/schedule": jnp.array(0.0)})

    def update_fn(grads, state, params):
        count = state.count
        base_state = state.base_state
        train_time = state.train_time
        grad_stats = state.grad_stats
        prev_update = state.prev_update

        count = count + 1

        grads = util.tree_scalar_mul(grads, 1.0 / grad_stats)

        updates, base_state = base_optimizer.update(grads, base_state, params)

        schedule = schedule_fn(count, train_time)

        rng_key, to_use = jax.random.split(state.rng_key)

        if config.log_inner_product:
            inner_prod = util.tree_dot(prev_update, grads)
            next_prev_update = updates
        else:
            inner_prod = None
            next_prev_update = None
        if config.grad_schedule:
            grad_stats = config.grad_stats_beta * grad_stats + (
                1.0 - config.grad_stats_beta
            ) * jnp.abs(util.tree_dot(updates, grads))
            # schedule = schedule / (grad_stats / (1.0-config.grad_stats_beta**count))
            # schedule = schedule / jnp.abs(util.tree_dot(updates, grads))
        if config.randomize_schedule:
            schedule = jax.random.exponential(to_use) * schedule

        updates = jtu.tree_map(lambda x: schedule * x, updates)
        log_data = {"lr/schedule": schedule, "lr/innerprod": inner_prod}

        new_state = ScheduleState(
            count=count,
            train_time=train_time,
            base_state=base_state,
            grad_stats=grad_stats,
            rng_key=rng_key,
            log_data=logstate.Log(log_data),
            prev_update=next_prev_update,
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def schedule_fn(
    count: int,
    train_time: duration.TrainTime,
    config: DictConfig,
    train_duration: duration.TrainDuration,
    loader: Any,
    peak: float,
    # logger: Optional[Callable] = None,
):
    train_elapsed = train_time / train_duration
    # if train_duration.minutes != float("inf"):
    #     train_elapsed = jnp.asarray(
    #         (timestamp.timestamp / 60 - train_duration.start_time)
    #         / train_duration.minutes
    #     )
    # else:
    #     if train_duration.iterations != float("inf"):
    #         max_iter = train_duration.iterations
    #     if train_duration.epochs != float("inf"):
    #         max_iter = len(loader) * train_duration.epochs
    #     train_elapsed = count / max_iter

    warmup = config.lr_warmup

    fraction_remaining = (1 - train_elapsed) / (1 - warmup)
    if config.lr_decay == "linear":
        decay_value = fraction_remaining
    elif config.lr_decay == "cosine":
        decay_value = jnp.cos(fraction_remaining * jnp.pi) * 0.5 + 0.5
    elif config.lr_decay == "sqrt":
        decay_value = jnp.sqrt(fraction_remaining)
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
    train_duration: duration.TrainDuration,
    train_loader,
    logger: Optional[Callable] = None,
):
    if not config.log_callback_data:
        logger = None
    opt_config = config.optim
    schedule = jtu.Partial(
        schedule_fn,
        loader=train_loader,
        train_duration=train_duration,
        config=opt_config,
        peak=opt_config.lr,
    )

    # set the learning rate to 1.0 here - we will scale
    # by the learning rate schedule later.
    if opt_config.name == "sgd":
        optimizer = optax.chain(
            optax.add_decayed_weights(opt_config.weight_decay),
            optax.sgd(learning_rate=1.0, momentum=opt_config.momentum),
        )
    elif opt_config.name == "adamw":
        optimizer = optax.adamw(
            learning_rate=1.0,
            b1=opt_config.beta1,
            b2=opt_config.beta2,
            weight_decay=opt_config.weight_decay,
        )
    elif opt_config.name == "opt_adam":
        optimizer = optadam.opt_adam(
            lr=1.0,
            beta1=opt_config.beta1,
            beta2=opt_config.beta2,
            weight_decay=opt_config.weight_decay,
            use_max=opt_config.use_max,
        )
    elif opt_config.name == "cocob":
        optimizer = cocob.cocob()
    elif opt_config.name == "opt_adam_harsh":
        optimizer = optax.chain(
            optadam_harsh.scale_by_opt_laprop(opt_config.beta1, opt_config.beta2),
            optax.add_decayed_weights(weight_decay=opt_config.weight_decay),
        )
    elif opt_config.name == "otnc":
        optimizer = otnc.otnc(opt_config, jax.random.PRNGKey(12345))
    elif opt_config.name == "vector_preconditioned":
        optimizer = vector_preconditioned_momentum(
            lr=1.0,
            beta1=opt_config.beta1,
            beta2=opt_config.beta2,
            weight_decay=opt_config.weight_decay,
        )

    if opt_config.bake_schedule:
        if config.averaging == "anytime":
            optimizer = anytime_avg(
                optimizer, config.averaging_weight, config.averaging_momentum
            )
        optimizer = scale_by_schedule_logged(schedule, optimizer, opt_config)
        # optimizer = optax.chain(optimizer, scale_by_schedule_logged(schedule))

    # we do gradient clipping before wrapping with mechanize, so mechanic sees the raw gradients
    if opt_config.gradient_clip_val is not None:
        grad_clip = optax.clip_by_global_norm(opt_config.gradient_clip_val)
        optimizer = optax.chain(grad_clip, optimizer)

    if opt_config.do_exp_balancing:
        scalar = exponential_balancer.exponential_balancer(
            beta1=opt_config.exp_balancing.beta1,
            beta2=opt_config.exp_balancing.beta2,
            s_init=opt_config.exp_balancing.s_init,
            granularity=opt_config.exp_balancing.granularity,
            exp_type=opt_config.exp_balancing.exp_type,
            exp_scaling=opt_config.exp_balancing.exp_scaling,
        )
        optimizer = optax.chain(optimizer, scalar)

    if opt_config.mechanize and opt_config.mechanize != "none":
        if opt_config.mechanize == "optax":
            optimizer = optax.contrib.mechanize(
                optimizer, weight_decay=opt_config.mechanic.weight_decay
            )
        elif opt_config.mechanize == "optax_redux":
            optimizer = new_mechanic.optax_like_mechanize(
                optimizer,
                weight_decay=opt_config.mechanic.weight_decay,
                incremental=opt_config.mechanic.incremental,
                randomize_incremental=opt_config.mechanic.randomize_incremental,
                use_incremental_variation=opt_config.mechanic.use_incremental_variation,
                averaging_momentum=opt_config.mechanic.averaging_momentum,
                freeze_s_iteration=opt_config.mechanic.freeze_s_iteration,
                randomize_after_freeze=opt_config.mechanic.randomize_after_freeze,
                betas=opt_config.mechanic.optax.betas,
                betas2=opt_config.mechanic.optax.betas2,
                num_iter=opt_config.mechanic.optax.num_iter,
                per_layer=opt_config.mechanic.per_layer,
                tuner_decay_schedule=opt_config.mechanic.tuner_decay_schedule,
            )
        elif opt_config.mechanize == "simplified_mechanic":
            optimizer = simplified_mechanic.optax_like_mechanize(
                optimizer,
                weight_decay=opt_config.mechanic.weight_decay,
                averaging_momentum=opt_config.mechanic.averaging_momentum,
                freeze_s_iteration=opt_config.mechanic.freeze_s_iteration,
                betas=opt_config.mechanic.optax.betas,
                betas2=opt_config.mechanic.optax.betas2,
                num_iter=opt_config.mechanic.optax.num_iter,
                bet_fraction_type=opt_config.mechanic.optax.bet_fraction_type,
                per_layer=opt_config.mechanic.per_layer,
                tuner_decay_schedule=opt_config.mechanic.tuner_decay_schedule,
                tuner_lr=opt_config.mechanic.tuner_lr,
            )
        elif opt_config.mechanize == "simplified_mirror_descent_mechanic":
            optimizer = simplified_mechanic.mirror_descent_mechanize(
                optimizer,
                weight_decay=opt_config.mechanic.weight_decay,
                averaging_momentum=opt_config.mechanic.averaging_momentum,
                freeze_s_iteration=opt_config.mechanic.freeze_s_iteration,
                betas=opt_config.mechanic.optax.betas,
                betas2=opt_config.mechanic.optax.betas2,
                bet_fraction_type=opt_config.mechanic.optax.bet_fraction_type,
                per_layer=opt_config.mechanic.per_layer,
                tuner_decay_schedule=opt_config.mechanic.tuner_decay_schedule,
                tuner_lr=opt_config.mechanic.tuner_lr,
            )
        elif opt_config.mechanize == "new":
            betas = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
            if opt_config.mechanic.use_one_beta:
                betas = [1.0] + betas
            optimizer = new_mechanic.mechanize(
                optimizer,
                betas=betas,
                optimistic=opt_config.mechanic.optimistic,
                weight_decay=opt_config.mechanic.weight_decay,
                incremental=opt_config.mechanic.incremental,
                randomize_incremental=opt_config.mechanic.randomize_incremental,
                per_layer=opt_config.mechanic.per_layer,
                max_tuner_output=opt_config.mechanic.max_tuner_output,
                use_incremental_variation=opt_config.mechanic.use_incremental_variation,
                freeze_s_iteration=opt_config.mechanic.freeze_s_iteration,
                randomize_after_freeze=opt_config.mechanic.randomize_after_freeze,
                tuner_decay_schedule=opt_config.mechanic.tuner_decay_schedule,
            )
        elif opt_config.mechanize == "singlebeta":
            optimizer = new_mechanic.mechanize_single_beta(
                optimizer,
                optimistic=opt_config.mechanic.optimistic,
                weight_decay=opt_config.mechanic.weight_decay,
                incremental=opt_config.mechanic.incremental,
                randomize_incremental=opt_config.mechanic.randomize_incremental,
                per_layer=opt_config.mechanic.per_layer,
                beta=opt_config.mechanic.single_beta_val,
                max_tuner_output=opt_config.mechanic.max_tuner_output,
                use_incremental_variation=opt_config.mechanic.use_incremental_variation,
                freeze_s_iteration=opt_config.mechanic.freeze_s_iteration,
                randomize_after_freeze=opt_config.mechanic.randomize_after_freeze,
                tuner_decay_schedule=opt_config.mechanic.tuner_decay_schedule,
            )
        else:
            raise ValueError(f"unknown mechanize option: {opt_config.mechanize}")
        if logger is not None:

            def log_fn(updates, state, params):
                return {"mechanic/s": util.tree_reduce_mean(state.s)}

            optimizer = log_optax(optimizer, log_fn)

    if not opt_config.bake_schedule:
        if config.averaging == "anytime":
            optimizer = anytime_avg(
                optimizer, config.averaging_weight, config.averaging_momentum
            )
        # optimizer = optax.chain(optimizer,
        optimizer = scale_by_schedule_logged(schedule, optimizer, opt_config)
        # if not opt_config.mechanize:
        #     optimizer = optax.chain(optimizer, optax.scale(opt_config.lr))

    if config.get("gradient_noise", 0) != 0:
        optimizer = optax.chain(
            add_noise(config.gradient_noise, jax.random.PRNGKey(1231)), optimizer
        )

    if opt_config.accumulation_steps != 1:
        optimizer = optax.MultiSteps(
            optimizer,
            every_k_schedule=opt_config.accumulation_steps,
            use_grad_mean=True,
            should_skip_update_fn=optax.skip_not_finite,
        )

    if opt_config.get("random_scale_type", "none") != "none":
        key = jax.random.PRNGKey(88)
        optimizer = otnc.random_scale(opt_config.random_scale_type, key, optimizer)

    # optimizer = optax.apply_if_finite(optimizer, 15)
    optimizer = skip_nonfinite(optimizer)
    # optimizer = optax.chain(optimizer, optax.stateless(zero_if_nan))

    opt_state = optimizer.init(util.tree_copy(eqx.filter(model, eqx.is_array)))
    return optimizer, opt_state
