import optax
from jaxtyping import PyTree
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import numpy
from typing import NamedTuple, Any, Optional, List, Tuple
from optax import tree_utils as optu
import util
import otnc
import logstate


def tree_add(a, b):
    return jtu.tree_map(lambda a_i, b_i: a_i + b_i, a, b)


def tree_subtract(a, b):
    return jtu.tree_map(lambda a_i, b_i: a_i - b_i, a, b)


def tree_dot_per_layer(v, w):
    return jtu.tree_map(lambda vi, wi: jnp.sum(vi * wi), v, w)


def tree_dot(v, w):
    return jtu.tree_reduce(lambda x, y: x + y, tree_dot_per_layer(v, w))


def tree_norm(t):
    return jnp.sqrt(
        jtu.tree_reduce(lambda x, y: x + y, jtu.tree_map(lambda z: jnp.sum(z**2), t))
    )


def tree_scale(t, s):
    return jtu.tree_map(lambda x: x * s, t)


class MirrorDescentTunerState(NamedTuple):
    sum_squared_grad: optax.Updates
    initial_value: optax.Params
    max_grad: optax.Updates


def mirror_descent_tuner(
    lr: float = jnp.sqrt(1.0 / 4.0),
    beta: float = 1.0,
    epsilon: float = 1e-8,
    small_value: float = 1e-6,
    max_tuner_output: float = None,
):
    def init_fn(params: optax.Params):
        state = MirrorDescentTunerState(
            sum_squared_grad=jtu.tree_map(jnp.zeros_like, params),
            initial_value=jtu.tree_map(jnp.array, params),
            max_grad=jtu.tree_map(jnp.zeros_like, params),
        )
        return state

    def update_fn(
        updates: optax.Updates, state: MirrorDescentTunerState, params: optax.Params
    ):
        sum_squared_grad = state.sum_squared_grad
        initial_value = state.initial_value
        max_grad = state.max_grad

        clipped_updates = updates

        next_sum_squared_grad = jtu.tree_map(
            lambda sum_i, u_i: beta**2 * sum_i + u_i**2, sum_squared_grad, updates
        )
        next_max_grad = jtu.tree_map(
            lambda m_i, u_i: jnp.maximum(beta * m_i, jnp.abs(u_i)), max_grad, updates
        )

        def link_fn(theta, V, M, init_i):
            V = V + small_value
            M = M + small_value
            return init_i * jnp.exp(theta / jnp.sqrt(V))

        def inv_link_fn(p, V, M, init_i):
            V = V + small_value
            M = M + small_value
            return jnp.log(p / init_i) * jnp.sqrt(V)

        def get_next_param(p_i, init_i, u_i, old_sum_i, next_sum_i, m_i):
            old_theta = inv_link_fn(p_i, next_sum_i, m_i, init_i)
            theta = beta * old_theta - u_i  # - u_i**2/jnp.sqrt(next_sum_i)
            next_p_i = link_fn(theta, next_sum_i, m_i, init_i)
            next_p_i = jnp.clip(next_p_i, a_min=init_i, a_max=max_tuner_output)
            return next_p_i

        next_params = jtu.tree_map(
            get_next_param,
            params,
            initial_value,
            clipped_updates,
            sum_squared_grad,
            next_sum_squared_grad,  # clipped_updates, sum_squared_grad
            next_max_grad,
        )

        next_updates = tree_subtract(next_params, params)
        next_state = MirrorDescentTunerState(
            sum_squared_grad=next_sum_squared_grad,
            initial_value=initial_value,
            max_grad=next_max_grad,
        )

        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class OptimisticMirrorDescentTunerState(NamedTuple):
    sum_squared_grad: optax.Updates
    initial_value: optax.Params
    max_grad: optax.Updates
    prev_grad: optax.Updates


def optimistic_mirror_descent_tuner(
    lr: float = jnp.sqrt(1.0 / 4.0),
    beta: float = 1.0,
    epsilon: float = 1e-8,
    small_value: float = 1e-10,
    max_tuner_output: float = 1e-3,
):
    def init_fn(params: optax.Params):
        state = OptimisticMirrorDescentTunerState(
            sum_squared_grad=jtu.tree_map(jnp.zeros_like, params),
            initial_value=jtu.tree_map(jnp.array, params),
            max_grad=jtu.tree_map(jnp.zeros_like, params),
            prev_grad=jtu.tree_map(jnp.zeros_like, params),
        )
        return state

    def update_fn(
        updates: optax.Updates, state: MirrorDescentTunerState, params: optax.Params
    ):
        sum_squared_grad = state.sum_squared_grad
        initial_value = state.initial_value
        max_grad = state.max_grad
        prev_grad = state.prev_grad

        next_sum_squared_grad = jtu.tree_map(
            lambda sum_i, u_i, pg_i: beta**2 * sum_i + (u_i - pg_i) ** 2,
            sum_squared_grad,
            updates,
            prev_grad,
        )
        next_max_grad = jtu.tree_map(
            lambda m_i, u_i: jnp.maximum(beta * m_i, jnp.abs(u_i)), max_grad, updates
        )

        def link_fn(theta, V, M, init_i):
            V = V + small_value
            M = M + small_value
            return init_i * jnp.exp(theta / jnp.maximum(M, jnp.sqrt(V)))

        def inv_link_fn(p, V, M, init_i):
            V = V + small_value
            M = M + small_value
            return jnp.log(p / init_i) * jnp.maximum(M, jnp.sqrt(V))

        def get_next_param(p_i, init_i, u_i, old_sum_i, next_sum_i, m_i, prev_u_i):
            old_theta = inv_link_fn(p_i, next_sum_i, m_i, init_i)
            theta = beta * old_theta - 2 * u_i + prev_u_i
            next_p_i = link_fn(theta, next_sum_i, m_i, init_i)
            next_p_i = jnp.clip(next_p_i, a_min=init_i, a_max=max_tuner_output)
            return next_p_i

        next_params = jtu.tree_map(
            get_next_param,
            params,
            initial_value,
            updates,
            sum_squared_grad,
            next_sum_squared_grad,
            next_max_grad,
            prev_grad,
        )

        next_updates = tree_subtract(next_params, params)
        next_state = OptimisticMirrorDescentTunerState(
            sum_squared_grad=next_sum_squared_grad,
            initial_value=initial_value,
            max_grad=next_max_grad,
            prev_grad=updates,
        )

        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class OptaxTunerState(NamedTuple):
    reward: PyTree
    s_init: PyTree
    max_grad: PyTree
    sum_squared_grad: PyTree
    iter_count: PyTree


def optax_tuner(beta=1.0, eps=1e-8, num_iter=None, beta2=None):
    if beta2 is None:
        beta2 = beta**2

    def init_fn(s_init: PyTree):
        state = OptaxTunerState(
            reward=optu.tree_zeros_like(s_init),
            s_init=util.tree_copy(s_init),
            max_grad=optu.tree_zeros_like(s_init),
            sum_squared_grad=optu.tree_zeros_like(s_init),
            iter_count=0,
        )
        return state

    def update_fn(grads, state, s_values):
        clipped_grads = jtu.tree_map(
            lambda g_i, m_i: jax.lax.clamp(-m_i, g_i, m_i), grads, state.max_grad
        )
        next_max_grad = jtu.tree_map(
            lambda g_i, m_i: jnp.maximum(beta * m_i, jnp.abs(g_i) + eps),
            grads,
            state.max_grad,
        )

        next_iter_count = state.iter_count + 1
        if num_iter is None:
            next_sum_squared_grad = jtu.tree_map(
                lambda v_i, g_i: beta2 * v_i + g_i**2, state.sum_squared_grad, grads
            )
            debiased_next_sum_squared_grad = next_sum_squared_grad
        else:
            next_sum_squared_grad = jtu.tree_map(
                lambda v_i, g_i: beta2 * v_i + (1 - beta2) * g_i**2,
                state.sum_squared_grad,
                grads,
            )
            debiased_next_sum_squared_grad = jtu.tree_map(
                lambda v_i: v_i / (1.0 - beta2**next_iter_count),
                next_sum_squared_grad,
            )

        next_reward = jtu.tree_map(
            lambda r_i, s_i, g_i: beta * r_i - g_i * s_i,
            state.reward,
            s_values,
            clipped_grads,
        )

        # jax.debug.print(
        #     "reward: {r}, grads: {g}, s_value: {s} clipped_grads: {c}, max: {m}",
        #     r=next_reward,
        #     g=grads,
        #     s=s_values,
        #     c=clipped_grads,
        #     m=state.max_grad,
        # )

        wealth = jtu.tree_map(
            lambda s_init_i, m_i, r_i: s_init_i * m_i + jnp.clip(r_i, 0),
            state.s_init,
            next_max_grad,
            next_reward,
        )

        if num_iter is None:
            beta_scaling = 1.0
        elif num_iter == "anytime":
            beta_scaling = 1.0 / jnp.sqrt(next_iter_count)
        elif num_iter == "usebeta":
            beta_scaling = jnp.sqrt(1 - beta2)
        else:
            beta_scaling = 1.0 / jnp.sqrt(num_iter)

        next_s = jtu.tree_map(
            lambda w, v: w / (jnp.sqrt(v) + eps) * beta_scaling,
            wealth,
            debiased_next_sum_squared_grad,
        )

        next_state = OptaxTunerState(
            reward=next_reward,
            s_init=state.s_init,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            iter_count=next_iter_count,
        )

        updates = tree_subtract(next_s, s_values)

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class AdditionState(NamedTuple):
    substates: List[optax.OptState]
    subparams: List[optax.Params]


def add_optimizers(optimizers: Tuple[optax.GradientTransformation]):
    """wrapper for adding up a bunch of optimizer outputs"""

    def init_fn(params: optax.Params):
        substate = [opt.init(params) for opt in optimizers]
        subparams = [jtu.tree_map(jnp.array, params) for op in optimizers]
        return AdditionState(substate, subparams)

    def update_fn(
        updates: optax.Updates,
        state: AdditionState,
        params: Optional[optax.Params] = None,
    ):
        substates = state.substates
        subparams = state.subparams
        updates__state = [
            opt.update(updates, s, p)
            for opt, s, p in zip(optimizers, substates, subparams)
        ]

        # updates__state = util.map_over_lists(
        #     lambda o, s, p: o.update(updates, s, p),
        #     optimizers,
        #     substates,
        #     subparams
        # )
        # next_substates = util.map_over_lists(
        #     lambda x: x[1],
        #     updates__state
        # )
        # next_subparams = jtu.tree_map(
        #     lambda sp, u_s: optax.apply_updates(sp, u_s[0]),
        #     supbarams,
        #     updates__state
        # )
        # subupdates = [u__s[0] for u__s in updates__state]
        # updates = jtu.tree_map(
        #     lambda *u_i: jnp.sum(jnp.array(u_i), axis=0) / len(updates__state),
        #     *subupdates #*[u__s[0] for u__s in updates__state]
        # )

        # util.map_over_lists(
        #     lambda x: x[1],
        #     updates__state
        # )

        next_substates = [u__s[1] for u__s in updates__state]
        subupdates = [u__s[0] for u__s in updates__state]
        next_subparams = optax.apply_updates(subparams, subupdates)

        next_state = AdditionState(next_substates, next_subparams)

        # updates = jnp.sum(jnp.array(subupdates), axis=0)
        next_updates = jtu.tree_map(
            lambda *u_i: jnp.sum(jnp.array(u_i), axis=0) / len(updates__state),
            *subupdates  # *[u__s[0] for u__s in updates__state]
        )
        # jax.debug.print("new state: {}",new_state)
        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class MechanicState(NamedTuple):
    offset: optax.Updates  # this is the Delta in the paper.
    base_state: optax.OptState
    tuner_state: optax.OptState
    s: jax.Array
    key: jax.Array
    prev_random_scale: jax.Array
    incremental_variation: jax.Array
    incremental_sum: jax.Array
    iter_count: jax.Array
    update_count: jax.Array
    logging: logstate.Log


def mechanize_single_beta(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    optimistic: bool = False,
    weight_decay: float = 0.0,
    incremental: bool = False,
    per_layer: bool = False,
    randomize_incremental: bool = False,
    beta: float = 1.0,
    max_tuner_output: float = 1e-3,
    use_incremental_variation: bool = False,
    **kwargs
) -> optax.GradientTransformation:
    if optimistic:
        tuner = optimistic_mirror_descent_tuner(
            beta=beta, max_tuner_output=max_tuner_output
        )
    else:
        tuner = mirror_descent_tuner(beta=beta, max_tuner_output=max_tuner_output)
    return mechanize(
        base_optimizer,
        tuner_optimizer=tuner,
        s_init=s_init,
        weight_decay=weight_decay,
        incremental=incremental,
        per_layer=per_layer,
        randomize_incremental=randomize_incremental,
        use_incremental_variation=use_incremental_variation,
        **kwargs
    )


def summed_optax_tuner(
    betas=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], num_iter=None, betas2=None
):
    if betas2 is None:
        betas2 = [None for b in betas]
    tuners = [
        optax_tuner(beta=beta, num_iter=num_iter, beta2=beta2)
        for beta, beta2 in zip(betas, betas2)
    ]
    return add_optimizers(tuners)


def summed_mirror_descent(
    betas=[1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    optimistic=False,
    max_tuner_output=1e-3,
):
    """generate an optimizer by summing mirror descent optimizers for different beta values"""
    epsilons = [1.0 / (1e-8 + 1.0 - b) for b in betas]
    eps_sum = sum(epsilons)
    epsilons = [eps * len(epsilons) / eps_sum for eps in epsilons]
    epsilons = [1e-8 for b in betas]

    if optimistic:
        tuner_factory = optimistic_mirror_descent_tuner
    else:
        tuner_factory = mirror_descent_tuner

    md_tuners = [
        tuner_factory(beta=b, epsilon=eps, max_tuner_output=max_tuner_output)
        for b, eps in zip(betas, epsilons)
    ]
    return add_optimizers(md_tuners)


def optax_like_mechanize(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    betas: List[float] = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    incremental: bool = False,
    randomize_incremental: bool = False,
    use_incremental_variation: bool = False,
    betas2=None,
    num_iter=None,
    **kwargs
) -> optax.GradientTransformation:
    """
    re-implement the original mechanic in optax using this framework.
    """
    tuner = summed_optax_tuner(betas, num_iter, betas2=betas2)
    return mechanize(
        base_optimizer,
        tuner_optimizer=tuner,
        s_init=s_init / len(betas),
        weight_decay=weight_decay,
        incremental=incremental,
        randomize_incremental=randomize_incremental,
        use_incremental_variation=use_incremental_variation,
        **kwargs
    )


def mechanize(
    base_optimizer: optax.GradientTransformation,
    tuner_optimizer: optax.GradientTransformation = None,
    s_init: float = 1e-8,
    optimistic: bool = False,  # only used if tuner_optimizer is not sp
    betas=[1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    incremental: bool = False,  # if true, do an an update relative to the previous iterate rather than the starting  iterate.
    randomize_incremental: bool = False,  #  if true,  use random exponential scaling on an incremental update, otherwise don't scale (i.e. scale by 1.0)
    per_layer: bool = False,
    max_tuner_output: float = None,
    use_incremental_variation: bool = False,
    averaging_momentum: float = 0.0,
    freeze_s_iteration: Optional[int] = None,
    randomize_after_freeze: bool = False,
    tuner_decay_schedule="constant",
) -> optax.GradientTransformation:
    """
    Args:
    base_optimizer: the base optimizer to learn learning rate for.
    tuner_optimizer: the optimizer  to  use to learn the learning rate.
        if not specified, will default to "summed_mirror_descent", which
        is a tuner based on mirror descent rather than coin betting.
    s_init: initial learning rate used to initialize tuner_optimizer.
    optimistic: if tuner_optimizer is not specified, then controls whether the
        mirror descent tuner  is optimistic or not.
    betas: list of beta values for the tuner (only relevant if tuner_optimizer is not specified)
    weight_decay: same as lambda value in original mechanic (probably should have a better name)
    incremental: if true, do an an update relative to the previous iterate rather than the starting  iterate.
    randomize_incremental: if true,  use random exponential scaling on an incremental update, otherwise don't scale (i.e. scale by 1.0)
    per_layer: if true, learn a per-layer scale.
    max_tuner_output: provided to tuner to clip its outputs.
    averaging_momentum: funny kind of momentum to use with non-incremental mechanic.
        update_{t+1} = update_t + ((average of iterates x_t) - x_1)
        This is distinct from iterate averaging.
        So far, this never helps :)
    freeze_s_iteration: after this many iterations, stop updating s. If we are using randomized
        scaling in the incremental update, also stop applying the randomized scaling.
    randomize_after_freeze:  if true, apply randomization after freeze where  relevant.
    tuner_decay_schedule: schedule to apply to the tuner updates. Can be:
        'constant' (no schedule)
        'linearl' (linear decay)
    """

    if tuner_decay_schedule == "constant":
        tuner_decay_fn = lambda t, updates: updates
    elif tuner_decay_schedule == "linear":
        tuner_decay_fn = lambda t, updates: jtu.tree_map(
            lambda x: x * (freeze_s_iteration - t) / freeze_s_iteration, updates
        )
    else:
        raise ValueError("unknown tuner_decay_schedule")

    if tuner_optimizer is None:
        tuner_optimizer = summed_mirror_descent(
            betas, optimistic, max_tuner_output=max_tuner_output
        )

    def init_fn(params: optax.Params):
        offset = jtu.tree_map(jnp.zeros_like, params)
        base_state = base_optimizer.init(params)
        if not per_layer:
            s = jnp.array(s_init)
        else:
            s = jtu.tree_map(lambda p: jnp.array(s_init), params)
        tuner_state = tuner_optimizer.init(s)
        # print("initial s: ",jtu.tree_leaves(s))
        if per_layer:
            incremental_sum = jtu.tree_map(lambda x: 0.0, params)
        else:
            incremental_sum = 0.0
        return MechanicState(
            offset=offset,
            base_state=base_state,
            tuner_state=tuner_state,
            s=s,
            key=jax.random.PRNGKey(1231),
            prev_random_scale=0.0,
            incremental_variation=0.0,
            incremental_sum=incremental_sum,
            update_count=0,
            iter_count=0,
            logging=logstate.Log(
                {
                    "reward": 0.0,
                    "reward_std": 0.0,
                    "mechanic/max_s": 0.0,
                    "mechanic/min_s": 0.0,
                    "mechanic/tuner_update_count": 0,
                    "mechanic/incremental_variation": 0.0,
                    "mechanic/incremental_sum": 0.0,
                    "mechanic/offset_norm": 0.0,
                    "mechanic/scaled_offset_norm": 0.0,
                }
            ),
        )

    def update_fn(
        grads: optax.Updates,
        state: MechanicState,
        params: Optional[optax.Params] = None,
    ):
        offset = state.offset
        base_state = state.base_state
        tuner_state = state.tuner_state
        s = state.s
        next_key, to_use = jax.random.split(state.key)
        reward = state.logging.data["reward"]
        reward_std = state.logging.data["reward_std"]
        incremental_variation = state.incremental_variation
        incremental_sum = state.incremental_sum
        prev_random_scale = state.prev_random_scale
        update_count = state.update_count
        iter_count = state.iter_count

        if randomize_incremental:
            random_scale = jax.random.exponential(to_use)
        else:
            random_scale = 1.0

        base_updates, next_base_state = base_optimizer.update(grads, base_state, params)
        # base updates is "u" in the paper. Add this to Delta to get the next
        # value of Delta.
        if incremental:
            next_offset = base_updates
        else:
            next_offset = tree_add(offset, base_updates)

        if not per_layer:
            inner_product = tree_dot(
                offset,
                tree_add(
                    grads,
                    tree_scale(
                        params,
                        state.s
                        * weight_decay
                        * tree_norm(grads)
                        / (tree_norm(params) + 1e-8),
                    ),
                ),
            )
            reward_increment = inner_product * s
            next_incremental_variation = (
                incremental_variation * use_incremental_variation + inner_product**2
            )
            next_incremental_sum = (
                incremental_sum * use_incremental_variation + inner_product
            )

        else:
            inner_product = jtu.tree_map(
                lambda o, g, s, p: jnp.sum(
                    o
                    * (
                        g
                        + p
                        * s
                        * weight_decay
                        * jnp.linalg.norm(g)
                        / (jnp.linalg.norm(p) + 1e-8)
                    )
                ),
                offset,
                grads,
                state.s,
                params,
            )
            reward_increment = tree_dot(inner_product, s)

            next_incremental_variation = (
                incremental_variation + tree_norm(inner_product) ** 2
            )
            next_incremental_sum = tree_add(incremental_sum, inner_product)

        log_next_incremental_variation = next_incremental_variation
        log_next_incremental_sum = next_incremental_sum
        next_reward_std = jnp.sqrt(reward_std**2 + reward_increment**2)

        # jax.debug.print("grads: {g}, inner product: {i}",g=grads, i=inner_product)
        if use_incremental_variation:
            s_update, next_tuner_state = tuner_optimizer.update(
                next_incremental_sum, tuner_state, s
            )
        else:
            s_update, next_tuner_state = tuner_optimizer.update(
                inner_product, tuner_state, s
            )

        s_update = tuner_decay_fn(iter_count, s_update)

        next_s = tree_add(s, s_update)

        if freeze_s_iteration is not None:
            should_freeze_s = iter_count > freeze_s_iteration
            # if we have exceeded the freeze_s_iteration, then stop updating s
            next_s, next_tuner_state = jax.lax.cond(
                should_freeze_s,
                lambda: (s, tuner_state),
                lambda: (next_s, next_tuner_state),
            )

        if use_incremental_variation:
            baseline_incremental_sum = optu.tree_zeros_like(incremental_sum)
            (
                random_scale,
                next_update_count,
                next_incremental_variation,
                next_incremental_sum,
                next_s,
                next_tuner_state,
            ) = jax.lax.cond(
                jnp.sqrt(2 * next_incremental_variation)
                > tree_norm(next_incremental_sum),
                lambda: (
                    random_scale,
                    update_count,
                    next_incremental_variation,
                    next_incremental_sum,
                    s,
                    tuner_state,
                ),
                lambda: (
                    random_scale,
                    update_count + 1,
                    0.0,
                    baseline_incremental_sum,
                    next_s,
                    next_tuner_state,
                ),
            )
        else:
            next_incremental_variation = state.incremental_variation
            next_incremental_sum = state.incremental_sum
            next_update_count = update_count + 1

        max_s = jtu.tree_reduce(lambda a, b: jnp.maximum(a, b), next_s)
        min_s = jtu.tree_reduce(lambda a, b: jnp.minimum(a, b), next_s)

        def compute_update_global(base_i, next_offset_i, offset_i):
            if incremental:
                # update to  apply if we are still updating s
                standard_update = base_i * next_s * random_scale
                # update to  apply if we have stopped updating s
                frozen_update = base_i * next_s
                if randomize_after_freeze:
                    frozen_update = frozen_update * random_scale
            else:
                # update for non-incremental mechanic
                # non-incremental update is:
                # next_offset * next_s - offset * s
                # = (offset + base_update) * (s + s_update) - offset * s
                # = base_update * s + s_update * next_offset

                # if averaging_momentum is  included, then we also set the center to b
                # center = center + (param-center)*averaging_momentum
                # note that param - center = offset * s
                # so overall, we have
                # next_offset * s - offset * s *  (1-averaging_momentum)
                # = (offset + base_update) * (s+s_update)  - offset * s* ( 1- avmom)
                # = base_update * s + s_update * next_offset + offset *  s * avmom
                if averaging_momentum == "iter_count":
                    avmom = 1.0 / (iter_count + 1)
                else:
                    avmom = averaging_momentum

                # update to  apply if we are still updating s
                standard_update = (
                    base_i * s + next_offset_i * s_update + offset_i * s * avmom
                )

                # update to apply if we have stopped updating s
                frozen_update = base_i * s + offset_i * s * avmom

            if freeze_s_iteration is not None:
                return jax.lax.cond(
                    should_freeze_s, lambda: frozen_update, lambda: standard_update
                )
            else:
                return standard_update

        def compute_update_per_layer(
            base_i, next_offset_i, old_s_i, next_s_i, s_update_i
        ):
            if incremental:
                return base_i * next_s_i * random_scale
            else:
                return base_i * old_s_i + next_offset_i * s_update_i

        if per_layer:
            updates = jtu.tree_map(
                compute_update_per_layer, base_updates, next_offset, s, next_s, s_update
            )
        else:
            updates = jtu.tree_map(
                compute_update_global, base_updates, next_offset, offset
            )

        if per_layer:
            scaled_offset_norm = tree_norm(
                jtu.tree_map(lambda o_i, s_i: o_i * s_i, next_offset, next_s)
            )
        else:
            scaled_offset_norm = next_s * tree_norm(next_offset)
        next_state = MechanicState(
            offset=next_offset,
            base_state=next_base_state,
            tuner_state=next_tuner_state,
            s=next_s,
            key=next_key,
            prev_random_scale=random_scale,
            incremental_variation=next_incremental_variation,
            incremental_sum=next_incremental_sum,
            update_count=next_update_count,
            iter_count=iter_count + 1,
            logging=logstate.Log(
                {
                    "reward": reward + reward_increment,
                    "reward_std": next_reward_std,
                    "mechanic/max_s": max_s,
                    "mechanic/min_s": min_s,
                    "mechanic/tuner_update_count": next_update_count,
                    "mechanic/incremental_variation": log_next_incremental_variation,
                    "mechanic/incremental_sum": tree_norm(log_next_incremental_sum),
                    "mechanic/offset_norm": tree_norm(next_offset),
                    "mechanic/scaled_offset_norm": scaled_offset_norm,
                }
            ),
        )

        return updates, next_state

    optimizer = optax.GradientTransformation(init_fn, update_fn)

    return optimizer
