import optax
from jaxtyping import PyTree
import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import numpy
from typing import NamedTuple, Any, Optional, List, Tuple, Union
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


class OptaxTunerState(NamedTuple):
    reward: PyTree
    s_init: PyTree
    max_grad: PyTree
    sum_grad: PyTree
    sum_squared_grad: PyTree
    s_values: PyTree
    iter_count: PyTree


def optax_tuner(
    betas=[1.0],
    eps=1e-8,
    num_iter=Union[None, int, str],
    betas2=None,
    bet_fraction_type="sqrt",
):
    """
    implements the tuner as in the original mechanic implementation in optax, with a
    few new options.

    betas: list of beta values for decaying the reward.
    betas2: list of betas values for decaying the second order stats (if None, will be beta**2)
    eps: for numerical precision
    num_iter:
        If this is None, is ignore.

        If a number, then instead of having the betting fraction be
        1/sqrt(v), we do 1/sqrt(v*(1-beta2)/(1-beta2**current_iter)  * num_iter)
        That is, we compute the debiased average v*(1-beta2)/(1-beta2**current_iter), and then
        rescale this by the number of iterations that we think are going to happen.

        If a string, it must be equal to 'anytime'.
        This will behave the same as if it is a number, but we will use current_iter as the guess
        for the number of iterations.
    bet_fraction_type:
        whether to use the original betting fraction ('sqrt')
        or a theoreically maybe-better one ('ftrl')

    """

    if betas2 is None:
        betas2 = [beta**2 for beta in betas]

    betas = jnp.array(betas)
    betas2 = jnp.array(betas2)

    def make_copies(X, n=len(betas)):
        # Ensure X is an array
        X = jnp.array(X)
    
        # Ensure n is within bounds
        if n <= 0:
            raise ValueError("Invalid value of n, it must be greater than 0.")
    
        # Repeat the array X along the last axis
        Y = jnp.repeat(X[..., None], n, axis=-1)
    
        return Y
    def tree_make_copies(t):
        return jtu.tree_map(
            make_copies,
            t
        )

    def tree_make_zero_init(t):
        return jtu.tree_map(
            make_zero_init,
            t
        )

    def make_zero_init(x):
        return make_copies(jnp.zeros_like(x))
    

    def init_fn(s_init: jax.Array):
        state = OptaxTunerState(
            reward=tree_make_zero_init(s_init),
            s_init=util.tree_copy(s_init),
            max_grad=tree_make_zero_init(s_init),
            sum_squared_grad=tree_make_zero_init(s_init),
            sum_grad=tree_make_zero_init(s_init),
            s_values=tree_make_copies(s_init),
            iter_count=0,
        )
        return state

    def update_fn(grads, state, summed_s_value):
        grads = tree_make_copies(grads)
        # jtu.tree_map(
        #     lambda x: x[...,None],
        #     grads
        # )
        
        clipped_grads = jtu.tree_map(
            lambda g_i, m_i: jax.lax.clamp(-m_i, g_i, m_i), grads, state.max_grad
        )
        next_max_grad = jtu.tree_map(
            lambda g_i, m_i: jnp.maximum(betas * m_i, jnp.abs(g_i) + eps),
            grads,
            state.max_grad,
        )

        next_iter_count = state.iter_count + 1
        if num_iter is None:
            next_sum_squared_grad = jtu.tree_map(
                lambda v_i, g_i: betas2 * v_i + g_i**2, state.sum_squared_grad, grads
            )
            next_sum_grad = jtu.tree_map(
                lambda m_i, g_i: betas * m_i + g_i, state.sum_grad, grads
            )
            debiased_next_sum_grad = next_sum_grad
            debiased_next_sum_squared_grad = next_sum_squared_grad
        else:
            next_sum_squared_grad = jtu.tree_map(
                lambda v_i, g_i: betas2 * v_i + (1 - betas2) * g_i**2,
                state.sum_squared_grad,
                grads,
            )
            next_sum_grad = jtu.tree_map(
                lambda m_i, g_i: betas * m_i + (1 - betas) * g_i, state.sum_grad, grads
            )
            debiased_next_sum_grad = jtu.tree_map(
                lambda m_i: m_i / (1.0 - betas**next_iter_count),
                next_sum_grad,
            )
            debiased_next_sum_squared_grad = jtu.tree_map(
                lambda v_i: v_i / (1.0 - betas2**next_iter_count),
                next_sum_squared_grad,
            )

        next_reward = jtu.tree_map(
            lambda r_i, s_i, g_i: betas * r_i - g_i * s_i,
            state.reward,
            state.s_values,
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
            lambda m_i, r_i, s_init_i: s_init_i * m_i + jnp.clip(r_i, 0),
            next_max_grad,
            next_reward,
            state.s_init
        )

        if num_iter is None:
            beta_scaling = 1.0
        elif num_iter == "anytime":
            beta_scaling = 1.0 / jnp.sqrt(next_iter_count)
        elif num_iter == "usebeta1":
            beta_scaling = jnp.sqrt(1 - betas)
        elif num_iter == "usebeta2":
            beta_scaling = jnp.sqrt(1 - betas2)
        else:
            beta_scaling = 1.0 / jnp.sqrt(num_iter)

        if bet_fraction_type == "ftrl":
            bet_fraction = jtu.tree_map(
                lambda m_i, v_i: jnp.clip(-m_i, a_min=0)
                / (v_i + eps)
                * beta_scaling**2,
                debiased_next_sum_grad,
                debiased_next_sum_squared_grad,
            )
        elif bet_fraction_type == "sqrt":
            bet_fraction = jtu.tree_map(
                lambda v_i: 1.0 / (jnp.sqrt(v_i) + eps) * beta_scaling,
                debiased_next_sum_squared_grad,
            )

        next_s_values = jtu.tree_map(lambda w_i, b_i: w_i * b_i, wealth, bet_fraction)

        next_state = OptaxTunerState(
            reward=next_reward,
            s_init=state.s_init,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            sum_grad=next_sum_grad,
            s_values=next_s_values,
            iter_count=next_iter_count,
        )

        next_summed_s_value = jtu.tree_map(
            lambda s: jnp.sum(s, axis=-1)/len(betas),
            next_s_values
        )
        prev_summed_s_value = jtu.tree_map(
            lambda s: jnp.sum(s, axis=-1)/len(betas),
            state.s_values
        )


        updates = tree_subtract(next_summed_s_value ,prev_summed_s_value)

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)




class MechanicState(NamedTuple):
    offset: optax.Updates  # this is the Delta in the paper.
    base_state: optax.OptState
    tuner_state: optax.OptState
    s: jax.Array
    iter_count: jax.Array
    logging: logstate.Log

def per_layer_mechanize(
    base_optimizer,
    s_init=1e-8,
    beta=[1.0,1.0,1.0,1.0,1.0,1.0],
    weight_decay=0.0,
    betas2=[0.9,0.99,0.999,0.9999,0.99999,0.999999],
    num_iter='anytime',
    bet_fraction_type="sqrt",
    freeze_s_iteration: Optional[int] = None,
    tuner_lr: float = 1.0,
    tuner_decay_schedule="constant",
    ):
    return optax_like_mechanize(
        base_optimizer,
        s_init,
        betas2,
        weight_decay,
        betas2,
        num_iter,
        bet_fraction_type,
        freeze_s_iteration,
        tuner_lr,
        tuner_decay_schedule
    )

def optax_like_mechanize(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    betas: List[float] = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    betas2=None,
    num_iter=None,
    bet_fraction_type="sqrt",
    freeze_s_iteration: Optional[int] = None,
    tuner_lr: float = 1.0,
    tuner_decay_schedule="constant",
    **kwargs
) -> optax.GradientTransformation:
    """
    re-implement the original mechanic in optax using this framework (with some changes).

    s_init:
        initial s value
    betas:
        list of  betas for the tuner
    weight_decay:
        the weight decay (lambda) value
    betas2:
        beta2 values for the tuner (it's a newer feature, see tuner doc string - set to None to use defaults)
    num_iter:
        total number of iterations for use by the tuner. Set to None to recover original behavior  (see tuner
        docstring)
    bet_fraction_type:
        argument for tuner, see tuner docstring.
    tuner_lr: scale the tuner updates by this amount.
    freeze_s_iteration: after this many iterations, stop updating s. If we are using randomized
        scaling in the incremental update, also stop applying the randomized scaling.
    tuner_decay_schedule: schedule to apply to the tuner updates. Can be:
        'constant' (no schedule)
        'linear' (linear decay)

    """
    tuner = optax_tuner(betas=betas, num_iter=num_iter, betas2=betas2, bet_fraction_type=bet_fraction_type)
    return mechanize(
        base_optimizer,
        tuner_optimizer=tuner,
        s_init=s_init / len(betas),
        weight_decay=weight_decay,
        **kwargs
    )


def mechanize(
    base_optimizer: optax.GradientTransformation,
    tuner_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    betas=[1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    per_layer: bool = False,
    averaging_momentum: float = 0.0,
    freeze_s_iteration: Optional[int] = None,
    tuner_lr: float = 1.0,
    tuner_decay_schedule="constant",
) -> optax.GradientTransformation:
    """
    Args:
    base_optimizer: the base optimizer to learn learning rate for.
    tuner_optimizer: the optimizer  to  use to learn the learning rate.
    s_init: initial learning rate used to initialize tuner_optimizer.
    betas: list of beta values for the tuner (only relevant if tuner_optimizer is not specified)
    weight_decay: same as lambda value in original mechanic (probably should have a better name)
    per_layer: if true, learn a per-layer scale.
    averaging_momentum: funny kind of momentum to use with non-incremental mechanic.
        update_{t+1} = update_t + ((average of iterates x_t) - x_1)
        This is distinct from iterate averaging. Setting to 0 turns it off.
        So far, this never helps :)
    tuner_lr: scale the tuner updates by this amount.
    freeze_s_iteration: after this many iterations, stop updating s. If we are using randomized
        scaling in the incremental update, also stop applying the randomized scaling.
    tuner_decay_schedule: schedule to apply to the tuner updates. Can be:
        'constant' (no schedule)
        'linear' (linear decay)
    """

    if tuner_decay_schedule == "constant":
        tuner_decay_fn = lambda t, updates: jtu.tree_map(
            lambda x: x * tuner_lr, updates
        )
    elif tuner_decay_schedule == "linear":
        tuner_decay_fn = lambda t, updates: jtu.tree_map(
            lambda x: x * tuner_lr * (freeze_s_iteration - t) / freeze_s_iteration, updates
        )
    else:
        raise ValueError("unknown tuner_decay_schedule")

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
            iter_count=0,
            logging=logstate.Log(
                {
                    "reward": 0.0,
                    "reward_std": 0.0,
                    "mechanic/max_s": 0.0,
                    "mechanic/min_s": 0.0,
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
        reward = state.logging.data["reward"]
        reward_std = state.logging.data["reward_std"]
        iter_count = state.iter_count

        base_updates, next_base_state = base_optimizer.update(grads, base_state, params)
        # base updates is "u" in the paper. Add this to Delta to get the next
        # value of Delta.
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

        else:
            inner_product = jtu.tree_map(
                lambda o, g, s, p: jnp.sum(
                    o * (
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

        next_reward_std = jnp.sqrt(reward_std**2 + reward_increment**2)

        # jax.debug.print("grads: {g}, inner product: {i}",g=grads, i=inner_product)
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
        max_s = jtu.tree_reduce(lambda a, b: jnp.maximum(a, b), next_s)
        min_s = jtu.tree_reduce(lambda a, b: jnp.minimum(a, b), next_s)

        def compute_update_global(base_i, next_offset_i, offset_i):
            # mechanic update is:
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
            base_i, next_offset_i, old_s_i, next_s_i, s_update_i, offset_i
        ):
            if averaging_momentum == "iter_count":
                avmom = 1.0 / (iter_count + 1)
            else:
                avmom = averaging_momentum
            standard_update = (
                base_i * old_s_i
                + next_offset_i * s_update_i
                + offset_i * old_s_i * avmom
            )
            # update to apply if we have stopped updating s
            frozen_update = base_i * old_s_i + offset_i * old_s_i * avmom

            if freeze_s_iteration is not None:
                return jax.lax.cond(
                    should_freeze_s, lambda: frozen_update, lambda: standard_update
                )
            else:
                return standard_update

        if per_layer:
            updates = jtu.tree_map(
                compute_update_per_layer,
                base_updates,
                next_offset,
                s,
                next_s,
                s_update,
                offset,
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
            iter_count=iter_count + 1,
            logging=logstate.Log(
                {
                    "reward": reward + reward_increment,
                    "reward_std": next_reward_std,
                    "mechanic/max_s": max_s,
                    "mechanic/min_s": min_s,
                    "mechanic/offset_norm": tree_norm(next_offset),
                    "mechanic/scaled_offset_norm": scaled_offset_norm,
                }
            ),
        )

        return updates, next_state

    optimizer = optax.GradientTransformation(init_fn, update_fn)

    return optimizer
