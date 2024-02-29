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
    iter_count: PyTree


def optax_tuner(
    beta=1.0, eps=1e-8, num_iter=Union[None, int, str], beta2=None, bet_fraction_type='sqrt'
):
    '''
    implements the tuner as in the original mechanic implementation in optax, with a  
    few new options.
    
    beta: beta value for decaying the reward.
    beta2: beta value for decaying the second order stats (if None, will be beta**2)
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

    '''      

    
    if beta2 is None:
        beta2 = beta**2

    def init_fn(s_init: PyTree):
        state = OptaxTunerState(
            reward=optu.tree_zeros_like(s_init),
            s_init=util.tree_copy(s_init),
            max_grad=optu.tree_zeros_like(s_init),
            sum_squared_grad=optu.tree_zeros_like(s_init),
            sum_grad=optu.tree_zeros_like(s_init),
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
            next_sum_grad = jtu.tree_map(
                lambda m_i, g_i: beta * m_i + g_i, state.sum_grad, grads
            )
            debiased_next_sum_grad = next_sum_grad
            debiased_next_sum_squared_grad = next_sum_squared_grad
        else:
            next_sum_squared_grad = jtu.tree_map(
                lambda v_i, g_i: beta2 * v_i + (1 - beta2) * g_i**2,
                state.sum_squared_grad,
                grads,
            )
            next_sum_grad = jtu.tree_map(
                lambda m_i, g_i: beta * m_i + (1-beta) * g_i, state.sum_grad, grads
            )
            debiased_next_sum_grad = jtu.tree_map(
                lambda m_i: m_i / (1.0 - beta**next_iter_count),
                next_sum_grad,
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


        if bet_fraction_type == 'ftrl':
            bet_fraction = jtu.tree_map(
                lambda m_i, v_i: jnp.max(-m_i, 0)/(v_i*beta_scaling + eps),
                debiased_next_sum_grad,
                debiased_next_sum_squared_grad
            )
        elif bet_fraction_type == 'sqrt':
            bet_fraction = jtu.tree_map(
                lambda v_i: 1.0/(jnp.sqrt(v_i) * beta_scaling + eps),
                debiased_next_sum_squared_grad
            )
        
        next_s = jtu.tree_map(
            lambda w_i, b_i: w_i * b_i,
            wealth,
            bet_fraction
        )

        next_state = OptaxTunerState(
            reward=next_reward,
            s_init=state.s_init,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            sum_grad=next_sum_grad,
            iter_count=next_iter_count,
        )

        updates = tree_subtract(next_s, s_values)

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class AdditionState(NamedTuple):
    substates: List[optax.OptState]
    subparams: List[optax.Params]


# for some reasonn this thing compiles EXTREMELY SLOWLY
# if the base optimizers are per-layer and there is more than one of them...
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

        next_substates = [u__s[1] for u__s in updates__state]
        subupdates = [u__s[0] for u__s in updates__state]
        next_subparams = optax.apply_updates(subparams, subupdates)

        next_state = AdditionState(next_substates, next_subparams)

        next_updates = jtu.tree_map(
            lambda *u_i: jnp.sum(jnp.array(u_i), axis=0) / len(updates__state),
            *subupdates
        )
        return next_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class MechanicState(NamedTuple):
    offset: optax.Updates  # this is the Delta in the paper.
    base_state: optax.OptState
    tuner_state: optax.OptState
    s: jax.Array
    iter_count: jax.Array
    logging: logstate.Log


def summed_optax_tuner(
    betas=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], num_iter=None, betas2=None, **kwargs
):
    if betas2 is None:
        betas2 = [None for b in betas]
    tuners = [
        optax_tuner(beta=beta, num_iter=num_iter, beta2=beta2, **kwargs)
        for beta, beta2 in zip(betas, betas2)
    ]
    return add_optimizers(tuners)


def optax_like_mechanize(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    betas: List[float] = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    betas2=None,
    num_iter=None,
    bet_fraction_type='sqrt',
    **kwargs
) -> optax.GradientTransformation:
    """
    re-implement the original mechanic in optax using this framework.

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
    
    """
    tuner = summed_optax_tuner(betas, num_iter, betas2=betas2, bet_fraction_type=bet_fraction_type)
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
    freeze_s_iteration: after this many iterations, stop updating s. If we are using randomized
        scaling in the incremental update, also stop applying the randomized scaling.
    tuner_decay_schedule: schedule to apply to the tuner updates. Can be:
        'constant' (no schedule)
        'linear' (linear decay)
    """

    if tuner_decay_schedule == "constant":
        tuner_decay_fn = lambda t, updates: updates
    elif tuner_decay_schedule == "linear":
        tuner_decay_fn = lambda t, updates: jtu.tree_map(
            lambda x: x * (freeze_s_iteration - t) / freeze_s_iteration, updates
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
