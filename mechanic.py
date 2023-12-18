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
    small_value: float = 1e-10,
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

        # clipped_updates = jtu.tree_map(
        #     lambda u_i, s_i: jnp.clip(u_i, -jnp.sqrt(s_i), jnp.sqrt(s_i)),
        #     updates,
        #     # max_grad,
        #     sum_squared_grad,
        # )
        clipped_updates = updates

        # clipped_updates = jtu.tree_map(
        #     lambda u_i, m_i: jnp.clip(u_i, -m_i, m_i),
        #     updates,
        #     max_grad,
        #     # sum_squared_grad,
        # )
        next_sum_squared_grad = jtu.tree_map(
            lambda sum_i, u_i: beta**2 * sum_i + u_i**2, sum_squared_grad, updates
        )
        next_max_grad = jtu.tree_map(
            lambda m_i, u_i: jnp.maximum(beta * m_i, jnp.abs(u_i)), max_grad, updates
        )

        def link_fn(theta, V, M, init_i):
            V = V + small_value
            M = M + small_value
            # exponent = jax.lax.cond(jnp.abs(theta) < V/M, lambda: theta**2 /V, lambda : 2*theta/M-V/M**2)
            # return jnp.sign(theta) * epsilon * (jnp.exp(exponent) -1)
            # return jnp.sign(theta) * epsilon * (jnp.exp(theta**2 / V) - 1)
            return init_i * jnp.exp(theta / jnp.sqrt(V))
            # return jnp.sign(theta) * epsilon * (jnp.exp(jnp.abs(theta)/jnp.sqrt(V)) - 1)

        def inv_link_fn(p, V, M, init_i):
            V = V + small_value
            M = M + small_value
            return jnp.log(p / init_i) * jnp.sqrt(V)

            # exponent = jnp.log(jnp.abs(p)/epsilon + 1)
            # pred = exponent < jnp.exp(V/M**2)
            # x1 = jnp.sqrt(exponent) * V
            # x2 = 0.5 * M * (exponent + V/M)
            # jax.debug.print("pred: {}, x1: {}, x2: {}", pred, x1, x2)
            # print(pred, x1, x2)
            # print(len(pred))
            # theta = jax.lax.cond(exponent < jnp.exp(V/M**2), lambda: jnp.sqrt(exponent * V), lambda : 0.5 * M * (exponent  + V/M**2))
            # return theta
            # return jnp.sqrt(jnp.log(jnp.abs(p)/epsilon + 1) * V ) * jnp.sign(p)
            # return jnp.log(jnp.abs(p)/epsilon + 1) * jnp.sqrt(V) * jnp.sign(p)

        def get_next_param(p_i, init_i, u_i, old_sum_i, next_sum_i, m_i):
            # old_theta = inv_link_fn(p_i - init_i, next_sum_i, m_i, init_i)
            old_theta = inv_link_fn(p_i, next_sum_i, m_i, init_i)
            theta = beta * old_theta - u_i  # - u_i**2/jnp.sqrt(next_sum_i)
            next_p_i = link_fn(theta, next_sum_i, m_i, init_i)
            # jax.debug.print("theta: {}, next_sum_i: {}, u_i: {}", theta, jnp.sqrt(next_sum_i), u_i)
            # next_p_i = link_fn(theta, next_sum_i, m_i, init_i) + init_i
            # fake_next_pi = link_fn(old_theta, next_sum_i, m_i) + init_i
            # next_p_i = jnp.sign(theta) * ((jnp.abs(p_i) + epsilon) * jnp.exp((-2 * u_i * old_theta + u_i**2)/(next_sum_i+small_value)) - epsilon)
            # fake_next_pi = jnp.sign(old_theta) * ((jnp.abs(p_i) + epsilon) * jnp.exp((-2 * 0 * old_theta + 0**2)/(next_sum_i+small_value)) - epsilon)
            # jax.debug.print("old_theta: {}, u: {}, p: {}, next_p: {}, v: {}, beta: {}, fake: {}",old_theta, u_i, p_i, next_p_i, next_sum_i, beta, fake_next_pi)#epsilon)
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


def optax_tuner(beta=1.0, eps=1e-8):
    def init_fn(s_init: PyTree):
        state = OptaxTunerState(
            reward=optu.tree_zeros_like(s_init),
            s_init=util.tree_copy(s_init),
            max_grad=optu.tree_zeros_like(s_init),
            sum_squared_grad=optu.tree_zeros_like(s_init),
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

        next_sum_squared_grad = jtu.tree_map(
            lambda v_i, g_i: beta**2 * v_i + g_i**2, state.sum_squared_grad, grads
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

        next_s = jtu.tree_map(
            lambda w, v: w / (jnp.sqrt(v) + eps),
            wealth,
            next_sum_squared_grad,
        )

        next_state = OptaxTunerState(
            reward=next_reward,
            s_init=state.s_init,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
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

        next_substates = [u__s[1] for u__s in updates__state]
        subupdates = [u__s[0] for u__s in updates__state]
        next_subparams = optax.apply_updates(subparams, subupdates)

        next_state = AdditionState(next_substates, next_subparams)

        updates = jtu.tree_map(
            lambda *u_i: jnp.sum(jnp.array(u_i), axis=0) / len(updates__state),
            *[u__s[0] for u__s in updates__state]
        )
        # jax.debug.print("new state: {}",new_state)

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class MechanicState(NamedTuple):
    offset: optax.Updates  # this is the Delta in the paper.
    base_state: optax.OptState
    tuner_state: optax.OptState
    s: jax.Array
    key: jax.Array
    logging: logstate.LoggedState


def mechanize_no_beta(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    optimistic: bool = False,
    weight_decay: float = 0.0,
    incremental: bool=False,
) -> optax.GradientTransformation:
    if optimistic:
        tuner = optimistic_mirror_descent_tuner(beta=1.0)
    else:
        tuner = mirror_descent_tuner(beta=1.0)
    return mechanize(
        base_optimizer,
        tuner_optimizer=tuner,
        s_init=s_init,
        weight_decay=weight_decay,
        incremental=incremental,
    )


def summed_optax_tuner(betas=[0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]):
    tuners = [optax_tuner(beta) for beta in betas]
    return add_optimizers(tuners)


def optax_mechanize(
    base_optimizer: optax.GradientTransformation,
    s_init: float = 1e-8,
    betas: List[float] = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    incremental: bool =  False,
) -> optax.GradientTransformation:
    tuner = summed_optax_tuner(betas)
    return mechanize(
        base_optimizer,
        tuner_optimizer=tuner,
        s_init=s_init / len(betas),
        weight_decay=weight_decay,
        incremental=incremental,
    )


def summed_mirror_descent(
    betas=[1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999], optimistic=False
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

    md_tuners = [tuner_factory(beta=b, epsilon=eps) for b, eps in zip(betas, epsilons)]
    return add_optimizers(md_tuners)


# class AdaptiveMDState(NamedTuple):
#     sum_squared_grad: optax.Updates
    
# def adaptive_md(
#     lr: float=1e-3,
#     regularizer: float=1e-3,
# ) -> optax.GradientTransformation:

#     def init_fn(params: optax.Params):

#         state = AdaptiveMDState(
#             sum_squared_grad=jtu.tree_map(jnp.zeros_like, params),
#         )

#     def update_fn(grads: optax.Updates, state: AdaptiveMDState, params: Optax.Params):

#         def get_update(g_i, p_i, v_i):
#             next_p_i = p_i - g_i * lr / jnp.sqrt(v_i + small_value)
        
    

def mechanize(
    base_optimizer: optax.GradientTransformation,
    tuner_optimizer: optax.GradientTransformation = None,
    s_init: float = 1e-8,
    optimistic: bool = False,
    betas=[1.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999],
    weight_decay: float = 0.0,
    incremental: bool = False,
) -> optax.GradientTransformation:
    if tuner_optimizer is None:
        tuner_optimizer = summed_mirror_descent(betas, optimistic)

    def init_fn(params: optax.Params):
        offset = jtu.tree_map(jnp.zeros_like, params)
        base_state = base_optimizer.init(params)
        s = jnp.array(s_init)
        tuner_state = tuner_optimizer.init(s)

        return MechanicState(
            offset=offset,
            base_state=base_state,
            tuner_state=tuner_state,
            s=s,
            key=jax.random.PRNGKey(1231),
            logging=logstate.LoggedState(None, {'reward': 0.0})
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
        reward = state.logging.get_logs()['reward']

        random_scale = jax.random.exponential(to_use)

        base_updates, next_base_state = base_optimizer.update(grads, base_state, params)
        # base updates is "u" in the paper. Add this to Delta to get the next
        # value of Delta.
        if incremental:
            next_offset = base_updates
        else:
            next_offset = tree_add(offset, base_updates)

        inner_product = tree_dot(
            offset,
            tree_add(
                grads,
                tree_scale(
                    params,
                    state.s * weight_decay * tree_norm(grads) / (tree_norm(params) + 1e-8),
                ),
            ),
        )

        # jax.debug.print("grads: {g}, inner product: {i}",g=grads, i=inner_product)

        s_update, next_tuner_state = tuner_optimizer.update(
            inner_product, tuner_state, s
        )

        next_s = s + s_update

        def compute_update(base_i, offset_i):
            # update is:
            # next_offset * next_s - offset * s
            # = (offset + base_update) * (s + s_update) - offset * s
            # = base_update * s + s_update * next_offset
            if incremental:
                return base_i * s + offset_i * s_update
            else:
                return base_i * next_s * random_scale

        updates = jtu.tree_map(compute_update, base_updates, next_offset)

        next_state = MechanicState(
            offset=next_offset,
            base_state=next_base_state,
            tuner_state=next_tuner_state,
            s=next_s,
            key=next_key,
            logging=logstate.LoggedState(None,{'reward':reward +  s*inner_product})
        )

        return updates, next_state
    optimizer = optax.GradientTransformation(init_fn, update_fn)
    # if incremental:
    #     optimizer = otnc.random_scale('exponential', jax.random.PRNGKey(7213), optimizer)

    return optimizer
