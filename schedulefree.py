import jax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax import numpy as jnp
import optax
from optax import tree_utils as optu
from typing import NamedTuple, Any, Optional, List, Tuple, Union, Callable
from jaxtyping import PyTree

import logstate


"""
original formula:
weights: w_t, alpha_t
base optimizer iterates: z_t

x_t = sum_{i=1}^t w_i z_i / \sum_{i=1}^t w_t
y_t = (1-alpha_t) x_t + alpha_t z_t

output y_t, get gradient g_t at y_t

send g_t to base optimizer to get z_{t+1}

Reparametrization in terms of "updates" for easier optax implementation
define base optimizer updates: Delta_t = z_{t+1} - z_t

define m_t = x_{t+1}- x_t
define beta_t = w_{1:t}/w_{1:t+1}
define gamma_t = w_{t+1} w_{1:t-1}/(w_{1:t}w_t)
Notice that
m_t = x_{t+1} - x_t
    = w_{t+1}/w_{1:t+1} (z_{t+1} - x_t)
    = w_{t+1}/w_{1:t+1}(Delta_t + z_t-x_t)
    = w_{t+1}/w_{1:t+1}(Delta_t + w_{1:t-1}(x_t - x_{t-1})/w_t)
    = beta_t * gamma_t * m_{t-1} + (1-beta_t) Delta_t

In the special case that w_t=1 for all t:
m_t = m_{t-1} * (t-1)/(t+1) + Delta_t / (t+1)

Now, define the updates:
u_t = y_{t+1} - y_t

Then:
u_t = y_{t+1} - y_t
    = (1-alpha_{t+1}) x_{t+1} + alpha_{t+1} z_{t+1} - (1-alpha_t) x_t - alpha_t z_t
    = (1-alpha_t) m_t + alpha_t Delta_t + (alpha_t-alpha_{t+1})(x_{t+1} - z_{t+1})
    = (1-alpha_t) m_t + alpha_t Delta_t + (alpha_t-alpha_{t+1})(x_t - x_{t+1})w_{1:t}/w_t
    = (1-alpha_t - (alpha_t - alpha_{t+1}) w_{1:t}/w_t) m_t + alpha_t Delta_t

In the special case that alpha_t is a constant alpha:

u_t = (1-alpha) m_t + alpha Delta_t

With this formula, we require only one slot (m_t) to compute the update.

If we want to recover x_T for evaluation, we can just set alpha_T = 0 and apply the update again.
We can have a special function to compute the update for this case.
For example, if alpha_t = alpha for t<T and alpha_T = 0 and w_t=1 for all t, then the final update is:
u_{T-1} = (1- T alpha)m_{T-1} + alpha Delta_{T-1}

the algebra is complicated, so let's check:
x_{T-1} (1-alpha) + alpha z_{T-1} + (1- T alpha) (x_T - x_{T-1}) + alpha (z_T - z_{T-1})
= x_{T-1} (T-1) alpha + (1- alpha T) x_T + alpha z_T
= x_T + alpha ((T-1) x_{t-1} - T x_T) + alpha z_T
= x_T

So, it turns out even to compute the final eval point x_T we don't need an extra slot either.

"""

class ScheduleMomentumState(NamedTuple):
    sum_weight: jax.Array
    current_weight: jax.Array
    current_alpha: jax.Array
    iter_count: jax.Array
    running_average_difference: PyTree
    base_state: optax.OptState
    alpha_state: Any
    weight_state: Any
    log_data: logstate.Log

# default generators for alpha ane w values

def constant_weight_fn(
    grads: optax.Updates,
    schedule_state: ScheduleMomentumState,
    params: optax.Params,
    base_updates: optax.Updates,
):
    return 1.0, schedule_state.weight_state


def constant_alpha_fn(
    alpha: float,
    grads: optax.Updates,
    schedule_state: ScheduleMomentumState,
    params: optax.Params,
    base_updates: optax.Updates,
):
    return alpha, schedule_state.alpha_state


def linear_decay_alpha(max_iter, min_value=0.0):
    def alpha_fn(
        grads: optax.Updates,
        schedule_state: ScheduleMomentumState,
        params: optax.Params,
        base_updates: optax.Updates,
    ):
        return jnp.clip(1.0 - (schedule_state.iter_count+1)/max_iter, min_value), schedule_state.alpha_state

    return alpha_fn
    

def pass_through_transform():
    """
    simple "noop" gradient transformation that passes
    the updates through unchanged
    """

    def init_fn(params: optax.Params):
        return None

    def update_fn(updates: optax.Updates, state=None, params=None):
        return updates, None

    return optax.GradientTransformation(init_fn, update_fn)



def schedule_momentum(
    alpha: Union[Callable, float] = 0.9,
    weight_fn: Callable = constant_weight_fn,
    base_optimizer: Optional[optax.GradientTransformation] = None,
    max_iter: Optional[int] = None,
    alpha_state: Any = None,
    weight_state: Any = None,
) -> optax.GradientTransformation:
    """
    generates a scheduler using the "schedule free" strategy.

    Arguments:
        alpha: function or float. If a function, then this function
            is called every iteration to generate an alpha weighting value.
            If a float, then the float is taken to be the alpha weighting
            value every iteration.

        weight_fn: function to be called every iteration to generate the weight
            values.
            The default function outputs a weight of 1.0 at every iteration.
            See below for description of these functions.

        base_optimizer: an optimizer to schedule. If not provided, the default value
            will just pass the input updates as the base_updates. This way,
            schedule_free_momentum can be easily used with optax.chain like:

            optax.chain(optax.adamw(...), schedule_free_momentum())

        max_iter: how many iterates to train for. If provided, will force
            alpha = 0 for the final update, thus providing a last-iterate
            guarantee.
            Actually, it's fine (in theory) for this to be an underestimate of
            the actual number of iterateions: we will just continue setting
            alpha=0 for any iterations greater than this max_iter which
            will continue to have a last-iterate guarantee.

        alpha_state:
            internal state for the alpha generating function. Not needed 
            for default settings.
        weight_state:
            internal state for the weight generating function. Not needed
            for default settings.

    returns:
        a GradientTransformation that applies the schedule.


    Information about alpha and weight_fn:

    both alpha and weight_fn should have the signature:
    alpha_or_weight_fn(
        grads: optax.Updates,
        schedule_state: ScheduleFreeState
        params: optax.Params,
        base_updates: optax.Updates,
    ) -> Tuple[float, Any]

    The return value should be a tuple whose first entry is the alpha or weight value,
    and the second entry is an updated value for the "alpha_state" or "weight_state" field
    in the ScheduleFreeState for the scheduler.

    The inputs grads, schedule_state, params are the inputs to the scheduler gradient transformation.
    The base_updates are the updates produced by the base optimizer.

    """


    if base_optimizer is None:
        base_optimizer = pass_through_transform()

    if not callable(alpha):
        alpha_fn = jtu.Partial(constant_alpha_fn, alpha)
    else:
        alpha_fn = alpha

    def init_fn(params: optax.Params) -> ScheduleMomentumState:
        sum_weight = 1.0
        current_weight = 1.0
        current_alpha = 1.0
        iter_count = 0
        m = optu.tree_zeros_like(params)
        base_state = base_optimizer.init(params)

        log_data = logstate.Log(
            {
                "schedule_momentum/current_weight": 1.0,
                "schedule_momentum/current_alpha": 1.0,
                "schedule_momentum/sum_weight": 1.0,
                "schedule_momentum/running_average_diff_norm": 0.0,
                "schedule_momentum/base_opt_norm": 0.0,
                "schedule_momentum/final_update_norm": 0.0,
            }
        )

        return ScheduleMomentumState(
            sum_weight=sum_weight,
            current_weight=current_weight,
            current_alpha=current_alpha,
            iter_count=iter_count,
            running_average_difference=m,
            base_state=base_state,
            alpha_state=alpha_state,
            weight_state=weight_state,
            log_data=log_data,
        )


    def update_fn(
        grads: optax.Updates,
        state: ScheduleMomentumState,
        params: Optional[optax.Params] = None,
    ):
        next_iter_count = state.iter_count + 1

        weighted_grads = optu.tree_scalar_mul(state.current_weight, grads)

    
        base_updates, next_base_state = base_optimizer.update(
            weighted_grads, state.base_state, params
        )

        next_alpha, next_alpha_state = alpha_fn(grads, state, params, base_updates)

        if max_iter is not None:
            # if we have a max_iter value, then set alpha to 0 if
            # we're about to hit the max iter.
            # note that it's ok to keep training after the max_iter:
            # this will just continue setting alpha=0, which
            # will cause the algorithm to switch to primal/anytime averaging.
            # In theory, every iterate after that this point is a "good"
            # eval iterate.
            next_alpha = next_alpha * (next_iter_count < max_iter)
    

        next_weight, next_weight_state = weight_fn(grads, state, params, base_updates)

        next_sum_weight = state.sum_weight + next_weight
        prev_sum_weight = state.sum_weight - state.current_weight

        m_decay_factor = (
            next_weight * prev_sum_weight / (state.current_weight * next_sum_weight)
        )

        next_running_average_difference = jtu.tree_map(
            lambda m_i, d_i: m_i * m_decay_factor + d_i * next_weight / next_sum_weight,
            state.running_average_difference,
            base_updates,
        )

        update_m_factor = (
            1.0
            - state.current_alpha
            + (next_alpha - state.current_alpha) * state.sum_weight / next_weight
        )

        updates = jtu.tree_map(
            lambda m_i, d_i: m_i * update_m_factor + d_i * state.current_alpha,
            next_running_average_difference,
            base_updates,
        )

        log_data = logstate.Log(
            {
                "schedule_momentum/current_weight": next_weight,
                "schedule_momentum/current_alpha": next_alpha,
                "schedule_momentum/sum_weight": next_sum_weight,
                "schedule_momentum/running_average_diff_norm": optu.tree_l2_norm(
                    next_running_average_difference
                ),
                "schedule_momentum/base_opt_norm": optu.tree_l2_norm(base_updates),
                "schedule_momentum/final_update_norm": optu.tree_l2_norm(updates),
            }
        )

        next_state = ScheduleMomentumState(
            sum_weight=next_sum_weight,
            current_weight=next_weight,
            current_alpha=next_alpha,
            iter_count=next_iter_count,
            running_average_difference=next_running_average_difference,
            base_state=next_base_state,
            alpha_state=next_alpha_state,
            weight_state=next_weight_state,
            log_data=log_data,
        )

        return updates, next_state



    return optax.GradientTransformation(init_fn, update_fn)
