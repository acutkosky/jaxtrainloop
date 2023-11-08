import jax
from typing import NamedTuple, Callable
from jaxtyping import PyTree
from jax import tree_util as jtu
from jax import numpy as jnp
import equinox as eqx


class LoggedState(eqx.Module):
    _state: PyTree
    _log_data: PyTree

    def __init__(self, state: PyTree, log_data: PyTree):
        self._state = state
        self._log_data = log_data

    def __iter__(self):
        yield self._state
        yield self._log_data

    def get_state(self):
        return self._state

    def get_logs(self):
        return self._log_data

    def __getattr__(self, name):
        return getattr(self._state, name)


def map_logs(fn: Callable, tree: PyTree, state_fn: Callable = lambda x: x):
    def map_fn(logged_state):
        if not isinstance(logged_state, LoggedState):
            return state_fn(logged_state)
        state = logged_state.get_state()
        log_data = logged_state.get_logs()

        state = map_logs(fn, state)
        log_data = fn(log_data)
        return LoggedState(state, log_data)

    return jtu.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, LoggedState))


def prune_logs(tree: PyTree):
    def map_fn(logged_state):
        if not isinstance(logged_state, LoggedState):
            return logged_state
        else:
            return prune_logs(logged_state.state)

    pruned = jtu.tree_map(map_fn, tree, is_leaf=lambda x: isinstance(x, LoggedState))
    logs = filter_logs(tree)
    return pruned, logs


def filter_logs(tree: PyTree):
    return map_logs(lambda x: x, tree, state_fn=lambda x: None)


def list_of_logs(tree: PyTree):
    result = []
    map_logs(result.append, tree)
    return result


def set_all_logs(tree: PyTree, value=None):
    return map_logs(lambda x: value, tree)
