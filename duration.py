import time
import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Optional, Dict
from jaxtyping import PyTree
from jax import tree_util as jtu


def safe(x):
    if eqx.is_array(x):
        return np.array(x)
    return x


# all time is relative to Nov 7th 2023 (the day this code was written)
# this is to avoid float32 roundoff headaches.
# This hack will stop working eventually of course.
START_TIME = 1699401021


def offset_time():
    return convert_time(time.time())


def convert_time(t):
    return t - START_TIME


class JaxTimeStamp(eqx.Module):
    timestamp: jax.Array

    def __init__(self, value: Optional[int] = None):
        if value is None:
            value = offset_time()
        self.timestamp = jnp.array(value)


def set_timestamp(tree: PyTree, timestamp=None):
    if timestamp is None:
        timestamp = offset_time()

    def update(node):
        if isinstance(node, JaxTimeStamp):
            return JaxTimeStamp(timestamp)
        return node

    return jtu.tree_map(update, tree, is_leaf=lambda x: isinstance(x, JaxTimeStamp))


TIME_KEYS = ["epochs", "iterations", "hours"]


class Time(eqx.Module):
    epochs: Optional[int] = None
    iterations: Optional[int] = None
    hours: Optional[int] = None

    def __init__(self, spec, epochs=None, iterations=None, hours=None):
        # TODO: write this with regular expressions or something
        # else less stupid
        epochs = epochs
        iterations = iterations
        hours = hours
        if isinstance(spec, Time):
            self.epochs = spec.epochs
            self.iterations = spec.iterations
            self.hours = spec.hours
        elif isinstance(spec, str):
            if "ep" in spec:
                self.epochs = int(spec.split("ep")[0])
            if "it" in spec:
                self.iterations = int(spec.split("it")[0])
            if "min" in spec:
                self.hours = float(spec.split("min")[0]) / 60
            if "hr" in spec:
                self.hours = float(spec.split("hr")[0])

    def __contains__(self, value: str):
        try:
            return self[value] != None
        except:
            return False

    def keys(self):
        for x in TIME_KEYS:
            if self[x] is not None:
                yield x

    def items(self):
        for x in self:
            yield x, self[x]

    def __len__(self):
        return len(list(self.keys()))

    def __getitem__(self, value: str):
        if value in ["epochs", "ep"]:
            return self.epochs

        if value in ["iterations", "it"]:
            return self.iterations
        if value in ["hours", "hrs", "hr"]:
            return self.hours

        if value in ["min", "minutes", "mins"]:
            return self.hours * 60

        if value in ["sec", "seconds", "secs"]:
            return self.hours * 60 * 60

        raise KeyError

    def _greater_than(self, other, strict: bool):
        def compfunc(x, y):
            if strict:
                return x > y
            else:
                return x >= y

        result = None

        if len(self) > len(other):
            raise ValueError("comparable Time objects must have same units!")

        for u in self.keys():
            if u not in other:
                raise ValueError("comparable Time objects must have the same units!")
            if result is None:
                result = compfunc(self[u], other[u])
            elif result != compfunc(self[u], other[u]):
                raise ValueError("ambiguous comparison among time objects!")
        return result

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for u in self.keys():
            if self[u] != other[u]:
                return False
        return True

    def set_epoch(self, value):
        return Time(epochs=value, iterations=self.iterations, hours=self.hours)
    def set_iterations(self, value):
        return Time(epochs=self.epochs, iterations=value, hours=self.hours)
    def set_hours(self, value):
        return Time(epochs=self.epochs, iterations=self.iterations, hours=value)

    def is_compatible(self, other):
        return len(self) == len(other)

    def _arithmetic(self, other, operation):
        if self.is_compatible(other):
            raise ValueError("incompatible time arithmetic!")

        values = {u: operation(self[u], other[u]) for u in self.keys()}
        return Time(**values)

    def __add__(self, other):
        return self._arithmetic(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._arithmetic(other, lambda x, y: x - y)

    def __ge__(self, other):
        return self._comparison(other, strict=False)

    def __le__(self, other):
        return other >= self

    def __gt__(self, other):
        return self._comparison(other, strict=True)

    def __lt__(self, other):
        return other > self


def minimum(*times):
    values = {k: None for k in TIME_KEYS}

    for k in values:
        for t in times:
            if k in t:
                if values[k] is None:
                    values[k] = t[k]
                values[k] = min(values[k], t[k])
    return Time(**values)


PROGRAM_START = offset_time()


def program_start() -> Time:
    epochs = 0
    iterations = 0
    hours = PROGRAM_START


def compatable_now(interval):
    start_hours = None if interval.hours is None else offset_time()
    start_epochs = None if interval.epochs is None else 0
    start_iterations = None if interval.iterations is None else 0

    now = Time(epochs=start_epochs, hours=start_epochs, iterations=start_iterations)
    return now
 

class TimeDuration(eqx.Module):
    start_time: Time
    end_time: Time
    interval: Time

    def __init__(self, *specs, start_time=None):
        self.interval = minimum(*[Time(spec) for spec in specs])
        if self.interval.hours is not None:
            start_hours = program_start()
        else:
            start_hours = None

        if start_time is None:
            start_time = compatable_now(self.interval)
        self.start_time = start_time
        self.end_time = self.start_time + self.interval

    def reset(self, now: Time, overwrite_timestamp: bool=False):
        if overwrite_timestamp:
            now = now.set_hours(offset_time())
        start_time = now
        end_time = now + self.interval
        interval = self.interval

        return eqx.tree_at(
            lambda t: (t.start_time, t.end_time, t.interval),
            self,
            (start_time, end_time, interval),
        )


    def elapsed(self, comparison: Time):
        return comparison > end_time
        


class Duration:
    def __init__(self, *specs):
        self.durations = [_Duration(spec) for spec in specs]

    @property
    def minutes(self):
        return min([d.minutes for d in self.durations])

    @property
    def epochs(self):
        return min([d.epochs for d in self.durations])

    @property
    def iterations(self):
        return min([d.iterations for d in self.durations])

    @property
    def start_time(self):
        end_time = float("inf")
        start_time = 0.0
        for d in self.durations:
            cur_end = d.start_time + d.minutes
            if cur_end < end_time:
                start_time = d.start_time

        return start_time

    @property
    def start_epochs(self):
        end_epochs = float("inf")
        start_time = 0.0
        for d in self.durations:
            cur_end = d.start_epochs + d.epochs
            if cur_end < end_epochs:
                start_epochs = d.start_epochs

        return start_epochs

    @property
    def start_iterations(self):
        end_iterations = float("inf")
        start_time = 0.0
        for d in self.durations:
            cur_end = d.start_iterations + d.iterations
            if cur_end < end_iterations:
                start_iterations = d.start_iterations

        return start_iterations

    def elapsed(self, epoch: int, iterations: int):
        for d in self.durations:
            if d.elapsed(epoch, iterations):
                return True
        return False

    def reset(self, epoch: int = 0, iterations: int = 0):
        for d in self.durations:
            d.reset(epoch, iterations)

    def elapsed_and_reset(self, epoch: int, iterations: int):
        result = False
        for d in self.durations:
            result = result or d.elapsed_and_reset(epoch, iterations)

        return result

    def __str__(self):
        return f"{[str(d) for d in self.durations]}"


class _Duration:
    def __init__(self, spec):
        if isinstance(spec, Duration):
            self.epochs = spec.epochs
            self.iterations = spec.iterations
            self.minutes = spec.minutes
            self.start_epochs = spec.start_epochs
            self.start_iterations = spec.start_iterations
            self.start_time = spec.start_time
            return
        if spec is None:
            spec = ""

        self.epochs = float("inf")
        self.iterations = float("inf")
        self.minutes = float("inf")

        # TODO: write this with regular expressions or something
        # else less stupid
        if "ep" in spec:
            self.epochs = int(spec.split("ep")[0])
        if "it" in spec:
            self.iterations = int(spec.split("it")[0])
        if "min" in spec:
            self.minutes = float(spec.split("min")[0])
        if "hr" in spec:
            self.minutes = 60 * float(spec.split("hr")[0])

        self.reset()

    def elapsed(self, epoch: int, iterations: int):
        if epoch >= self.epochs + self.start_epochs:
            return True
        if iterations >= self.iterations + self.start_iterations:
            return True
        if offset_time() / 60 > self.minutes + self.start_time:
            return True

        return False

    def reset(self, epoch: int = 0, iterations: int = 0):
        self.start_time = offset_time() / 60
        self.start_epochs = safe(epoch)
        self.start_iterations = safe(iterations)

    def elapsed_and_reset(self, epoch: int, iterations: int):
        result = self.elapsed(epoch, iterations)
        if result:
            self.reset(epoch, iterations)
        return result

    def __str__(self):
        return f"Duration(epochs={self.epochs}, iterations={self.iterations}, minutes={self.minutes}, start_epoch={self.start_epochs}, start_iter={self.start_iterations})"
