import time
import numpy as np
import jax
from jax import numpy as jnp
import equinox as eqx
from typing import Optional, Dict, Sequence
from jaxtyping import PyTree
from jax import tree_util as jtu
import re



def safe(x):
    if eqx.is_array(x):
        return np.array(x)
    return x


# all time is relative to Nov 7th 2023 (the day this code was written)
# this is to avoid float32 roundoff headaches.
# This hack will stop working eventually of course.
START_TIME = 1699401021


def offset_hrs():
    return offset_time() / (60 * 60)


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


TIME_KEYS = ["ep", "it", "hr", "ex", "tok"]


def number_regex_builder(decimal_allowed: bool, unit_name: str):
    """
    makes a regex that looks for things like 23123.123min
    """
    number_group = r"\d+"
    if decimal_allowed:
        number_group += r"(?:\.\d*)?"

    # non-capturing group
    unit_group = f"(?:{unit_name})"

    return f"^({number_group}){unit_name}$"


def parse_time_str(spec: str):
    unit_to_value = {}
    # epochs
    regex = number_regex_builder(False, "ep")
    match = re.match(regex, spec)
    if match:
        unit_to_value["ep"] = int(match.group(1))

    # iterations
    regex = number_regex_builder(False, "it")
    match = re.match(regex, spec)
    if match:
        unit_to_value["it"] = int(match.group(1))

    # hours
    regex = number_regex_builder(True, "hr")
    match = re.match(regex, spec)
    if match:
        unit_to_value["hr"] = float(match.group(1))

    # minutes
    regex = number_regex_builder(True, "min")
    match = re.match(regex, spec)
    if match:
        unit_to_value["hr"] = float(match.group(1)) / 60.0

    # days
    regex = number_regex_builder(True, "day")
    match = re.match(regex, spec)
    if match:
        unit_to_value["hr"] = float(match.group(1)) * 24.0

    # examples
    regex = number_regex_builder(False, "ex")
    match = re.match(regex, spec)
    if match:
        unit_to_value["ex"] = int(match.group(1))

    # tokens
    regex = number_regex_builder(False, "tok")
    match = re.match(regex, spec)
    if match:
        unit_to_value["tok"] = int(match.group(1))

    if len(unit_to_value) == 0:
        raise ValueError("unparseable time string!")

    return unit_to_value


class TrainDuration(eqx.Module):
    unit_to_value: Dict[str, jax.Array]

    def __init__(self, *specs, **kw_spec):
        unit_to_value = {k: None for k in TIME_KEYS}
        if len(specs) > 0:
            if isinstance(specs[0], TrainDuration):
                unit_to_value = specs[0].unit_to_value
            else:
                for spec in specs:
                    unit_to_value.update(parse_time_str(spec))

        for unit, value in kw_spec.items():
            assert unit in TIME_KEYS or unit in ["min", "day"]
            if unit == "min":
                unit_to_value["hr"] = value / 60.0
            elif unit == "day":
                unit_to_value["hr"] = value * 24
            else:
                unit_to_value[unit] = value

        self.unit_to_value = unit_to_value

    def dict(self):
        return {k: v for k, v in self.items()}

    def loggable_dict(self, prefix=""):
        full_names = {
            'it': 'iteration',
            'ep': 'epoch',
            'tok': 'tokens',
            'ex': 'examples',
            'hr': 'hours',
        }
        result = {}
        for k,v in self.items():
            result[prefix + full_names[k]] = v
            if k == 'hr':
                result[prefix+'seconds'] = v * 60 * 60
                result[prefix+'minutes'] = v * 60
        return result


    def __contains__(self, value: str):
        try:
            return self[value] is not None
        except KeyError:
            return False

    @property
    def it(self):
        return self["it"]

    @property
    def ep(self):
        return self["ep"]

    @property
    def tok(self):
        return self["tok"]

    @property
    def ex(self):
        return self["ex"]

    @property
    def hr(self):
        return self["hr"]

    @property
    def min(self):
        return 60 * self["hr"]

    @property
    def day(self):
        return self["hr"] / 24

    def keys(self):
        for x in TIME_KEYS:
            if self[x] is not None:
                yield x

    def items(self):
        for x in self.keys():
            yield x, self[x]

    def __len__(self):
        return len(list(self.keys()))

    def __getitem__(self, k: str):
        return self.unit_to_value[k]

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return jnp.all(
            jnp.array(jtu.tree_leaves(jtu.tree_map(lambda x, y: x == y, self, other)))
        )

    def __ne__(self, other):
        return jnp.logical_not(self == other)

    def set_value(self, unit: str, value: jax.Array):
        return eqx.tree_at(lambda t: t.unit_to_value[unit], self, value)

    def get_timestamp(self):
        return 60 * 60 * self.reference_timestamp + START_TIME

    def is_compatible(self, other):
        return len(self) == len(other)

    def __truediv__(self, other):
        if isinstance(other, TrainDuration):
            results = [jnp.inf]
            for k in TIME_KEYS:
                if k in other and k in self:
                    results.append(self[k] / other[k])
            return jnp.min(jnp.array(results))
        else:
            values = {self[k] / other for k in self}
            return TrainDuration(**values)

    def __mul__(self, other):
        values = {self[k] / other for k in self}
        return TrainDuration(**values)

    def _arithmetic(self, other, operation):
        values = {k: None for k in TIME_KEYS}
        for k in TIME_KEYS:
            if k in self and k in other:
                values[k] = operation(self[k], other[k])
            if k in self and k not in other:
                values[k] = self[k]
            if k not in self and k in other:
                values[k] = other[k]
        return TrainDuration(**values)

    def __add__(self, other):
        if not isinstance(other, TrainDuration):
            other = TrainDuration(**{k: other for k in TIME_KEYS})
        return self._arithmetic(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._arithmetic(other, lambda x, y: x - y)

    def __gt__(self, other):
        return jnp.logical_and(self >= other, self != other)

    def __le__(self, other):
        return other >= self

    def __ge__(self, other):
        result = None

        results = []
        for u in self.keys():
            if u not in other:
                continue
            results.append(self[u] >= other[u])
        return jnp.all(jnp.array(results))

    def __lt__(self, other):
        return other > self

    def copy(self):
        return jtu.tree_map(lambda x: x, self)


class TrainTime(TrainDuration):
    unit_to_value: Dict[str, jax.Array]
    name: str = eqx.field(static=True)
    # reference_timestamp: jax.Array

    def __init__(self, *specs, name="train", **kw_spec):
        super().__init__(*specs, **kw_spec)
        self.name = name
        for k in TIME_KEYS:
            if self.unit_to_value[k] is None:
                self.unit_to_value[k] = 0.0

    @jax.jit
    def _update(self, **kwargs):
        kvs = list(kwargs.items())
        return eqx.tree_at(
            lambda t: [t[k] for k, v in kvs], self, [self[k] + v for k, v in kvs]
        )


class TimeUpdater:
    def __init__(self):
        self.last_update = {'train': offset_hrs()}

    def start(self, *names):
        for name in names:
            self.last_update[name] = offset_hrs()

    def update(self, train_time: TrainTime, time_delta=None, **kwargs):
        if time_delta is None:
            current_time = offset_hrs()
            time_delta = current_time - self.last_update
            self.last_update[train_time.name] = current_time
        kwargs["hr"] = time_delta
        return train_time._update(**kwargs)

    def __call__(self, ref_time: TrainTime, tree: Optional[PyTree] = None, **kwargs):
        current_time = offset_hrs()
        time_delta = current_time - self.last_update[ref_time.name]
        self.last_update[ref_time.name] = current_time
        kwargs["hr"] = time_delta
        ref_time = ref_time._update(**kwargs)
        if tree is None:
            return ref_time
        else:
            return broadcast_time(tree, ref_time)

        

@eqx.filter_jit
def broadcast_time(tree: PyTree, ref_time: TrainTime):
    def update(node):
        if isinstance(node, TrainTime) and node.name == ref_time.name:
            return ref_time.copy()
        return node

    return jtu.tree_map(update, tree, is_leaf=lambda x: isinstance(x, TrainTime))

