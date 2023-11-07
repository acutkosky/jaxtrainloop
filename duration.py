import time
import numpy as np
import equinox as eqx


def safe(x):
    if eqx.is_array(x):
        return np.array(x)
    return x


class Time:
    def __init__(self, *spec_string, **spec_dict):
        # TODO: write this with regular expressions or something
        # else less stupid
        self.unit_to_value = dict(spec_dict)
        for spec in spec_string:
            if "ep" in spec:
                self.unit_to_value["ep"] = int(spec.split("ep")[0])
            if "it" in spec:
                self.unit_to_value["it"] = int(spec.split("it")[0])
            if "min" in spec:
                self.unit_to_value["min"] = float(spec.split("min")[0])
            if "hr" in spec:
                self.unit_to_value["min"] = 60 * float(spec.split("hr")[0])

    def _comparison(self, other, compfunc):
        result = None

        if len(self.unit_to_value) > len(other.unit_to_value):
            raise ValueError("comparable Time objects must have same units!")

        for u in self.unit_to_value:
            if u not in other.unit_to_value:
                raise ValueError("comparable Time objects must have the same units!")
            if result is None:
                result = compfunc(self.unit_to_value[u], other.unit_to_value)
            elif result != compfunc(self.unit_to_value[u], other.unit_to_value):
                raise ValueError("ambiguous comparison among time objects!")
        return result

    def __ge__(self, other):
        return self._comparison(other, lambda x, y: x >= y)

    def __le__(self, other):
        return other >= self

    def __gt__(self, other):
        return self._comparison(other, lambda x, y: x > y)

    def __lt__(self, other):
        return other > self


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
        if time.time() / 60 > self.minutes + self.start_time:
            return True

        return False

    def reset(self, epoch: int = 0, iterations: int = 0):
        self.start_time = time.time() / 60
        self.start_epochs = safe(epoch)
        self.start_iterations = safe(iterations)

    def elapsed_and_reset(self, epoch: int, iterations: int):
        result = self.elapsed(epoch, iterations)
        if result:
            self.reset(epoch, iterations)
        return result

    def __str__(self):
        return f"Duration(epochs={self.epochs}, iterations={self.iterations}, minutes={self.minutes}, start_epoch={self.start_epochs}, start_iter={self.start_iterations})"
