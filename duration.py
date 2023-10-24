import time



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
        end_time = float('inf')
        start_time = 0.0
        for d in self.durations:
            cur_end = d.start_time + d.minutes
            if cur_end < end_time:
                start_time = d.start_time

        return start_time

    @property
    def start_epochs(self):
        end_epochs = float('inf')
        start_time = 0.0
        for d in self.durations:
            cur_end = d.start_epochs + d.epochs
            if cur_end < end_epochs:
                start_epochs = d.start_epochs

        return start_epochs

    @property
    def start_iterations(self):
        end_iterations = float('inf')
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

def min_duration(*durations):
    result = Duration("")

    for d in durations:
        result.epochs = min(result.epochs, d.epochs)
        result.iterations = min(result.iterations, d.iterations)
        result.minutes = min(result.minutes, d.minutes)

    curtime = time.time() / 60
    result.start_time = curtime
    result.start_iterations = result.iterations
    result.start_epochs = result.epochs
    for d in durations:
        if d.minutes != float("inf"):
            end_time = d.start_time + d.minutes
            result.start_time = min(result.start_time, end_time - result.minutes)

        if d.epochs != float("inf"):
            end_epochs = d.start_epochs + d.epochs
            result.start_epochs = min(result.start_epochs, end_epochs - result.epochs)

        if d.iterations != float("inf"):
            end_iterations = d.start_iterations + d.iterations
            result.start_iterations = min(
                result.start_iterations, end_iterations - result.iterations
            )

    return result


class _Duration:
    def __init__(self, spec):
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
        self.start_epochs = epoch
        self.start_iterations = iterations

    def elapsed_and_reset(self, epoch: int, iterations: int):
        result = self.elapsed(epoch, iterations)
        if result:
            self.reset(epoch, iterations)
        return result

    def __str__(self):
        return f"Duration(epochs={self.epochs}, iterations={self.iterations}, minutes={self.minutes}, start_epoch={self.start_epochs}, start_iter={self.start_iterations})"
