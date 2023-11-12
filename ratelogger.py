import time

class RateLimitedLog:
    def __init__(self, log_fn, max_frequency=1.0):
        self.log_fn = log_fn
        self.max_frequency = max_frequency
        self.last_time = time.time() - 1.0 / self.max_frequency
        self.metrics = {}

    def __call__(self, metrics, *args, commit=True, force=False, **kwargs):
        self.metrics.update(metrics)
        if commit:
            self.commit(force=force, *args, **kwargs)

    def commit(self, force=False, *args, **kwargs):
        if len(self.metrics) != 0:
            cur_time = time.time()
            if force or cur_time >= self.last_time + 1.0 / self.max_frequency:
                self.log_fn(self.metrics, *args, **kwargs)
                self.last_time = cur_time
                self.metrics = {}
