from collections import defaultdict
import time

class WindowAvg:
    def __init__(self, window_size):
        self.window = []
        self.max_size = window_size

    def update(self, value):
        self.window.append(value)
        while len(self.window) > self.max_size:
            self.window.pop(0)

    @property
    def value(self):
        if len(self.window) == 0:
            return float('nan')
        return sum(self.window)/len(self.window)


class TimeKeeper:
    def __init__(self, window_size=10):
        self.timestamps = {}
        self.average_intervals = defaultdict(lambda: WindowAvg(window_size))
        self.periods = defaultdict(lambda: WindowAvg(window_size))

    def mark(self, start_events=[], end_events={}):
        cur_time = time.time()
        for e, c in end_events.items():
            if c is None:
                c = 0
            if c > 0:
                delta = (cur_time - self.timestamps[e]) / c
                self.average_intervals[e].update(delta)
        for s in start_events:
            if s in self.timestamps:
                delta = cur_time - self.timestamps[s]
                self.periods[s].update(delta)
            self.timestamps[s] = cur_time

        return cur_time

    def get_intervals(self):
        return {k: v.value for k, v in self.average_intervals.items()}

    def get_proportions(self):
        return {
            k: self.average_intervals[k].value / self.periods[k].value
            for k in self.periods
        }

