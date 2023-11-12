import duration
import time
import pathlib
import os
from jax import tree_util as jtu
import equinox as eqx


class CheckpointManager:
    def __init__(self, config):
        self.interval = duration.TrainDuration(config.train.checkpoint_interval)
        self.last_checkpoint = duration.TrainTime()
        self.checkpoint_path = pathlib.Path(config.train.checkpoint_path)

    def maybe_save(self, train_state, train_time):
        if self.interval < train_time - self.last_checkpoint:
            self.last_checkpoint = train_time.copy()
            filename = self.checkpoint_path / f"checkpoint.{int(time.time())}.tree"
            with open(filename, "wb") as fp:
                eqx.tree_serialise_leaves(fp, train_state)

    def load(self, train_state, name='latest'):
        if name == 'latest':
            files = sorted(
                self.checkpoint_path.glob("*.*.tree"),
                key=lambda x: int(str(x).split(".")[1]),
            )
            name = files[-1]
        else:
            name = self.checkpoint_path / name

        with open(name, "rb") as fp:
            return eqx.tree_deserialise_leaves(fp, train_state)
