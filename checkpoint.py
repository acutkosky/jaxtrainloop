import duration
import time
import pathlib
import equinox as eqx


class CheckpointManager:
    def __init__(self, config):
        if config.train.checkpoint_interval is not None:
            self.interval = duration.TrainDuration(config.train.checkpoint_interval)
        else:
            self.interval = None
        self.last_checkpoint = duration.TrainTime()
        if config.train.checkpoint_path is not None:
            self.checkpoint_path = pathlib.Path(config.train.checkpoint_path)
        else:
            self.checkpoint_path = None

    def maybe_save(self, train_state, train_time):
        if self.interval is None:
            return
        if self.checkpoint_path is None:
            return
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
