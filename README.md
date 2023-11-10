# jaxtrainloop
A training loop in jax, with some bells and whistles. Based on https://github.com/acutkosky/jaxgptc4.

* `mechanic.py` contains a customizable version of the mechanic optimizer wrapper. It doesn't implement the tuner in the official `optax.contrib.mechanize`, but it makes it easier to add new tuners.
* `duration.py` contains some utilities for handling time intervals measured in epochs, iterations, or mins/hours. The `Time` and `TimeDuration` classes are currently unused. An important one here is the `JaxTimeStamp` class, which can be used with `set_timestamp` to update all timestamps in a tree.
* `logstate.py` contains a class `LoggedState` that is intended to help plumb logging data through a pytree. Just wrap a pytree in a `LoggedState` like `LoggedState(tree, log_data)`. Typical use is for `log_data` to be a dictionary. Then you can access the original state with `state = logged_state.get_state()`. Given a pytree `tree` with some `LoggedState`s inside the tree, you can extract all logs with `list_of_logs(tree)`.

  
