"""Experiment logging for PyTorch.

    import atalaya as atl

    atl.init(project="mnist", config={"lr": 1e-3})   # logs/<name>/
    for epoch in range(10):
        with atl.timer("train"):
            loss = train()
        atl.log({"loss": loss})              # from any module of this process
    atl.finish()

    df = atl.read_scalars(atl.run.logdir)  # or a path
    atl.terminal.print_ok("done")

``atl.run`` is the current :class:`Writer` (None before ``init``), and
``atl.config`` its config. ``atl.Writer`` can also be used directly, without a
current run.

Everything is imported on first use, so ``import atalaya`` is instant and
``atl.Timer`` does not load tensorboardX.
"""

from importlib import import_module

# public name -> (module, attribute); attribute None means the module itself.
# A function must not take the name of a submodule (e.g. pull.pull): importing
# the submodule would replace one with the other on the package.
_EXPORTS = {
    "init": ("_session", "init"),
    "log": ("_session", "log"),
    "finish": ("_session", "finish"),
    "log_timings": ("_session", "log_timings"),
    "timer": ("timer", None),  # callable module: atl.timer("section")
    "Writer": ("writer", "Writer"),
    "Timer": ("timer", "Timer"),
    "format_time": ("timer", "format_time"),
    "terminal": ("terminal", None),  # the module; print_ok etc. forward to the shared instance
    "read_scalars": ("read", "read_scalars"),
    "read_info": ("read", "read_info"),
    "read_config": ("read", "read_config"),
    "read_timings": ("read", "read_timings"),
    "upload_pluto": ("upload", "upload_pluto"),
    "read": ("read", None),
    "video": ("video", None),
}

# looked up on every access, never cached: they change as runs start and finish
_DYNAMIC = ("run", "config")

__all__ = sorted(_EXPORTS) + list(_DYNAMIC) + ["__version__"]


def __getattr__(name):
    if name in _DYNAMIC:
        from . import _session

        run = _session.current()
        if name == "run":
            return run
        return run.config if run is not None else None
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("atalaya")
        except PackageNotFoundError:  # running from a source checkout
            value = "0+unknown"
    elif name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = module if attr is None else getattr(module, attr)
    else:
        raise AttributeError(f"module 'atalaya' has no attribute {name!r}")
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__():
    return __all__
