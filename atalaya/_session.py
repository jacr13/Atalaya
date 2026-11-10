"""The process-wide current run behind ``atl.init`` / ``atl.log`` / ``atl.finish``.

Python imports a module once per process, so every module that does
``import atalaya as atl`` sees the same state here: ``atl.log`` in a data
module, a model module and the training script all write to the run that
``atl.init`` started. Separate processes (DataLoader workers, multiprocessing)
have their own state and no run.
"""

import threading
import warnings
from datetime import datetime
from pathlib import Path

_lock = threading.RLock()
_run = None


def current():
    return _run


def _require():
    run = _run
    if run is None:
        raise RuntimeError("no active run: call atl.init() first")
    return run


def _clear(writer):
    """Called by Writer.close(), so a closed run stops being the current one."""
    global _run
    with _lock:
        if _run is writer:
            _run = None


def init(
    name: str | None = None,
    project: str | None = None,
    logdir: str = "logs",
    config: dict | None = None,
    save_as_csv: bool = True,
    **writer_options,
):
    """Start a run and make it the current one; returns its :class:`Writer`.

    Each run gets its own directory, ``<logdir>/<name>``. ``name`` defaults to
    ``<script>_<YYYYmmdd_HHMMSS>``, ``project`` to the script's name. Any other
    :class:`Writer` option can be passed (``csv_compress``, ``async_writes``,
    ``log_git_info``, ...). An active run is finished first.
    """
    global _run
    from .writer import Writer

    script = _script_name()
    if name is None:
        name = f"{script}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if project is None:
        project = script
    with _lock:
        if _run is not None:
            warnings.warn(f"atl.init(): finishing the active run '{_run.name}' first")
            _run.close()
        writer_options.setdefault("add_name_to_logdir", True)
        _run = Writer(
            name=name,
            project=project,
            logdir=logdir,
            config=config,
            save_as_csv=save_as_csv,
            **writer_options,
        )
        _run._auto_step = 0
        return _run


def log(data: dict, step: int | None = None, prefix: str | None = None):
    """Log scalars to the current run.

    Without ``step``, each call uses the next step (0, 1, 2, ...), like
    ``wandb.log``. With ``step``, that step is used and the next call without
    one continues from step + 1.
    """
    run = _require()
    with _lock:
        if step is None:
            step = run._auto_step
        run._auto_step = max(run._auto_step, step + 1)
    run.log(data, step=step, prefix=prefix)


def finish():
    """Close the current run (writes info.json, timings.json, flushes data.csv)."""
    run = _run
    if run is not None:
        run.close()


def log_timings(step: int | None = None, steps: int | None = None, columns=("frac", "mean")):
    """Log the current run's timer (see :meth:`Writer.log_timings`)."""
    run = _require()
    if step is None:
        step = max(run._auto_step - 1, 0)
    run.log_timings(_run_timer(run), global_step=step, steps=steps, columns=columns)


def _run_timer(run):
    timer = getattr(run, "_session_timer", None)
    if timer is None:
        timer = run._session_timer = run.create_timer()
    return timer


class _DeferredScope:
    """``atl.timer("name")``: resolves the current run when entered.

    So it can decorate functions at import time, before ``atl.init()``. With
    no active run it times nothing.
    """

    def __init__(self, name):
        self.name = name
        self._entered = threading.local()

    def __enter__(self):
        run = _run
        stack = getattr(self._entered, "stack", None)
        if stack is None:
            stack = self._entered.stack = []
        scope = _run_timer(run)(self.name) if run is not None else None
        if scope is not None:
            scope.__enter__()
        stack.append(scope)
        return self

    def __exit__(self, exc_type, exc, tb):
        scope = self._entered.stack.pop()
        if scope is not None:
            scope.__exit__(exc_type, exc, tb)
        return False

    def __call__(self, func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


_scopes: dict = {}


def timer(name: str) -> _DeferredScope:
    """Time a section in the current run: ``with atl.timer("x"):`` or ``@atl.timer("x")``."""
    scope = _scopes.get(name)
    if scope is None:
        scope = _scopes[name] = _DeferredScope(name)
    return scope


def _script_name():
    import sys

    argv0 = sys.argv[0] if sys.argv else ""
    stem = Path(argv0).stem if argv0 and argv0 not in ("-c", "-m") else ""
    return stem or "run"
