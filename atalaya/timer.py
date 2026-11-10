"""Wall-clock timing of named code sections.

    from atalaya.timer import Timer

    timer = Timer()                          # optionally Timer(cuda_sync=True)

    with timer("train/update_actor"):        # named section, as a context manager
        ...

    @timer("rollout")                        # same section object, as a decorator
    def rollout(...): ...

    timer.start("eval"); ...; timer.stop("eval")   # when a `with` does not fit

    with Timer("training") as t:             # a single block
        train()
    t.report()

Statistics come back as a dict (``stats()``), flat for logging (``flat()``,
``Writer.log_timings``) or cumulative for a file (``summary()``, written to
timings.json by ``Writer.attach_timer``). Memory is constant: each section
keeps counters, not a list of every duration.

GPU work runs asynchronously, so without synchronisation a section's time can
include work launched by an earlier section, or miss its own. ``cuda_sync=True``
calls ``torch.cuda.synchronize()`` when a section starts and stops; it makes
the split between sections trustworthy and slows training a little.
"""

import functools
import sys
import threading
import time

from .terminal import terminal


def format_time(seconds, time_format="%d-%H:%M:%S.%f", floating_precision=3):
    """Format seconds with %d (days), %H, %M, %S and %f (milliseconds).

    The largest unit present absorbs everything above it; a format with a
    single placeholder prints that unit with decimals (e.g. "%S" -> "75.250").
    """
    time_components = {
        # placeholder, units per next unit, zero-padded width
        "milliseconds": ("%f", 1000, 3),
        "seconds": ("%S", 60, 2),
        "minutes": ("%M", 60, 2),
        "hours": ("%H", 24, 2),
        "days": ("%d", 1, 1),
    }
    time_totals = {
        "milliseconds": seconds * 1000,
        "seconds": seconds,
        "minutes": seconds / 60,
        "hours": seconds / 3600,
        "days": seconds / (24 * 3600),
    }
    total_placeholders = time_format.count("%")
    for time_unit, (placeholder, unit_limit, precision) in time_components.items():
        remaining_placeholders = time_format.count("%")
        if remaining_placeholders == 0:
            break
        if placeholder in time_format:
            total_value = time_totals[time_unit]
            if total_placeholders == 1:
                formatted_value = f"{total_value:.0{floating_precision}f}"
            elif remaining_placeholders == 1:
                formatted_value = f"{int(total_value):0{precision}d}"
            else:
                formatted_value = f"{int(total_value % unit_limit):0{precision}d}"
            time_format = time_format.replace(placeholder, formatted_value)
    if time_format.count("%") != 0:
        raise ValueError("The given time format is not valid.")
    return time_format


class _Stats:
    """Counters for one section: cumulative, and since the last interval reset."""

    __slots__ = ("count", "total", "min", "max", "last", "i_count", "i_total", "i_min", "i_max")

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0
        self.last = 0.0
        self.reset_interval()

    def reset_interval(self):
        self.i_count = 0
        self.i_total = 0.0
        self.i_min = float("inf")
        self.i_max = 0.0

    def add(self, seconds):
        self.count += 1
        self.total += seconds
        self.min = min(self.min, seconds)
        self.max = max(self.max, seconds)
        self.last = seconds
        self.i_count += 1
        self.i_total += seconds
        self.i_min = min(self.i_min, seconds)
        self.i_max = max(self.i_max, seconds)


class _Scope:
    """``timer("name")``: a context manager and a decorator for one section.

    Each entry keeps its own start time, so the same scope can be nested,
    recursive or used from several threads at once.
    """

    def __init__(self, timer, name):
        self._timer = timer
        self.name = name
        self._starts = threading.local()

    def __enter__(self):
        if not self._timer.disabled:
            stack = getattr(self._starts, "stack", None)
            if stack is None:
                stack = self._starts.stack = []
            stack.append(self._timer._now())
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._timer.disabled:
            start = self._starts.stack.pop()
            # recorded even when the block raised: that time was spent too
            self._timer.record(self.name, self._timer._now() - start)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class Timer:
    """Named sections timed with ``time.perf_counter``.

    ``name`` is only needed to use the timer itself as a context manager
    (``with Timer("training"):``), which times a section of that name.
    """

    def __init__(
        self,
        name: str | None = None,
        cuda_sync: bool = False,
        disabled: bool = False,
        color: str = "magenta",
        clock=time.perf_counter,
    ):
        self.name = name
        self.cuda_sync = cuda_sync
        self.disabled = disabled
        self.color = color
        self._clock = clock
        self._lock = threading.Lock()
        self._sections: dict[str, _Stats] = {}
        self._scopes: dict[str, _Scope] = {}
        self._running: dict[str, float] = {}
        self._cuda_ready = None
        self._created = clock()
        self._interval_start = self._created
        self._interval_steps = None

    # ------------------------------------------------------------- measuring
    def __call__(self, name: str) -> _Scope:
        scope = self._scopes.get(name)
        if scope is None:
            scope = self._scopes[name] = _Scope(self, name)
        return scope

    def __enter__(self):
        if self.name is None:
            raise TypeError("Timer() without a name: use `with timer('section'):` or Timer('name')")
        self.start(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.name in self._running:
            self.stop(self.name)
        return False

    def start(self, name: str) -> None:
        if self.disabled:
            return
        if name in self._running:
            raise RuntimeError(f"section '{name}' is already running")
        self._running[name] = self._now()

    def stop(self, name: str) -> float:
        """Stop ``name`` and return its duration in seconds."""
        if self.disabled:
            return 0.0
        if name not in self._running:
            raise RuntimeError(f"section '{name}' is not running")
        seconds = self._now() - self._running.pop(name)
        self.record(name, seconds)
        return seconds

    def running(self, name: str) -> bool:
        return name in self._running

    def record(self, name: str, seconds: float) -> None:
        """Add a duration measured elsewhere."""
        if self.disabled:
            return
        with self._lock:
            stats = self._sections.get(name)
            if stats is None:
                stats = self._sections[name] = _Stats()
            stats.add(seconds)

    def _now(self):
        if self.cuda_sync:
            self._synchronize()
        return self._clock()

    def _synchronize(self):
        torch = sys.modules.get("torch")
        if torch is None:
            return
        if self._cuda_ready is None:
            self._cuda_ready = bool(torch.cuda.is_available())
        # is_initialized: never create a CUDA context just to time something
        if self._cuda_ready and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    # ------------------------------------------------------------ reporting
    def stats(self, steps: int | None = None, reset_interval: bool = True) -> dict:
        """Per-section statistics, since the last interval reset and overall.

        Per section, for the interval: ``count``, ``sum``, ``mean``, ``min``,
        ``max``, ``frac`` (share of the interval's wall time) and, with
        ``steps``, ``rate`` (steps per second of time spent in the section).
        Overall: ``total_count``, ``total``, ``total_mean``, ``total_frac``
        and ``total_rate``. Top level: ``wall``, ``interval_wall`` and, with
        ``steps``, ``steps_per_sec`` and ``total_steps_per_sec``.

        ``steps`` is a running count (agent steps, samples, ...); rates use the
        difference since the previous call.
        """
        now = self._clock()
        wall = now - self._created
        interval_wall = now - self._interval_start
        d_steps = None
        if steps is not None:
            d_steps = steps - (self._interval_steps or 0)
        out = {"wall": wall, "interval_wall": interval_wall, "sections": {}}
        if steps is not None:
            out["steps_per_sec"] = d_steps / interval_wall if interval_wall > 0 else None
            out["total_steps_per_sec"] = steps / wall if wall > 0 else None
        with self._lock:
            for name, s in self._sections.items():
                entry = {
                    "count": s.i_count,
                    "sum": s.i_total,
                    "mean": s.i_total / s.i_count if s.i_count else None,
                    "min": s.i_min if s.i_count else None,
                    "max": s.i_max if s.i_count else None,
                    "frac": s.i_total / interval_wall if interval_wall > 0 else None,
                    "total_count": s.count,
                    "total": s.total,
                    "total_mean": s.total / s.count if s.count else None,
                    "total_frac": s.total / wall if wall > 0 else None,
                }
                if steps is not None:
                    entry["rate"] = d_steps / s.i_total if s.i_total > 0 else None
                    entry["total_rate"] = steps / s.total if s.total > 0 else None
                out["sections"][name] = entry
                if reset_interval:
                    s.reset_interval()
        if reset_interval:
            self._interval_start = now
            if steps is not None:
                self._interval_steps = steps
        return out

    def flat(
        self,
        steps: int | None = None,
        columns=("frac", "mean"),
        prefix: str | None = None,
        reset_interval: bool = True,
    ) -> dict:
        """``{"<prefix>/<section>/<column>": value}`` for logging; None values dropped."""
        stats = self.stats(steps=steps, reset_interval=reset_interval)
        pre = f"{prefix}/" if prefix else ""
        out = {}
        for name, entry in stats["sections"].items():
            for col in columns:
                value = entry.get(col)
                if value is not None:
                    out[f"{pre}{name}/{col}"] = value
        if "steps_per_sec" in stats and stats["steps_per_sec"] is not None:
            out[f"{pre}steps_per_sec"] = stats["steps_per_sec"]
        return out

    def summary(self, steps: int | None = None) -> dict:
        """Cumulative statistics, JSON-ready; does not reset the interval."""
        now = self._clock()
        wall = now - self._created
        out = {"wall": wall, "sections": {}}
        if steps is not None:
            out["steps"] = steps
            out["steps_per_sec"] = steps / wall if wall > 0 else None
        with self._lock:
            for name, s in self._sections.items():
                entry = {
                    "count": s.count,
                    "total": s.total,
                    "mean": s.total / s.count if s.count else None,
                    "min": s.min if s.count else None,
                    "max": s.max if s.count else None,
                    "frac": s.total / wall if wall > 0 else None,
                }
                if steps is not None:
                    entry["rate"] = steps / s.total if s.total > 0 else None
                out["sections"][name] = entry
        if self.cuda_sync:
            out["cuda_sync"] = True
        return out

    def report(self, print_it: bool = True) -> str:
        """Table of cumulative statistics, printed through ``terminal``."""
        summary = self.summary()
        title = f"[{self.name}] " if self.name else ""
        lines = [f"{title}wall {format_time(summary['wall'])}"]
        sections = summary["sections"]
        if sections:
            width = max(len(n) for n in sections)
            lines.append(f"{'section':<{width}}  {'count':>7}  {'total':>12}  {'mean':>9}  {'max':>9}  {'frac':>6}")
            for name, e in sorted(sections.items(), key=lambda kv: -kv[1]["total"]):
                lines.append(
                    f"{name:<{width}}  {e['count']:>7d}  {format_time(e['total']):>12}  "
                    f"{e['mean']:>8.4f}s  {e['max']:>8.4f}s  {100 * (e['frac'] or 0):>5.1f}%"
                )
        text = "\n".join(lines)
        if print_it:
            terminal.print_message(text, color=self.color)
        return text

    def reset(self) -> None:
        """Forget every measurement and restart the wall clock."""
        with self._lock:
            self._sections.clear()
            self._running.clear()
            self._created = self._interval_start = self._clock()
            self._interval_steps = None


# `atl.timer("name")` times a section of the current run (atl.init). The
# function cannot simply be called `timer` in the package: it would clash with
# this module, which `from atalaya.timer import Timer` needs. So the module
# itself is callable.
class _CallableModule(sys.modules[__name__].__class__):
    def __call__(self, name: str):
        from ._session import timer

        return timer(name)


sys.modules[__name__].__class__ = _CallableModule
