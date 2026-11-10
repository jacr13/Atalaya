import json
import sys
import threading

import pytest

from atalaya import Writer
from atalaya.read import read_scalars, read_timings
from atalaya.timer import Timer, format_time


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


# -------------------------------------------- the five ways to use it ---
def test_named_section_as_context_manager(clock):
    timer = Timer(clock=clock)
    with timer("train/update_actor"):
        clock.advance(2.0)
    s = timer.summary()["sections"]["train/update_actor"]
    assert s["count"] == 1 and s["total"] == 2.0


def test_same_section_object_as_decorator(clock):
    timer = Timer(clock=clock)

    @timer("rollout")
    def rollout(n):
        """docstring kept"""
        clock.advance(n)
        return n * 10

    assert rollout(1.0) == 10 and rollout(3.0) == 30
    assert rollout.__name__ == "rollout" and rollout.__doc__ == "docstring kept"
    s = timer.summary()["sections"]["rollout"]
    assert (s["count"], s["total"], s["min"], s["max"]) == (2, 4.0, 1.0, 3.0)
    assert timer("rollout") is timer("rollout")  # one scope object per name


def test_explicit_start_stop(clock):
    timer = Timer(clock=clock)
    timer.start("eval")
    assert timer.running("eval")
    clock.advance(0.5)
    assert timer.stop("eval") == 0.5
    assert not timer.running("eval")
    with pytest.raises(RuntimeError, match="not running"):
        timer.stop("eval")
    timer.start("eval")
    with pytest.raises(RuntimeError, match="already running"):
        timer.start("eval")


def test_single_block_timer_and_report(clock, capsys):
    with Timer("training", clock=clock) as t:
        clock.advance(75.25)
    text = t.report()
    assert "[training]" in text and "training" in capsys.readouterr().out
    assert t.summary()["sections"]["training"]["total"] == 75.25
    with pytest.raises(TypeError):
        with Timer():
            pass


def test_all_forms_together_in_one_timer(clock):
    timer = Timer(clock=clock)

    @timer("rollout")
    def rollout():
        clock.advance(1.0)

    with timer("epoch"):
        rollout()
        timer.start("eval")
        clock.advance(2.0)
        timer.stop("eval")
    sections = timer.summary()["sections"]
    assert sections["epoch"]["total"] == 3.0
    assert sections["rollout"]["total"] == 1.0 and sections["eval"]["total"] == 2.0


# ---------------------------------------------------------- correctness ---
def test_section_that_raises_is_still_stopped_and_recorded(clock):
    timer = Timer(clock=clock)

    @timer("flaky")
    def flaky():
        clock.advance(1.0)
        raise ValueError("boom")

    for _ in range(2):  # the second call must not hit "already running"
        with pytest.raises(ValueError):
            flaky()
    assert timer.summary()["sections"]["flaky"]["count"] == 2


def test_recursion_and_nesting_of_the_same_section(clock):
    timer = Timer(clock=clock)

    @timer("fib")
    def f(n):
        clock.advance(1.0)
        return n if n < 2 else f(n - 1) + f(n - 2)

    f(4)
    assert timer.summary()["sections"]["fib"]["count"] == 9


def test_threads_time_their_own_entries():
    timer = Timer()

    def work():
        for _ in range(200):
            with timer("work"):
                pass

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert timer.summary()["sections"]["work"]["count"] == 800


def test_interval_vs_cumulative_stats_and_rates(clock):
    timer = Timer(clock=clock)
    with timer("sim"):
        clock.advance(3.0)
    clock.advance(1.0)  # untimed
    first = timer.stats(steps=400)
    sim = first["sections"]["sim"]
    assert first["interval_wall"] == 4.0 and sim["frac"] == 0.75
    assert sim["rate"] == pytest.approx(400 / 3.0)
    assert first["steps_per_sec"] == 100.0

    with timer("sim"):
        clock.advance(1.0)
    clock.advance(1.0)
    second = timer.stats(steps=600)
    sim = second["sections"]["sim"]
    # interval: only what happened since the previous stats() call
    assert (sim["count"], sim["sum"], sim["frac"]) == (1, 1.0, 0.5)
    assert sim["rate"] == 200.0 and second["steps_per_sec"] == 100.0
    # cumulative
    assert (sim["total_count"], sim["total"]) == (2, 4.0)
    assert sim["total_frac"] == pytest.approx(4.0 / 6.0)
    assert sim["total_rate"] == 150.0


def test_flat_for_logging(clock):
    timer = Timer(clock=clock)
    with timer("a"):
        clock.advance(1.0)
    clock.advance(1.0)
    flat = timer.flat(steps=10, columns=("frac", "mean"), prefix="timings")
    assert flat == {"timings/a/frac": 0.5, "timings/a/mean": 1.0, "timings/steps_per_sec": 5.0}
    # an interval with no call: None columns (mean) are dropped, not logged as NaN
    clock.advance(1.0)
    assert timer.flat(columns=("frac", "mean")) == {"a/frac": 0.0}


def test_disabled_timer_does_nothing(clock):
    timer = Timer(disabled=True, clock=clock)

    @timer("x")
    def f():
        return 1

    with timer("y"):
        pass
    timer.start("z")
    assert timer.stop("z") == 0.0 and f() == 1
    assert timer.summary()["sections"] == {}


def test_cuda_sync_only_when_cuda_is_initialised(monkeypatch):
    calls = []

    class Cuda:
        initialized = False

        @staticmethod
        def is_available():
            return True

        @classmethod
        def is_initialized(cls):
            return cls.initialized

        @staticmethod
        def synchronize():
            calls.append(1)

    fake_torch = type(sys)("torch")
    fake_torch.cuda = Cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    timer = Timer(cuda_sync=True)
    with timer("a"):
        pass
    assert calls == []  # no CUDA context yet: never create one
    Cuda.initialized = True
    with timer("a"):
        pass
    assert len(calls) == 2  # at start and at stop
    with Timer()("b"):
        pass
    assert len(calls) == 2  # cuda_sync off by default


def test_reset(clock):
    timer = Timer(clock=clock)
    with timer("a"):
        clock.advance(1.0)
    timer.reset()
    assert timer.summary()["sections"] == {} and timer.summary()["wall"] == 0.0


def test_format_time():
    assert format_time(75.25, "%S") == "75.250"
    assert format_time(3725.5, "%H:%M:%S") == "01:02:05"
    assert format_time(90061.25) == "1-01:01:01.250"  # default %d-%H:%M:%S.%f


# --------------------------------------------------------------- writer ---
def _writer(tmp_path, **kw):
    for key, value in dict(
        log_git_info=False, save_code=False, capture_exceptions=False,
        handle_sigterm=False, log_environment=False, save_requirements=False,
    ).items():
        kw.setdefault(key, value)
    return Writer(name="run", project="p", logdir=str(tmp_path), **kw)


def test_writer_logs_timings_and_writes_json_at_close(tmp_path):
    with _writer(tmp_path, save_as_csv=True) as w:
        timer = w.attach_timer(Timer())
        for step in range(3):
            with timer("update"):
                pass
            w.log({"loss": 1.0}, step=step)
            w.log_timings(timer, global_step=step, columns=("frac",))
    tags = set(read_scalars(tmp_path)["tag"])
    assert {"loss", "timings/update/frac", "timings/steps_per_sec"} <= tags
    timings = read_timings(tmp_path)
    assert timings["sections"]["update"]["count"] == 3
    assert timings["steps"] == 2  # last logged step
    assert json.loads((tmp_path / "timings.json").read_text()) == timings


def test_create_timer_attaches_a_new_timer(tmp_path):
    with _writer(tmp_path) as w:
        timer = w.create_timer(cuda_sync=False)
        other = w.create_timer(name="eval_timings.json")
        assert isinstance(timer, Timer) and timer is not other
        with timer("rollout"):
            pass
        with other("eval"):
            pass
    assert "rollout" in read_timings(tmp_path)["sections"]
    assert "eval" in read_timings(tmp_path, name="eval_timings.json")["sections"]
