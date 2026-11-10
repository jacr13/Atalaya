import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import atalaya as atl
from atalaya.read import read_info, read_scalars, read_timings

REPO = Path(__file__).resolve().parents[1]
QUIET = dict(
    log_git_info=False, save_code=False, capture_exceptions=False,
    handle_sigterm=False, log_environment=False, save_requirements=False,
)


@pytest.fixture(autouse=True)
def no_run_left_behind():
    yield
    atl.finish()


def test_init_log_finish(tmp_path):
    run = atl.init(name="r1", project="p", logdir=str(tmp_path), config={"lr": 0.1}, **QUIET)
    assert atl.run is run and atl.config == {"lr": 0.1}
    assert run.logdir == tmp_path / "r1"  # one directory per run
    atl.log({"loss": 1.0})
    atl.log({"loss": 0.5}, prefix="train")
    atl.finish()
    assert atl.run is None and atl.config is None
    df = read_scalars(tmp_path / "r1")
    assert df[["global_step", "tag"]].values.tolist() == [[0, "loss"], [1, "train/loss"]]
    assert read_info(tmp_path / "r1")["status"] == "finished"


def test_auto_step_like_wandb(tmp_path):
    atl.init(name="r", logdir=str(tmp_path), **QUIET)
    atl.log({"a": 1})            # step 0
    atl.log({"a": 2})            # step 1
    atl.log({"a": 3}, step=10)   # explicit
    atl.log({"a": 4})            # continues at 11
    atl.log({"a": 5}, step=5)    # going back does not move the counter back
    atl.log({"a": 6})            # 12
    atl.finish()
    assert read_scalars(tmp_path / "r")["global_step"].tolist() == [0, 1, 10, 11, 5, 12]


def test_log_without_init_says_what_to_do():
    with pytest.raises(RuntimeError, match=r"atl\.init\(\)"):
        atl.log({"loss": 1.0})


def test_second_init_finishes_the_first(tmp_path):
    first = atl.init(name="a", logdir=str(tmp_path), **QUIET)
    with pytest.warns(UserWarning, match="finishing the active run 'a'"):
        second = atl.init(name="b", logdir=str(tmp_path), **QUIET)
    assert first._closed and atl.run is second
    assert read_info(tmp_path / "a")["status"] == "finished"


def test_init_as_context_manager_clears_the_run(tmp_path):
    with atl.init(name="r", logdir=str(tmp_path), **QUIET) as run:
        assert atl.run is run
        atl.log({"x": 1})
    assert atl.run is None


def test_closing_the_writer_directly_also_clears_it(tmp_path):
    run = atl.init(name="r", logdir=str(tmp_path), **QUIET)
    run.close()
    assert atl.run is None


def test_timer_decorator_defined_before_init(tmp_path):
    @atl.timer("rollout")  # at import time, no run yet
    def rollout():
        return 42

    assert rollout() == 42  # no run: runs, times nothing
    atl.init(name="r", logdir=str(tmp_path), **QUIET)
    rollout()
    with atl.timer("update"):
        pass
    atl.log({"loss": 1.0})
    atl.log_timings(columns=("frac",))
    atl.finish()
    sections = read_timings(tmp_path / "r")["sections"]
    assert sections["rollout"]["count"] == 1 and sections["update"]["count"] == 1
    assert "timings/rollout/frac" in set(read_scalars(tmp_path / "r")["tag"])


def test_timer_module_is_still_a_module():
    from atalaya.timer import Timer, format_time  # noqa: F401
    import atalaya.timer

    assert atl.timer is atalaya.timer
    assert type(atl.timer("x")).__name__ == "_DeferredScope"


def test_modules_of_one_process_log_to_the_same_run(tmp_path):
    (tmp_path / "data.py").write_text(textwrap.dedent("""
        import atalaya as atl

        def load():
            atl.log({"data/n_samples": 128}, step=0)
    """))
    (tmp_path / "model.py").write_text(textwrap.dedent("""
        import atalaya as atl

        @atl.timer("model/forward")
        def forward(step):
            atl.log({"model/loss": 1.0 / (step + 1)}, step=step)
    """))
    (tmp_path / "train.py").write_text(textwrap.dedent("""
        import sys
        import atalaya as atl
        import data, model

        atl.init(name="run", logdir=sys.argv[1], log_git_info=False, save_code=False,
                 log_environment=False, save_requirements=False)
        data.load()
        for step in range(3):
            model.forward(step)
        atl.finish()
    """))
    proc = subprocess.run(
        [sys.executable, "train.py", str(tmp_path / "logs")],
        cwd=tmp_path, env={"PYTHONPATH": str(REPO)}, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    run_dir = tmp_path / "logs" / "run"
    assert set(read_scalars(run_dir)["tag"]) == {"data/n_samples", "model/loss"}
    assert read_timings(run_dir)["sections"]["model/forward"]["count"] == 3
    assert read_info(run_dir)["project"] == "train"  # default project: the script
