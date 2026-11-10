import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from atalaya import Writer
from atalaya.logging import CSVLogger, OutputCatcher, collect_environment, installed_packages
from atalaya.read import read_config, read_info, read_scalars, read_scalars_by_step

REPO = Path(__file__).resolve().parents[1]


def _writer(tmp_path, **kw):
    for key, value in dict(
        log_git_info=False,
        save_code=False,
        capture_exceptions=False,
        handle_sigterm=False,
        log_environment=False,
        save_requirements=False,
    ).items():
        kw.setdefault(key, value)
    return Writer(name="run", project="p", logdir=str(tmp_path), **kw)


def _run_script(tmp_path, body, timeout=60):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(body))
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "logs")],
        cwd=tmp_path,
        env={"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ------------------------------------------------------------ csv options ---
def test_csv_flushes_on_time_not_only_on_row_count(tmp_path):
    logger = CSVLogger(str(tmp_path), flush_interval=1000, flush_secs=0)
    logger.add("loss", 1.0, 0, walltime=time.time())
    rows = (tmp_path / "data.csv").read_text().splitlines()
    assert len(rows) == 2  # header + the row, without an explicit flush()


def test_csv_gzip_appends_members_and_reads_back(tmp_path):
    for start in (0, 2):  # two writers = a resumed run
        with _writer(tmp_path, save_as_csv=True, csv_compress=True) as w:
            for step in range(start, start + 2):
                w.log({"loss": float(step)}, step=step)
    path = tmp_path / "data.csv.gz"
    assert not (tmp_path / "data.csv").exists()
    lines = gzip.decompress(path.read_bytes()).decode().splitlines()
    assert lines[0].startswith("global_step") and sum(l.startswith("global_step") for l in lines) == 1
    assert read_scalars(tmp_path)["value"].tolist() == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------- reader ---
def test_read_scalars_long_and_wide(tmp_path):
    with _writer(tmp_path, save_as_csv=True) as w:
        for step in range(3):
            w.log({"loss": 1.0 / (step + 1), "acc": step / 10}, step=step)
    long = read_scalars(tmp_path)
    assert list(long.columns) == ["global_step", "tag", "value", "walltime", "relative_time"]
    assert len(long) == 6
    wide = read_scalars(tmp_path, wide=True)
    assert list(wide.index) == [0, 1, 2] and sorted(wide.columns) == ["acc", "loss"]
    assert wide.loc[2, "acc"] == pytest.approx(0.2)
    only = read_scalars(tmp_path, tags=["loss"])
    assert set(only["tag"]) == {"loss"}
    assert read_info(tmp_path)["status"] == "finished"


def test_reader_skips_repeated_headers_from_old_versions(tmp_path):
    (tmp_path / "data.csv").write_text(
        "global_step,tag,value,walltime,relative_time\n0,loss,1.0,10.0,0.0\n"
        "global_step,tag,value,walltime,relative_time\n1,loss,0.5,11.0,1.0\n"
    )
    by_step, walltimes = read_scalars_by_step(tmp_path)
    assert by_step == {0: {"loss": 1.0}, 1: {"loss": 0.5}}
    assert walltimes == {0: 10.0, 1: 11.0}


# ---------------------------------------------------------------- config ---
def test_config_saved_and_used_by_integrations(tmp_path, monkeypatch):
    calls = []

    class FakeWandb:
        class Settings:
            def __init__(self, **kw):
                pass

        @staticmethod
        def init(**kw):
            calls.append(kw)
            return type("Run", (), {"id": "x", "url": None, "offline": False, "finish": lambda self: None})()

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    with _writer(tmp_path, config={"lr": 0.1, "nested": {"a": 1}}) as w:
        w.with_wandb()
    assert read_config(tmp_path) == {"lr": 0.1, "nested": {"a": 1}}
    assert calls[0]["config"] == {"lr": 0.1, "nested": {"a": 1}}


# ----------------------------------------------------------- environment ---
def test_requirements_and_environment(tmp_path, monkeypatch):
    fake_torch = type(sys)("torch")
    fake_torch.__version__ = "9.9.9"
    fake_torch.version = type("v", (), {"cuda": "12.4"})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with _writer(tmp_path, log_environment=True, save_requirements=True):
        pass
    reqs = (tmp_path / "requirements.txt").read_text().splitlines()
    assert any(r.lower().startswith("numpy==") for r in reqs)
    env = read_info(tmp_path)["environment"]
    assert env["torch"]["version"] == "9.9.9" and env["torch"]["cuda"] == "12.4"


def test_no_gpus_recorded_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert collect_environment() == {}


def test_installed_packages_are_sorted_and_unique():
    pkgs = installed_packages()
    names = [p.split("==")[0].lower() for p in pkgs]
    assert names == sorted(names) and len(names) == len(set(names))


# --------------------------------------------------------------- sigterm ---
def test_sigterm_closes_writer_and_process_dies_by_signal(tmp_path):
    proc = _run_script(
        tmp_path,
        """
        import os, signal, sys
        from atalaya import Writer
        w = Writer(logdir=sys.argv[1], save_as_csv=True, log_git_info=False, save_code=False,
                   log_environment=False, save_requirements=False)
        for step in range(3):  # fewer rows than the CSV flush interval
            w.log({"loss": step}, step=step)
        os.kill(os.getpid(), signal.SIGTERM)
        print("not reached")
        """,
    )
    assert proc.returncode == -signal.SIGTERM, proc.stderr
    assert "not reached" not in proc.stdout
    info = read_info(tmp_path / "logs")
    assert info["status"] == "terminated" and info["last_step"] == 2
    assert len(read_scalars(tmp_path / "logs")) == 3


def test_sigterm_chains_to_the_programs_own_handler(tmp_path):
    proc = _run_script(
        tmp_path,
        """
        import os, signal, sys
        from atalaya import Writer

        def on_term(signum, frame):  # e.g. save a checkpoint, log once more, exit
            w.log({"final": 1.0}, step=99)
            sys.exit(0)

        signal.signal(signal.SIGTERM, on_term)
        w = Writer(logdir=sys.argv[1], save_as_csv=True, log_git_info=False, save_code=False,
                   log_environment=False, save_requirements=False)
        w.log({"loss": 0.5}, step=0)
        os.kill(os.getpid(), signal.SIGTERM)
        """,
    )
    assert proc.returncode == 0, proc.stderr
    info = read_info(tmp_path / "logs")
    assert info["status"] == "terminated" and info["last_step"] == 99
    assert set(read_scalars(tmp_path / "logs")["tag"]) == {"loss", "final"}


# -------------------------------------------------------- output catcher ---
def test_output_catcher_tees_and_keeps_stream_api(tmp_path, capsys):
    catcher = OutputCatcher(tmp_path, with_time=True, catch_stdout=True, catch_errors=True)
    try:
        print("hello", "world")
        sys.stdout.write("partial ")
        sys.stdout.write("line\n")
        print("oops", file=sys.stderr)
        assert sys.stdout.encoding
        assert isinstance(sys.stdout.isatty(), bool)
    finally:
        catcher.stop()
    out, err = capsys.readouterr()
    assert "hello world" in out and "oops" not in out
    assert "oops" in err  # stderr stays stderr
    lines = (tmp_path / "log.txt").read_text().splitlines()
    assert [l.split("] ", 1)[1] for l in lines] == ["hello world", "partial line", "oops"]
    assert all(l.startswith("[") for l in lines)  # one timestamp per line
    print("after stop")
    assert "after stop" not in (tmp_path / "log.txt").read_text()


def test_output_catcher_fileno_works_for_subprocess(tmp_path):
    proc = _run_script(
        tmp_path,
        """
        import subprocess, sys
        from atalaya.logging import OutputCatcher
        c = OutputCatcher(sys.argv[1])
        subprocess.run(["echo", "from child"], stdout=sys.stdout, check=True)
        print("from parent")
        c.stop()
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert "from child" in proc.stdout and "from parent" in proc.stdout


# ----------------------------------------------------------------- async ---
def test_async_writes_keep_order_and_call_time(tmp_path):
    with _writer(tmp_path, save_as_csv=True, async_writes=True) as w:
        before = time.time()
        for step in range(50):
            w.log({"loss": float(step)}, step=step)
        after = time.time()
        w.wait()
        assert w._worker_ident is not None
    df = read_scalars(tmp_path)
    assert df["global_step"].tolist() == list(range(50))
    assert df["walltime"].between(before, after).all()  # time of the call, not of the write
    assert read_info(tmp_path)["last_step"] == 49


def test_async_error_surfaces_on_next_call(tmp_path):
    w = _writer(tmp_path, async_writes=True)
    w.add_histogram("bad", object(), 0)  # fails on the writer thread
    w._executor.submit(lambda: None).result()
    with pytest.raises(RuntimeError, match="earlier asynchronous write failed"):
        w.log({"loss": 1.0}, step=1)
    w.close()


# ----------------------------------------------------------------- video ---
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_fast_video_writes_a_gif_summary(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    vid = np.random.rand(3, 4, 3, 16, 16).astype(np.float32)  # N=3 (padded to 4), NTCHW
    with _writer(tmp_path) as w:
        w.add_video("rollout", vid, global_step=5, fps=8)
    f = next(tmp_path.glob("events.out.tfevents.*"))
    ea = EventAccumulator(str(f), size_guidance={"images": 0})
    ea.Reload()
    (event,) = ea.Images("rollout")
    assert event.step == 5
    assert event.encoded_image_string[:6] == b"GIF89a"
    assert (event.width, event.height) == (16 * 4, 16)  # 1 x 4 grid


def test_video_layouts():
    from atalaya.video import make_grid, to_nthwc

    v = np.zeros((2, 5, 3, 8, 6), dtype=np.uint8)  # NTCHW
    assert to_nthwc(v, "NTCHW").shape == (2, 5, 8, 6, 3)
    assert to_nthwc(v[0], "TCHW").shape == (1, 5, 8, 6, 3)
    assert make_grid(to_nthwc(v, "NTCHW")).shape == (5, 8, 12, 3)
    with pytest.raises(ValueError):
        to_nthwc(v, "NTHW")


# ------------------------------------------------------------------ pull ---
@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_pull_copies_only_small_run_files(tmp_path):
    src = tmp_path / "cluster" / "logs"
    run = src / "exp1" / "seed0"
    (run / "ckpt").mkdir(parents=True)
    for name in ("data.csv", "info.json", "config.json", "git.diff", "events.out.tfevents.1.host"):
        (run / name).write_text(name)
    (run / "ckpt" / "model.pt").write_text("big")
    (src / "exp1" / "scores.json").write_text("{}")

    from atalaya.cli import main

    dest = tmp_path / "local"
    assert main(["pull", str(src), str(dest)]) == 0
    got = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert got == [f"exp1/seed0/{n}" for n in ("config.json", "data.csv", "git.diff", "info.json")]
    assert not (dest / "exp1" / "seed0" / "ckpt").exists()  # empty dirs pruned

    assert main(["pull", str(src), str(dest), "--with-events", "--include", "scores.json"]) == 0
    assert (dest / "exp1" / "seed0" / "events.out.tfevents.1.host").exists()
    assert (dest / "exp1" / "scores.json").exists()


# ----------------------------------------------------- tensorboardX guard ---
def test_event_file_found_through_tensorboardx_internals(tmp_path):
    """Fails if a tensorboardX release moves the attribute the writer reads.

    The writer falls back to globbing, so users are not affected; this test is
    here so CI notices and the fallback does not become the silent norm.
    """
    with _writer(tmp_path) as w:
        path = Path(w.file_writer.event_writer._ev_writer._file_name)
        assert w._event_file_path == path and path.exists()
