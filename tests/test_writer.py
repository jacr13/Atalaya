import csv
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from atalaya import Writer
from atalaya.logging import CSVLogger, GitInfo


def _events(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(Path(logdir).glob("events.out.tfevents.*"))
    out = {}
    for f in files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            out.setdefault(tag, []).extend(e.value for e in ea.Scalars(tag))
    return out


def _csv_rows(logdir):
    with (Path(logdir) / "data.csv").open() as f:
        return list(csv.reader(f))


def _info(logdir):
    return json.loads((Path(logdir) / "info.json").read_text())


def _writer(tmp_path, **kw):
    kw.setdefault("log_git_info", False)
    kw.setdefault("save_code", False)
    kw.setdefault("capture_exceptions", False)
    kw.setdefault("handle_sigterm", False)
    kw.setdefault("log_environment", False)
    kw.setdefault("save_requirements", False)
    return Writer(name="run", project="p", logdir=str(tmp_path), **kw)


# ----------------------------------------------------------------- csv ---
def test_csv_header_written_once_on_resume(tmp_path):
    for start in (0, 3):
        logger = CSVLogger(str(tmp_path), flush_interval=2)
        for step in range(start, start + 3):
            logger.add("loss", step, step, walltime=float(step))
        logger.flush()
    rows = list(csv.reader((tmp_path / "data.csv").open()))
    assert rows[0][0] == "global_step"
    assert sum(r[0] == "global_step" for r in rows) == 1
    assert [int(r[0]) for r in rows[1:]] == list(range(6))


def test_add_scalar_also_goes_to_csv(tmp_path):
    with _writer(tmp_path, save_as_csv=True) as w:
        w.add_scalar("direct", 1.5, 0)
        w.log_scalars({"a": 1, "b": 2}, 1, prefix="train")
    tags = [r[1] for r in _csv_rows(tmp_path)[1:]]
    assert tags == ["direct", "train/a", "train/b"]


def test_add_scalars_keeps_tensorboardx_meaning_for_str_tag(tmp_path):
    with _writer(tmp_path, save_as_csv=True) as w:
        w.add_scalars({"x": 1.0}, global_step=0)  # atalaya meaning
        w.add_scalars("loss", {"train": 1.0, "val": 2.0}, 0)  # tensorboardX meaning
    assert [r[1] for r in _csv_rows(tmp_path)[1:]] == ["x"]
    # tensorboardX writes main-tag scalars to sub-directories
    assert sorted(p.name for p in (tmp_path / "loss").iterdir()) == ["train", "val"]


# -------------------------------------------------------------- closing ---
def test_close_writes_status_and_last_step(tmp_path):
    with _writer(tmp_path) as w:
        for step in range(5):
            w.log({"loss": 1.0 / (step + 1)}, step=step)
    info = _info(tmp_path)
    assert info["status"] == "finished"
    assert info["last_step"] == 4
    assert info["duration_seconds"] >= 0
    assert len(_events(tmp_path)["loss"]) == 5


def test_context_manager_records_failure(tmp_path):
    with pytest.raises(ValueError):
        with _writer(tmp_path) as w:
            w.log({"loss": 1.0}, step=0)
            raise ValueError("boom")
    info = _info(tmp_path)
    assert info["status"] == "failed"
    assert info["error"]["type"] == "ValueError"
    assert "boom" in info["error"]["traceback"]


def test_close_is_idempotent(tmp_path):
    w = _writer(tmp_path)
    w.close()
    w.close()
    assert _info(tmp_path)["status"] == "finished"


def _run_script(tmp_path, body):
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(body))
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "logs")],
        cwd=tmp_path,
        env={"PYTHONPATH": str(repo_root), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


def test_unclosed_writer_is_closed_at_exit(tmp_path):
    proc = _run_script(
        tmp_path,
        """
        import sys
        from atalaya import Writer
        w = Writer(logdir=sys.argv[1], save_as_csv=True, log_git_info=False, save_code=False)
        for step in range(3):  # fewer rows than the CSV flush interval
            w.log({"loss": step}, step=step)
        """,
    )
    assert proc.returncode == 0, proc.stderr
    logdir = tmp_path / "logs"
    assert _info(logdir)["status"] == "exited_without_close"
    assert len(_csv_rows(logdir)) == 1 + 3


def test_uncaught_exception_is_recorded(tmp_path):
    proc = _run_script(
        tmp_path,
        """
        import sys
        from atalaya import Writer
        w = Writer(logdir=sys.argv[1], log_git_info=False, save_code=False)
        w.log({"loss": 1.0}, step=7)
        raise RuntimeError("crashed mid-training")
        """,
    )
    assert proc.returncode != 0
    assert "crashed mid-training" in proc.stderr  # previous excepthook still runs
    info = _info(tmp_path / "logs")
    assert info["status"] == "failed"
    assert info["last_step"] == 7
    assert info["error"]["type"] == "RuntimeError"


# ---------------------------------------------------------- re-init ---
def test_reinit_before_logging_leaves_one_event_file(tmp_path):
    with _writer(tmp_path) as w:
        w._initialize_writer()
        w.log({"loss": 1.0}, step=0)
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) == 1


def test_reinit_after_logging_keeps_data_and_warns(tmp_path):
    with _writer(tmp_path) as w:
        w.log({"loss": 1.0}, step=0)
        with pytest.warns(UserWarning, match="keeping"):
            w._initialize_writer()
        w.log({"loss": 2.0}, step=1)
    assert sorted(_events(tmp_path)["loss"]) == [1.0, 2.0]


def test_new_writer_options_take_precedence(tmp_path):
    w = _writer(tmp_path, flush_secs=120)
    w._initialize_writer(flush_secs=5)
    assert w._writer_options["flush_secs"] == 5
    w.close()


def test_comet_passes_config_as_keyword(tmp_path, monkeypatch):
    seen = {}
    w = _writer(tmp_path)
    monkeypatch.setitem(sys.modules, "comet_ml", type(sys)("comet_ml"))
    monkeypatch.setattr(w, "_initialize_writer", lambda **kw: seen.update(kw))
    with pytest.warns(UserWarning):
        w.with_comet(api_key="k", disabled=True)
    assert seen["comet_config"]["api_key"] == "k"
    w.close()


# ------------------------------------------------------------------ git ---
def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "train.py").write_text("lr = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


def test_git_diff_saved_and_applied_by_replay(repo, tmp_path):
    (repo / "train.py").write_text("lr = 2\n")
    (repo / "notes.txt").write_text("untracked\n")
    logdir = tmp_path / "logs"
    with _writer(logdir, log_git_info=True, git_root=repo):
        pass
    git = _info(logdir)["git"]
    assert git["dirty"] is True
    assert git["untracked"] == ["notes.txt"]
    assert "reset --hard" not in git["checkout_command"]
    assert git["checkout_command"].endswith(f"git apply {git['diff_file']}")

    # the patch recreates the dirty tree on a clean checkout of the commit
    replay = tmp_path / "replay"
    _git(tmp_path, "clone", "-q", str(repo), str(replay))
    _git(replay, "switch", "-q", "--detach", git["commit"])
    _git(replay, "apply", git["diff_file"])
    assert (replay / "train.py").read_text() == "lr = 2\n"


def test_clean_tree_has_no_diff(repo, tmp_path):
    logdir = tmp_path / "logs"
    with _writer(logdir, log_git_info=True, git_root=repo):
        pass
    git = _info(logdir)["git"]
    assert git["dirty"] is False
    assert "diff_file" not in git
    assert not (logdir / "git.diff").exists()


def test_git_info_on_missing_path_does_not_raise(tmp_path):
    GitInfo.collect(start_path=tmp_path / "does-not-exist")


# --------------------------------------------------------- integrations ---
def _fake_wandb(calls, fail_online=True, run_dir=None):
    class FakeRun:
        id, url = "run-id", "https://wandb.example/run"

        def __init__(self, offline):
            self.offline = offline
            self.dir = run_dir

        def finish(self):
            pass

    class FakeWandb:
        class Settings:
            def __init__(self, **kw):
                self.kw = kw

        @staticmethod
        def init(**kw):
            calls.append(kw)
            offline = kw.get("mode") == "offline"
            if fail_online and not offline:
                raise ConnectionError("ProxyError")
            return FakeRun(offline)

        @staticmethod
        def finish():
            pass

    return FakeWandb


def test_wandb_falls_back_to_offline_and_records_sync_command(tmp_path, monkeypatch):
    calls = []
    run_dir = tmp_path / "wandb" / "offline-run-20261110_120000-run" / "files"
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(calls, run_dir=str(run_dir)))
    w = _writer(tmp_path)
    with pytest.warns(UserWarning, match="continuing offline"):
        w.with_wandb(group="g", config={"lr": 1})
    # recorded straight away, not only at close: a run killed later still has it
    entry = _info(tmp_path)["integrations"]["wandb"]
    w.close()
    assert [c.get("mode") for c in calls] == [None, "offline"]
    assert calls[0]["config"] == {"lr": 1}
    assert calls[0]["settings"].kw == {"init_timeout": 90}
    assert entry["status"] == "offline_fallback"
    assert entry["error"] == "ConnectionError: ProxyError"
    assert "Run the command" in entry["note"]
    assert entry["command"] == f"wandb sync {run_dir.parent}"


def test_wandb_explicit_offline_also_gets_sync_command(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb([], fail_online=False))
    with _writer(tmp_path) as w:
        w.with_wandb(mode="offline")
    entry = _info(tmp_path)["integrations"]["wandb"]
    assert entry["status"] == "offline"
    assert entry["command"].startswith("wandb sync ")
    assert "error" not in entry


def test_wandb_online_records_url(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb([], fail_online=False))
    with _writer(tmp_path) as w:
        w.with_wandb()
    entry = _info(tmp_path)["integrations"]["wandb"]
    assert entry == {"status": "online", "id": "run-id", "url": "https://wandb.example/run"}


def test_wandb_failure_without_fallback_is_recorded_then_raised(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb([]))
    w = _writer(tmp_path)
    with pytest.raises(ConnectionError):
        w.with_wandb(fallback_offline=False)
    w.close()
    entry = _info(tmp_path)["integrations"]["wandb"]
    assert entry["status"] == "failed"
    assert "ProxyError" in entry["error"]


def test_clearml_falls_back_to_offline(tmp_path, monkeypatch):
    state = {"offline": False, "closed": False}

    class FakeTask:
        id = "task-id"

        @classmethod
        def init(cls, **kw):
            if not state["offline"]:
                raise ConnectionError("no server")
            return cls()

        @staticmethod
        def set_offline(offline_mode):
            state["offline"] = offline_mode

        def get_offline_mode_folder(self):
            return "/cache/clearml/offline/abc"

        def close(self):
            state["closed"] = True

    module = type(sys)("clearml")
    module.Task = FakeTask
    monkeypatch.setitem(sys.modules, "clearml", module)
    w = _writer(tmp_path)
    with pytest.warns(UserWarning):
        w.with_clearml()
    w.close()
    entry = _info(tmp_path)["integrations"]["clearml"]
    assert entry["status"] == "offline_fallback"
    assert "import_offline_session('/cache/clearml/offline/abc.zip')" in entry["command"]
    assert state["closed"]  # the offline zip is written on close


# ---------------------------------------------------------------- pluto ---
class _FakePlutoRun:
    def __init__(self, store):
        self.id = 42
        self.store = store

    def log(self, data, step=None, timestamp=None):
        self.store["logs"].append((dict(data), step, timestamp))

    def finish(self):
        self.store["finished"] = True


def _fake_pluto(store, fail=False):
    module = type(sys)("pluto")

    def init(**kw):
        store["init"].append(kw)
        if fail:
            raise ConnectionError("Failed to create or resume run. Check connection to Pluto server.")
        return _FakePlutoRun(store)

    module.init = init
    return module


def _pluto_store():
    return {"init": [], "logs": [], "finished": False}


def test_pluto_online_forwards_scalars(tmp_path, monkeypatch):
    store = _pluto_store()
    monkeypatch.setitem(sys.modules, "pluto", _fake_pluto(store))
    with _writer(tmp_path) as w:
        w.with_pluto(config={"lr": 0.1}, host="pluto.internal")
        w.log({"loss": 0.5}, step=3)
    init = store["init"][0]
    assert init["project"] == "p" and init["name"] == "run"
    assert init["settings"] == {"host": "pluto.internal"}
    assert store["logs"][0][:2] == ({"loss": 0.5}, 3)
    assert store["finished"]
    assert json.loads((tmp_path / "config.json").read_text()) == {"lr": 0.1}
    assert _info(tmp_path)["integrations"]["pluto"] == {"status": "online", "id": 42, "host": "pluto.internal"}


def test_pluto_failure_keeps_training_and_records_upload_command(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pluto", _fake_pluto(_pluto_store(), fail=True))
    w = _writer(tmp_path)  # CSV off: the fallback must switch it on
    with pytest.warns(UserWarning, match="continuing with local logs"):
        w.with_pluto(host="pluto.internal")
    w.log({"loss": 1.0}, step=0)
    w.close()
    entry = _info(tmp_path)["integrations"]["pluto"]
    assert entry["status"] == "local_only"
    assert "Failed to create or resume run" in entry["error"]
    assert "data.csv was switched on" in entry["note"]
    assert entry["command"] == f"python -m atalaya.upload pluto {tmp_path} --host pluto.internal"
    assert len(_csv_rows(tmp_path)) == 1 + 1


def test_upload_replays_csv_into_pluto(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pluto", _fake_pluto(_pluto_store(), fail=True))
    with pytest.warns(UserWarning):
        with _writer(tmp_path) as w:
            w.with_pluto(config={"lr": 0.1})
            for step in range(3):
                w.log({"loss": float(step), "acc": step / 10}, step=step)

    from atalaya.upload import main

    store = _pluto_store()
    monkeypatch.setitem(sys.modules, "pluto", _fake_pluto(store))
    assert main(["pluto", str(tmp_path)]) == 0
    assert store["init"][0]["config"] == {"lr": 0.1}
    assert store["init"][0]["name"] == "run"
    assert [(d, s) for d, s, _ in store["logs"]] == [
        ({"loss": float(s), "acc": s / 10}, s) for s in range(3)
    ]
    assert all(t is not None for _, _, t in store["logs"])  # original wall-clock times
    entry = _info(tmp_path)["integrations"]["pluto"]
    assert entry["status"] == "uploaded" and entry["uploaded_steps"] == 3
