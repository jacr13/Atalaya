import atexit
import functools
import inspect
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from tensorboardX import SummaryWriter

from .logging import (
    CSVLogger,
    GitInfo,
    OutputCatcher,
    collect_environment,
    collect_run_info,
    write_requirements,
)
from .video import ffmpeg_available, video_summary

# Writers still open, so an uncaught exception can be recorded in their info.json
# before atexit closes them. Weak, so the registry never keeps a writer alive.
_live_writers: "weakref.WeakSet[Writer]" = weakref.WeakSet()
_previous_excepthook = None


def _excepthook(exc_type, exc, tb):
    for writer in list(_live_writers):
        writer._record_exception(exc_type, exc, tb)
    (_previous_excepthook or sys.__excepthook__)(exc_type, exc, tb)


def _install_excepthook():
    global _previous_excepthook
    if sys.excepthook is not _excepthook:
        _previous_excepthook = sys.excepthook
        sys.excepthook = _excepthook


# Every open writer, for the SIGTERM handler.
_open_writers: "weakref.WeakSet[Writer]" = weakref.WeakSet()
_previous_sigterm = None


def _sigterm_handler(signum, frame):
    """Save what the writers hold before the process goes away.

    SLURM (and docker, systemd, k8s) send SIGTERM some seconds before SIGKILL.
    With no other handler installed, writers are closed (status "terminated")
    and the process then dies by the signal exactly as it would have. If the
    program had its own handler, writers are only flushed and marked, and that
    handler runs: it may still want to log (a last checkpoint, final metrics),
    and the writers are closed later, at exit.
    """
    previous = _previous_sigterm
    chain = callable(previous)
    for writer in list(_open_writers):
        try:
            writer._on_sigterm(close=not chain)
        except Exception:
            pass
    if chain:
        previous(signum, frame)
    elif previous == signal.SIG_IGN:
        return
    else:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


def _install_sigterm_handler():
    global _previous_sigterm
    if threading.current_thread() is not threading.main_thread():
        return False  # signal.signal only works in the main thread
    current = signal.getsignal(signal.SIGTERM)
    if current is not _sigterm_handler:
        _previous_sigterm = current
        signal.signal(signal.SIGTERM, _sigterm_handler)
    return True


# Methods that async_writes=True moves onto the writer thread.
_ASYNC_METHODS = (
    "add_scalar",
    "log_scalars",
    "add_scalars",
    "add_histogram",
    "add_image",
    "add_images",
    "add_figure",
    "add_video",
    "add_text",
    "add_models",
)


class Writer(SummaryWriter):
    """tensorboardX SummaryWriter that also keeps a run's metadata on disk.

    In ``logdir`` it writes, besides the event file:
      info.json   command, python, host, SLURM context, git state, and at the end
                  status (finished / failed / interrupted / exited_without_close),
                  end time, duration and last step
      git.diff    uncommitted changes to tracked files, when the tree is dirty
      data.csv    every scalar, long format (on by default); data.csv.gz
                  with ``csv_compress=True``
      log.txt     what the program prints, stdout and stderr (``output_catcher``)
      config.json the run's configuration (``config=``)
      requirements.txt  installed packages (``save_requirements=True``)

    Use it as a context manager, or call ``close()``. If neither happens, it is
    closed at interpreter exit, and an uncaught exception is recorded first. On
    SIGTERM (SLURM's warning before SIGKILL) it is closed or flushed in time.

    ``async_writes=True`` runs logging calls on one background thread, in
    order, so slow ones (videos, histograms, figures) do not stall the caller.
    Values are read on that thread: do not modify a logged array or tensor in
    place afterwards. An error raised there is re-raised on the next logging
    call.
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        logdir: str = "logs",
        add_name_to_logdir: bool = False,
        add_time: bool = False,
        output_catcher: bool = True,
        log_git_info: bool = True,
        save_git_diff: bool = True,
        git_root: str | Path | None = None,
        save_as_csv: bool = True,
        save_code: bool = True,
        extra_info: dict | None = None,
        config: dict | None = None,
        flush_on_log: bool = True,
        capture_exceptions: bool = True,
        handle_sigterm: bool = True,
        log_environment: bool = True,
        save_requirements: bool = True,
        csv_compress: bool = False,
        csv_flush_secs: float | None = 30.0,
        fast_video: bool = True,
        async_writes: bool = False,
        **writer_options,
    ):
        self._closed = True  # until fully initialised, close() is a no-op
        self._initial_time = time.time()
        from ._session import _script_name

        # same defaults as atl.init: the entry script names the project and run
        if name is None:
            name = f"{_script_name()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.name = name
        self.project = project if project is not None else _script_name()
        self.logdir = Path(logdir)

        if add_time:
            self.name = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if add_name_to_logdir:
            self.logdir = self.logdir / self.name

        self._wandb_run = None
        self._clearml_run = None
        self._pluto_run = None
        self._pluto_log_failed = False
        self._plt = None
        self._sns = None
        self._csv_logger = None
        self._output_catcher = None
        self._event_file_path = None
        self._logged_since_init = False
        self._flush_on_log = flush_on_log
        self._last_step = None
        self._status = None
        self._error = None
        self.config = config
        self._csv_compress = csv_compress
        self._csv_flush_secs = csv_flush_secs
        self._fast_video = fast_video
        self._comet_enabled = False
        self._timers = []
        self._executor = None
        self._worker_ident = None
        self._async_error = None

        if save_as_csv:
            self.with_csv_logger()

        if output_catcher:
            self.with_output_catcher()

        self._writer_options = writer_options
        self._initialize_writer()

        source_path = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
        if source_path is not None and not source_path.is_file():
            source_path = None
        if save_code and source_path is not None:
            destination = Path(self.logdir) / "code" / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination)

        self._run_info = collect_run_info(
            run_name=self.name,
            project=self.project,
            logdir=self.logdir,
            source_path=source_path,
            extra=extra_info,
        )

        if log_environment:
            self._run_info["environment"] = collect_environment()
        if save_requirements:
            write_requirements(self.logdir / "requirements.txt")
        if config is not None:
            self._save_config(config)

        if log_git_info:
            if git_root is None:
                git_root = source_path.parent if source_path is not None else Path.cwd()
            git_info = GitInfo.collect(
                start_path=git_root,
                run_name=self.name,
                diff_destination=self.logdir / "git.diff" if save_git_diff else None,
            )
            if git_info is not None:
                self._run_info["git"] = git_info.to_dict()

        self._info_path = Path(self.logdir) / "info.json"
        self._run_info["status"] = "running"
        self._write_info()

        self._closed = False
        atexit.register(self._atexit_close)
        _open_writers.add(self)
        if capture_exceptions:
            _live_writers.add(self)
            _install_excepthook()
        if handle_sigterm:
            _install_sigterm_handler()
        if async_writes:
            self._enable_async()

    # ------------------------------------------------------------------ setup
    def _initialize_writer(self, **writer_options):
        """(Re)create the underlying SummaryWriter.

        Re-creating is how wandb/comet hook in. The previous event file is
        removed only if nothing was logged to it; otherwise it is kept and a
        warning says so, instead of silently losing data.
        """
        self._writer_options.update(writer_options)

        previous = self._event_file_path
        if previous is not None:
            has_data = self._logged_since_init
            super().close()
            if has_data:
                warnings.warn(
                    f"Re-initialising the writer after logging: keeping {previous.name}. "
                    "Call with_wandb()/with_comet() before logging to have a single event file."
                )
            elif previous.exists():
                previous.unlink()

        options = dict(self._writer_options)
        if previous is not None and previous.exists():
            # Event files are named <time-in-seconds>.<host>; within the same
            # second the new file would truncate the one being kept.
            self._reinit_count = getattr(self, "_reinit_count", 0) + 1
            options["filename_suffix"] = f"{options.get('filename_suffix', '')}.{self._reinit_count}"

        self.logdir.mkdir(parents=True, exist_ok=True)
        super().__init__(logdir=str(self.logdir), **options)
        # SummaryWriter may overwrite logdir, ensure we keep a Path instance
        self.logdir = Path(self.logdir)
        self._event_file_path = self._find_event_file()
        self._logged_since_init = False

    def _get_file_writer(self):
        # Every add_* goes through here (and so does __init__, which resets the
        # flag afterwards). File size cannot tell: tensorboardX writes events
        # from a queue, so they reach the file only later.
        self._logged_since_init = True
        return super()._get_file_writer()

    def _find_event_file(self) -> Path | None:
        try:  # tensorboardX internals; fall back to the newest event file
            return Path(self.file_writer.event_writer._ev_writer._file_name)
        except AttributeError:
            files = sorted(self.logdir.glob("events.out.tfevents.*"), key=os.path.getmtime)
            return files[-1] if files else None

    def with_wandb(
        self,
        group: str | None = None,
        entity: str | None = None,
        monitor_gym: bool = False,
        save_code: bool = True,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
        config: dict | None = None,
        init_timeout: float | None = 90,
        fallback_offline: bool = True,
        **wandb_options,
    ):
        """Mirror this writer's TensorBoard logs to a Weights & Biases run.

        If ``wandb.init`` fails (no network, proxy errors, timeout) and
        ``fallback_offline`` is set, the run continues in offline mode. info.json
        then says so under ``integrations.wandb`` and gives the ``wandb sync``
        command to upload it. Local logging never depends on W&B.
        """
        import wandb

        if config is None:
            config = self.config
        kwargs = dict(
            project=self.project,
            group=group,
            entity=entity,
            dir=str(self.logdir),
            sync_tensorboard=True,
            monitor_gym=monitor_gym,
            save_code=save_code,
            name=self.name,
            id=self.name,
            resume=resume,
            config=config,
            **wandb_options,
        )
        if init_timeout is not None and "settings" not in kwargs:
            kwargs["settings"] = wandb.Settings(init_timeout=init_timeout)

        error = None
        try:
            self._wandb_run = wandb.init(**kwargs)
        except Exception as e:
            if not fallback_offline or kwargs.get("mode") in ("offline", "disabled"):
                self._record_integration(
                    "wandb", status="failed", error=f"{type(e).__name__}: {e}"
                )
                raise
            error = f"{type(e).__name__}: {e}"
            try:
                wandb.finish()
            except Exception:
                pass
            kwargs["mode"] = "offline"
            self._wandb_run = wandb.init(**kwargs)

        run = self._wandb_run
        if kwargs.get("mode") == "disabled":
            self._record_integration("wandb", status="disabled")
        elif kwargs.get("mode") == "offline" or getattr(run, "offline", False):
            command = f"wandb sync {self._wandb_offline_dir(run)}"
            if error is not None:
                note = (
                    "wandb.init failed, so this run was logged to W&B offline. "
                    "Local logs are complete. Run the command to upload it."
                )
                warnings.warn(f"wandb.init failed ({error}); continuing offline. Upload later with: {command}")
            else:
                note = "Logged to W&B offline. Run the command to upload it."
            self._record_integration(
                "wandb",
                status="offline_fallback" if error is not None else "offline",
                id=getattr(run, "id", None),
                error=error,
                note=note,
                command=command,
            )
        else:
            self._record_integration(
                "wandb", status="online", id=getattr(run, "id", None), url=getattr(run, "url", None)
            )

        # Reinitialize SummaryWriter so wandb's tensorboard sync picks it up
        self._initialize_writer()

    def _wandb_offline_dir(self, run) -> str:
        # run.dir is <logdir>/wandb/offline-run-<time>-<id>/files
        run_dir = getattr(run, "dir", None)
        if run_dir and Path(run_dir).name == "files":
            return str(Path(run_dir).parent)
        return f"{self.logdir}/wandb/offline-run-*"

    def _record_integration(self, name: str, **fields):
        """Record an integration's state under ``integrations.<name>`` in info.json.

        When it did not reach its server, ``note`` says what happened and
        ``command`` is what to run afterwards to upload the run.
        """
        entry = {k: v for k, v in fields.items() if v is not None}
        self._run_info.setdefault("integrations", {})[name] = entry
        self._write_info()

    def with_pluto(
        self,
        config: dict | None = None,
        tags: str | list[str] | None = None,
        host: str | None = None,
        fallback_local: bool = True,
        **pluto_options,
    ):
        """Send scalars to a Pluto run (pluto-ml, a W&B-like tracker that can be self-hosted).

        Scalars logged through this writer are forwarded to Pluto as they are
        logged (histograms, images and videos stay in TensorBoard only).
        ``host`` points at a self-hosted server; leave it None for Pluto's cloud.
        Credentials come from ``PLUTO_API_KEY`` or ``pluto login``.

        Pluto has no offline mode: it must create the run on its server. If that
        fails and ``fallback_local`` is set, the run continues with local logs
        only, data.csv is switched on if it was off, and info.json records
        ``python -m atalaya.upload pluto <logdir>``, which replays data.csv into
        Pluto later.
        """
        import pluto

        if config is None:
            config = self.config
        elif config is not self.config:
            self._save_config(config)
        settings = dict(pluto_options.pop("settings", None) or {})
        if host is not None:
            settings["host"] = host
        kwargs = dict(
            dir=str(self.logdir),
            project=self.project,
            name=self.name,
            config=config,
            tags=tags,
            settings=settings or None,
            **pluto_options,
        )
        try:
            self._pluto_run = pluto.init(**kwargs)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if not fallback_local:
                self._record_integration("pluto", status="failed", error=error)
                raise
            note = (
                "pluto.init failed, so nothing was sent to Pluto; local logs are complete. "
                "Run the command to upload this run from data.csv."
            )
            if self._csv_logger is None:
                self.with_csv_logger()
                note += " data.csv was switched on for this."
            if self._last_step is not None:
                note += " Scalars logged before with_pluto() are missing from data.csv."
            command = f"python -m atalaya.upload pluto {Path(self.logdir).resolve()}"
            if host is not None:
                command += f" --host {host}"
            warnings.warn(f"pluto.init failed ({error}); continuing with local logs. Upload later with: {command}")
            self._record_integration(
                "pluto", status="local_only", error=error, note=note, command=command
            )
            return

        self._record_integration(
            "pluto", status="online", id=getattr(self._pluto_run, "id", None), host=host
        )

    def _save_config(self, config: dict):
        """config.json next to info.json, so an upload done later has it too."""
        path = Path(self.logdir) / "config.json"
        path.write_text(json.dumps(config, indent=2, default=str))

    def with_comet(self, api_key: str | None = None, disabled: bool = False, **comet_config):
        """Configure and initialize comet_logger."""
        warnings.warn(
            "Comet integration is adapted from the public documentation and has not been fully validated within Atalaya. Use with caution.",
            UserWarning,
        )
        import comet_ml  # noqa: F401  (fail early if missing)

        if api_key is None:
            api_key = os.getenv("COMET_API_KEY")

        assert (
            api_key is not None
        ), "COMET_API_KEY is not set or api_key is not provided"

        comet_config["project_name"] = self.project
        comet_config["api_key"] = api_key
        comet_config["disabled"] = disabled

        self._initialize_writer(comet_config=comet_config)
        self._comet_enabled = not disabled
        # tensorboardX creates the Comet experiment lazily, on first use, so a
        # connection failure surfaces there rather than here.
        self._record_integration("comet", status="disabled" if disabled else "enabled")

    def with_clearml(self, fallback_offline: bool = True, **clearml_config):
        """Configure and initialize clearml_logger."""
        warnings.warn(
            "ClearML integration is adapted from the public documentation and has not been fully validated within Atalaya. Use with caution.",
            UserWarning,
        )
        from clearml import Task

        kwargs = dict(task_name=self.name, project_name=self.project, **clearml_config)
        try:
            self._clearml_run = Task.init(**kwargs)
            self._record_integration("clearml", status="online", id=getattr(self._clearml_run, "id", None))
        except Exception as e:
            if not fallback_offline:
                self._record_integration("clearml", status="failed", error=f"{type(e).__name__}: {e}")
                raise
            error = f"{type(e).__name__}: {e}"
            Task.set_offline(offline_mode=True)
            self._clearml_run = Task.init(**kwargs)
            # the offline session is zipped next to this folder when the task closes
            folder = self._clearml_run.get_offline_mode_folder()
            command = (
                'python -c "from clearml import Task; '
                f"Task.import_offline_session('{folder}.zip')\""
            )
            warnings.warn(f"ClearML Task.init failed ({error}); continuing offline. Upload later with: {command}")
            self._record_integration(
                "clearml",
                status="offline_fallback",
                error=error,
                note=(
                    "Task.init failed, so this run was logged to ClearML offline. "
                    "Local logs are complete. Run the command to upload it."
                ),
                command=command,
            )

    def with_output_catcher(
        self,
        filename: str = "log.txt",
        with_time: bool = True,
        catch_stdout: bool = True,
        catch_errors: bool = True,
    ):
        if self._output_catcher is not None:
            warnings.warn("You already have a print_catcher. This call is ignored.")
            return
        self._output_catcher = OutputCatcher(
            logdir=str(self.logdir),
            filename=filename,
            with_time=with_time,
            catch_stdout=catch_stdout,
            catch_errors=catch_errors,
        )

    def with_csv_logger(self):
        self._csv_logger = CSVLogger(
            str(self.logdir),
            initial_time=self._initial_time,
            compress=self._csv_compress,
            flush_secs=self._csv_flush_secs,
        )

    # ---------------------------------------------------------------- logging
    def add_scalar(
        self,
        tag: str,
        scalar_value,
        global_step: int | None = None,
        walltime: float | None = None,
        **extra_options,
    ):
        """tensorboardX ``add_scalar``, also written to data.csv when enabled."""
        walltime = time.time() if walltime is None else walltime
        if hasattr(scalar_value, "item"):
            scalar_value = scalar_value.item()
        super().add_scalar(tag, scalar_value, global_step, walltime, **extra_options)
        if global_step is not None:
            self._last_step = global_step if self._last_step is None else max(self._last_step, global_step)
        if self._csv_logger is not None:
            self._csv_logger.add(
                tag, scalar_value, global_step, walltime, initial_time=self._initial_time
            )
        if self._pluto_run is not None:
            self._log_to_pluto(tag, scalar_value, global_step, walltime)

    def _log_to_pluto(self, tag, value, step, walltime):
        try:
            self._pluto_run.log({tag: value}, step=step, timestamp=walltime)
        except Exception as e:  # never let the tracker stop training
            if not self._pluto_log_failed:
                self._pluto_log_failed = True
                warnings.warn(f"Pluto logging failed ({type(e).__name__}: {e}); local logs continue.")

    def log_scalars(
        self,
        scalar_values: dict,
        global_step: int,
        prefix: str | None = None,
        walltime: float | None = None,
        **extra_options,
    ):
        """Log a dictionary of scalar values at a specific step."""
        walltime = time.time() if walltime is None else walltime
        for tag, value in scalar_values.items():
            if prefix is not None:
                tag = f"{prefix}/{tag}"
            self.add_scalar(tag, value, global_step, walltime, **extra_options)

        if self._flush_on_log:
            self.flush()

    def add_scalars(self, *args, **kwargs):
        """Two meanings, chosen by the first argument.

        ``add_scalars({"loss": ...}, step)`` -> :meth:`log_scalars` (atalaya).
        ``add_scalars("main_tag", {"a": ...}, step)`` -> tensorboardX's own
        add_scalars (several curves on one chart), so code written for
        tensorboardX keeps working.
        """
        first = args[0] if args else kwargs.get("main_tag")
        if isinstance(first, str) or "main_tag" in kwargs:
            return super().add_scalars(*args, **kwargs)
        return self.log_scalars(*args, **kwargs)

    def add_models(
        self,
        model_dict: dict,
        global_step: int,
        log_type: Literal["gradients", "parameters", "all"] = "all",
        bins: int | str | None = 100,
        max_bins: int | None = None,
        heatmap: bool = False,
        heatmap_options: dict | None = None,
    ):
        """Log model parameters and gradients as histograms."""
        heatmap_options = heatmap_options or {}

        def _log(name: str, arr: np.ndarray):
            self.add_histogram(
                name,
                arr,
                global_step,
                bins=bins,
                max_bins=max_bins,
            )
            if heatmap:
                # TODO: heatmap hard to interpret because scale is different for each figure
                fig_heatmap = self._get_fig_heatmap(arr, **heatmap_options)
                self.add_figure(
                    f"heatmap_{name}",
                    fig_heatmap,
                    global_step,
                )
                if self._plt is not None:
                    self.flush()
                    self._plt.close(fig_heatmap)

        for model_name, model in model_dict.items():
            for param_name, param in model.named_parameters():
                if log_type in ("parameters", "all"):
                    _log(
                        f"parameters/{model_name}/{param_name}",
                        param.data.cpu().numpy(),
                    )
                if log_type in ("gradients", "all") and param.grad is not None:
                    _log(
                        f"gradients/{model_name}/{param_name}", param.grad.cpu().numpy()
                    )
            self.flush()

    def add_video(
        self,
        tag: str,
        vid_tensor,
        global_step: int | None = None,
        fps: float = 4,
        walltime: float | None = None,
        dataformats: str = "NTCHW",
    ):
        """tensorboardX ``add_video``, encoded with ffmpeg when it is installed.

        Same arguments and layout as tensorboardX; several videos (N > 1) are
        tiled in a grid. Much faster than tensorboardX's own path (float32
        conversion + moviepy), and no moviepy needed. Without ffmpeg, with
        ``fast_video=False``, or with Comet enabled (which needs tensorboardX's
        path), tensorboardX's own implementation is used.
        """
        if not self._fast_video or self._comet_enabled or not ffmpeg_available():
            return super().add_video(tag, vid_tensor, global_step, fps, walltime, dataformats)
        if hasattr(vid_tensor, "detach"):
            vid_tensor = vid_tensor.detach().cpu().numpy()
        summary = video_summary(tag, np.asarray(vid_tensor), fps, dataformats)
        if global_step is not None:
            self._last_step = global_step if self._last_step is None else max(self._last_step, global_step)
        self._get_file_writer().add_summary(summary, global_step, walltime)

    def attach_timer(self, timer, name: str = "timings.json"):
        """Write ``timer.summary()`` to ``<logdir>/<name>`` when the writer closes.

        The file is also written on SIGTERM and at exit, like info.json. Several
        timers can be attached under different names.
        """
        self._timers.append((timer, name))
        return timer

    def create_timer(self, name: str = "timings.json", **timer_options):
        """Create a :class:`atalaya.timer.Timer` and attach it in one call.

        ``writer.create_timer(cuda_sync=True)`` is
        ``writer.attach_timer(Timer(cuda_sync=True))``.
        """
        from .timer import Timer

        return self.attach_timer(Timer(**timer_options), name=name)

    def log_timings(
        self,
        timer,
        global_step: int,
        steps: int | None = None,
        columns=("frac", "mean"),
        prefix: str = "timings",
    ):
        """Log a few timing columns per section, for the interval since the last call.

        Call it every N steps rather than every step: the point is a cheap
        overview of where time goes. ``steps`` (default ``global_step``) is the
        running count rates are computed from, e.g. environment steps.
        """
        values = timer.flat(steps=global_step if steps is None else steps, columns=columns, prefix=prefix)
        if values:
            self.log_scalars(values, global_step)

    def _write_timers(self):
        for timer, name in self._timers:
            path = Path(self.logdir) / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            steps = self._last_step if self._last_step is not None else None
            tmp.write_text(json.dumps(timer.summary(steps=steps), indent=2, default=str))
            tmp.replace(path)

    # compatibility methods
    def log(self, data, step=None, prefix=None, **log_options):
        """Log a dictionary of scalar values at a specific step (wandb-style)."""
        self.log_scalars(
            scalar_values=data, global_step=step, prefix=prefix, **log_options
        )

    # ------------------------------------------------------------------ async
    def _enable_async(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atalaya-writer")
        for name in _ASYNC_METHODS:
            setattr(self, name, self._make_async(getattr(self, name)))

    def _make_async(self, method):
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        takes_walltime = signature is not None and "walltime" in signature.parameters

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            # on the writer thread (nested calls) or after close: run inline
            if self._executor is None or threading.get_ident() == self._worker_ident:
                return method(*args, **kwargs)
            self._raise_async_error()
            if takes_walltime:
                # time of the call, not of the write
                bound = signature.bind_partial(*args, **kwargs)
                if bound.arguments.get("walltime") is None:
                    kwargs["walltime"] = time.time()
            self._executor.submit(self._run_async, method, args, kwargs)

        return wrapper

    def _run_async(self, method, args, kwargs):
        self._worker_ident = threading.get_ident()
        try:
            method(*args, **kwargs)
        except BaseException as e:  # surfaced on the caller's next call
            if self._async_error is None:
                self._async_error = e

    def _raise_async_error(self):
        if self._async_error is not None:
            error, self._async_error = self._async_error, None
            raise RuntimeError(f"an earlier asynchronous write failed: {error!r}") from error

    def wait(self):
        """Block until every queued write is done (no-op without async_writes)."""
        if self._executor is not None:
            self._executor.submit(lambda: None).result()
            self._raise_async_error()

    def _drain_async(self):
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)
            if self._async_error is not None:
                warnings.warn(f"an asynchronous write failed: {self._async_error!r}")
                self._async_error = None

    # ---------------------------------------------------------------- closing
    def _on_sigterm(self, close: bool):
        if self._closed:
            return
        if self._status is None:
            self._status = "terminated"
        if close:
            self.close()
            return
        # another handler runs next and may still log: flush, mark, stay open
        self.wait()
        if self._csv_logger is not None:
            self._csv_logger.flush()
        self.flush()
        self._write_timers()
        self._run_info["status"] = "terminated"
        self._write_info()

    def close(self):
        """Flush everything, finish integrations and record how the run ended."""
        if getattr(self, "_closed", True):
            return
        self._drain_async()
        self._closed = True
        from ._session import _clear

        _clear(self)  # a closed run stops being atl's current run
        atexit.unregister(self._atexit_close)
        _live_writers.discard(self)
        _open_writers.discard(self)

        if self._csv_logger is not None:
            self._csv_logger.flush()
        super().close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._clearml_run is not None:
            self._clearml_run.close()
        if self._pluto_run is not None:
            self._pluto_run.finish()
        self._write_timers()
        self._finalize_info(self._status or "finished")
        if self._output_catcher is not None:
            self._output_catcher.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._record_exception(exc_type, exc, tb)
        self.close()
        return False

    def _atexit_close(self):
        if self._status is None:
            # sys.exit(), os-level exit paths, or simply no close() call
            self._status = "exited_without_close"
        self.close()

    def _record_exception(self, exc_type, exc, tb):
        if self._closed:
            return
        self._status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
        self._error = {
            "type": exc_type.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
        }

    def _finalize_info(self, status: str):
        end = time.time()
        self._run_info.update(
            status=status,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_seconds=round(end - self._initial_time, 3),
        )
        if self._last_step is not None:
            self._run_info["last_step"] = self._last_step
        if self._error is not None:
            self._run_info["error"] = self._error
        self._write_info()

    def _write_info(self):
        self._info_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._info_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._run_info, indent=2, default=str))
        tmp.replace(self._info_path)  # atomic: a crash never leaves half a file

    # ---------------------------------------------------------------- helpers
    def _get_fig_heatmap(self, data, **heatmap_options):
        if self._plt is None:
            import matplotlib.pyplot as plt

            self._plt = plt
        if self._sns is None:
            import seaborn as sns

            self._sns = sns

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        fig, ax = self._plt.subplots()
        self._sns.heatmap(data, ax=ax, cmap="viridis", cbar=True)
        ax.axis("off")  # Hide axis for a cleaner look
        return fig
