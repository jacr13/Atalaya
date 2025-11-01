import csv
import json
import os
import shutil
import sys
import threading
import time
import warnings
from datetime import datetime
from typing import Literal

import numpy as np
from tensorboardX import SummaryWriter


class OutputCatcher:
    def __init__(
        self,
        logdir,
        filename="log.txt",
        with_time=False,
        catch_stdout=True,
        catch_errors=False,
    ):
        self.log_filename = os.path.join(logdir, filename)
        self._with_time = with_time
        self._catch_stdout = catch_stdout
        self._catch_errors = catch_errors

        # Save the original sys.stdout and sys.stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create an OS-level pipe
        self.pipe_r, self.pipe_w = os.pipe()
        self.pipe_writer = os.fdopen(self.pipe_w, "w")

        # Replace both stdout and stderr with our catcher
        if self._catch_stdout:
            sys.stdout = self
        if self._catch_errors:
            sys.stderr = self

        # Control variable for the background thread
        self.running = True

        # Start the background thread that reads from the pipe
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def write(self, data):
        # Write data to the pipe
        self.pipe_writer.write(data)
        self.pipe_writer.flush()

    def flush(self):
        # flush() is implemented for a file-like object
        self.pipe_writer.flush()

    def _reader_thread(self):
        # Open a file object for the read end of the pipe
        pipe_reader = os.fdopen(self.pipe_r, "r")
        with open(self.log_filename, "a") as logfile:
            while self.running:
                # Read one line at a time (blocking)
                line = pipe_reader.readline()
                if not line:
                    break
                # Optionally write back to original stdout for live output
                self.original_stdout.write(line)
                self.original_stdout.flush()

                # Write the captured output to the log file
                if self._with_time:
                    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"
                logfile.write(line)
                logfile.flush()

    def stop(self):
        # Restore the original stdout and stderr
        self.running = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Closing the writer to signal the thread to stop
        self.pipe_writer.close()
        self.thread.join()


class CSVLogger:
    def __init__(
        self,
        logdir: str,
        filename: str = "data.csv",
        flush_interval: int = 10,
        initial_time: float = time.time(),
    ):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.filepath = os.path.join(logdir, filename)

        self._initial_time = initial_time
        self._flush_interval = flush_interval
        self._lines_to_write = []

        # add header
        self._lines_to_write.append(
            ["global_step", "tag", "value", "walltime", "relative_time"]
        )

    def add(
        self,
        tag: str,
        value,
        global_step: int,
        walltime: float,
        initial_time: float | None = None,
    ):
        """
        Add a new log entry to the CSV.

        Parameters:
            tag (str): The tag name, optionally with prefix (e.g. 'train/accuracy').
            value: The logged value (e.g. a scalar number).
            global_step (int): The current global step or epoch.
            walltime (float): The current walltime (e.g. time.time()).
            initial_time (float): A reference start time to compute the relative time.
        """
        initial_time = initial_time or self._initial_time

        # Calculate the relative time since the initial time.
        relative_time = walltime - initial_time
        self._lines_to_write.append([global_step, tag, value, walltime, relative_time])

        if len(self._lines_to_write) >= self._flush_interval:
            self.flush()

    def flush(self):
        with open(self.filepath, "a+") as file:
            csv_writer = csv.writer(file)
            for line in self._lines_to_write:
                csv_writer.writerow(line)

        self._lines_to_write = []


class Writer(SummaryWriter):
    def __init__(
        self,
        name: str = "default_name",
        project: str = "default_project",
        logdir: str = "logs",
        add_name_to_logdir: bool = False,
        add_time: bool = False,
        output_catcher: bool = False,
        save_as_csv: bool = False,
        save_code: bool = True,
        **writer_options,
    ):
        self._initial_time = time.time()
        self.name = name
        self.project = project
        self.logdir = logdir

        if add_time:
            self.name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.name}"

        if add_name_to_logdir:
            self.logdir = os.path.join(self.logdir, self.name)

        self._wandb_run = None
        self._neptune_run = None
        self._clearml_run = None
        self._plt = None
        self._sns = None

        self._csv_logging = save_as_csv

        self._output_catcher = None
        if output_catcher:
            self.with_output_catcher()

        self._writer_options = writer_options
        self._event_file_path = None
        self._initialize_writer()

        if save_code:
            shutil.copy(
                os.path.abspath(sys.argv[0]), os.path.join(self.logdir, sys.argv[0])
            )

    def _initialize_writer(self, **writer_options):
        """Initialize the writer and handle any existing event files."""
        # Close and remove any previous writer's event file if it exists
        if self._event_file_path is not None:
            super(Writer, self).close()
            if os.path.exists(self._event_file_path):
                os.remove(self._event_file_path)

        writer_options.update(self._writer_options)
        self._writer_options = writer_options

        # Initialize new writer and save the event file path
        super(Writer, self).__init__(logdir=self.logdir, **self._writer_options)
        self._event_file_path = self.file_writer.event_writer._ev_writer._file_name

    def with_wandb(
        self,
        group: str,
        entity: str,
        monitor_gym: bool = False,
        save_code: bool = True,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
        **wandb_options,
    ):
        """Configure and initialize a Weights & Biases (wandb) run."""
        import wandb

        # TODO: missing config
        self._wandb_run = wandb.init(
            project=self.project,
            group=group,
            entity=entity,
            dir=self.logdir,
            sync_tensorboard=True,
            monitor_gym=monitor_gym,
            save_code=save_code,
            name=self.name,
            id=self.name,
            resume=resume,
            **wandb_options,
        )

        # Reinitialize SummaryWriter to sync with wandb's file_writer
        self._initialize_writer()

    def with_comet(self, api_key: str, disabled: bool = False, **comet_config):
        """Configure and initialize comet_logger."""
        warnings.warn(
            "Comet integration is adapted from the public documentation and has not been fully validated within Atalaya. Use with caution.",
            UserWarning,
        )
        import comet_ml

        if api_key is None:
            api_key = os.getenv("COMET_API_KEY")

        assert (
            api_key is not None
        ), "COMET_API_KEY is not set or api_key is not provided"

        comet_config["project_name"] = self.project
        comet_config["api_key"] = api_key
        comet_config["disabled"] = disabled

        self._initialize_writer("comet_config", **comet_config)

    def with_clearml(self, **clearml_config):
        """Configure and initialize clearml_logger."""
        warnings.warn(
            "ClearML integration is adapted from the public documentation and has not been fully validated within Atalaya. Use with caution.",
            UserWarning,
        )
        from clearml import Task

        self._clearml_run = Task.init(
            task_name=self.name, project_name=self.project, **clearml_config
        )

    def with_neptune(self, entity: str, **neptune_config):
        """Configure and initialize neptune_logger."""
        warnings.warn(
            "Neptune integration is adapted from the public documentation and has not been fully validated within Atalaya. Use with caution.",
            UserWarning,
        )
        import neptune
        from neptune_tensorboard import enable_tensorboard_logging

        self._neptune_run = neptune.init_run(
            project=f"{entity}/{self.project}", name=self.name, **neptune_config
        )

        enable_tensorboard_logging(self._neptune_run)

        self._initialize_writer()

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
            logdir=self.logdir,
            filename=filename,
            with_time=with_time,
            catch_stdout=catch_stdout,
            catch_errors=catch_errors,
        )

    def with_csv_logger(self):
        self._csv_logging = True
        self._csv_logger = CSVLogger(self.logdir)

    def add_scalars(
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
            if hasattr(value, "item"):
                value = value.item()
            self.add_scalar(tag, value, global_step, walltime, **extra_options)

            if self._csv_logging:
                self._csv_logger.add(
                    tag, value, global_step, walltime, initial_time=self._initial_time
                )

        self.flush()

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

    def close(self):
        """Close the writer and finish the wandb run if applicable."""
        super(Writer, self).close()
        if self._wandb_run is not None:
            self._wandb_run.finish()

    # compatibility methods
    def log(self, data, step=None, prefix=None, **log_options):
        """Log a dictionary of scalar values at a specific step."""
        self.add_scalars(
            scalar_values=data, global_step=step, prefix=prefix, **log_options
        )

    # helper methods
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
