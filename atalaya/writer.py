import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from tensorboardX import SummaryWriter

from .logging import CSVLogger, GitInfo, OutputCatcher, collect_run_info


class Writer(SummaryWriter):
    def __init__(
        self,
        name: str = "default_name",
        project: str = "default_project",
        logdir: str = "logs",
        add_name_to_logdir: bool = False,
        add_time: bool = False,
        output_catcher: bool = False,
        log_git_info: bool = True,
        save_as_csv: bool = False,
        save_code: bool = True,
        extra_info: dict | None = None,
        **writer_options,
    ):
        self._initial_time = time.time()
        self.name = name
        self.project = project
        self.logdir = Path(logdir)

        if add_time:
            self.name = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if add_name_to_logdir:
            self.logdir = self.logdir / self.name

        self._wandb_run = None
        self._neptune_run = None
        self._clearml_run = None
        self._plt = None
        self._sns = None
        self._csv_logger = None
        self._output_catcher = None
        self._event_file_path = None

        if save_as_csv:
            self.with_csv_logger()

        if output_catcher:
            self.with_output_catcher()

        self._writer_options = writer_options
        self._initialize_writer()

        source_path = Path(sys.argv[0]).resolve()
        if save_code and source_path.exists():
            destination = Path(self.logdir) / "code" / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, destination)

        run_info = collect_run_info(
            run_name=self.name,
            project=self.project,
            logdir=self.logdir,
            source_path=source_path,
            extra=extra_info,
        )

        if log_git_info:
            git_info = GitInfo.collect(
                start_path=source_path.parent if source_path.exists() else Path.cwd(),
                run_name=self.name,
            )
            if git_info is not None:
                run_info["git"] = git_info.to_dict() 

        info_path = Path(self.logdir) / "info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(run_info, indent=2))

    def _initialize_writer(self, **writer_options):
        """Initialize the writer and handle any existing event files."""
        # Close and remove any previous writer's event file if it exists
        if self._event_file_path is not None:
            super(Writer, self).close()
            if self._event_file_path.exists():
                self._event_file_path.unlink()

        writer_options.update(self._writer_options)
        self._writer_options = writer_options

        # Initialize new writer and save the event file path
        self.logdir.mkdir(parents=True, exist_ok=True)
        super(Writer, self).__init__(logdir=str(self.logdir), **self._writer_options)
        self._event_file_path = Path(
            self.file_writer.event_writer._ev_writer._file_name
        )
        # SummaryWriter may overwrite logdir, ensure we keep a Path instance
        self.logdir = Path(self.logdir)

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
            dir=str(self.logdir),
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
            logdir=str(self.logdir),
            filename=filename,
            with_time=with_time,
            catch_stdout=catch_stdout,
            catch_errors=catch_errors,
        )

    def with_csv_logger(self):
        self._csv_logger = CSVLogger(str(self.logdir))

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

            if self._csv_logger is not None:
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
        if self._csv_logger is not None:
            self._csv_logger.flush()
        if self._output_catcher is not None:
            self._output_catcher.stop()

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
