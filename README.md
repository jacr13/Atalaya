# Atalaya

[![PyPI version](https://img.shields.io/pypi/v/atalaya.svg)](https://pypi.org/project/atalaya/)

Atalaya is a lightweight toolkit that helps you keep PyTorch experiments organised. It provides:

- a `Writer` built on top of `tensorboardX` with handy helpers for scalars, models, CSV exports, and optional integrations with Weights & Biases, Neptune, Comet, and ClearML;
- automatic Git metadata snapshots so you can trace runs back to the exact commit;
- a colour-aware `terminal` helper for structured CLI output with timestamps;
- a `Timer` utility to track wall-clock timing for blocks of code or functions;
- optional console log capture so that your scripts remain reproducible.

The writer uses `tensorboardX` as its event backend, so you get native TensorBoard support while still being able to plug in third-party experiment trackers such as Weights & Biases, Neptune, ClearML, or Comet.

| Integration       | Status                                                |
|-------------------|-------------------------------------------------------|
| TensorBoardX      | Works (core backend)                                  |
| Weights & Biases  | Works (`Writer.with_wandb`)                           |
| Neptune           | Not fully validated yet (`Writer.with_neptune`)       |
| ClearML           | Not fully validated yet (`Writer.with_clearml`)       |
| Comet ML          | Not fully validated yet (`Writer.with_comet`)         |

## Installation

Install the core package from PyPI:

```bash
pip install atalaya
```

Extras are available if you want to enable third-party integrations:

```bash
# Enable Weights & Biases support
pip install atalaya[wandb]

# Install everything (Neptune, Comet, ClearML, matplotlib, seaborn)
pip install atalaya[all]
```

## Quick Start

### TensorBoard logging

```python
from atalaya.writer import Writer

writer = Writer(
    name="baseline",
    project="my-awesome-project",
    logdir="logs",
    add_time=True,
    save_as_csv=True,
    output_catcher=True,
)

# Optional: sync logs with Weights & Biases
writer.with_wandb(group="experiments", entity="my-team")

for epoch in range(10):
    metrics = {"loss": 0.1 * epoch, "accuracy": 0.5 + 0.05 * epoch}
    writer.add_scalars(metrics, global_step=epoch, prefix="train")

    # WandB-style convenience API
    writer.log({"train/loss": metrics["loss"]}, step=epoch)

    # Log entire models to track parameter and gradient histograms
    writer.add_models(
        {"encoder": encoder, "decoder": decoder},
        global_step=epoch,
        log_type="all",  # "parameters", "gradients", or "all"
    )

writer.close()
```

CSV logging is enabled when `save_as_csv=True`, and calling `output_catcher=True` mirrors everything printed to the console into `log.txt` within the run folder.
When a Git repository is detected, `git-info.json` is written alongside the logs with the remote URL, branch, and commit so experiments can be reproduced.

### Optional integrations

```python
writer.with_wandb(group="experiments", entity="my-team")
writer.with_neptune(entity="my-workspace")
```

Install the matching extras first (for example `poetry add atalaya[wandb]` or `pip install atalaya[neptune]`).

### Timing utilities

```python
from atalaya.time_manager import Timer

timer = Timer("training")

with timer:
    run_training_loop()

timer.report(report_type="total_with_stats")  # prints coloured summary to the terminal
```

### Terminal helper

```python
from atalaya.terminal import terminal

# Optional: persist terminal output without a Writer output catcher
terminal.set_log_file("logs/terminal.log")

# Override colours either by name or raw ANSI code
terminal.set_named_color("orange", "\033[38;5;208m")
terminal.set_color("warning", "orange")

terminal.print_info("Loading data...")
terminal.print_warning("Loss is plateauing...", color="orange")
terminal.print_ok("Training finished successfully!")
```

Messages are timestamped with the process uptime by default.
When you enable `Writer(..., output_catcher=True)` the log file is handled automatically, so calling `set_log_file` is not required.

### Example project

The `example/` directory ships with a small PyTorch multi-layer perceptron that demonstrates writer usage end-to-end:

```bash
python example/mlp.py
```

Logs are saved in `example/logs/`, and the script shows how to combine `Writer`, `Timer`, and the `terminal` helper in a real training loop.

## Development and Publishing

This project is managed with [Poetry](https://python-poetry.org/).

```bash
# Install dependencies (add --with dev to use Poe tasks)
poetry install --with dev

# Clean previous builds and publish to PyPI
poe publish

# Or run the Poetry command directly
# poetry publish --build
```

The package metadata (version, dependencies, classifiers) lives in `pyproject.toml`.
