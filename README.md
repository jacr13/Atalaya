# Atalaya

[![PyPI version](https://img.shields.io/pypi/v/atalaya.svg)](https://pypi.org/project/atalaya/)

Atalaya is a lightweight toolkit that helps you keep PyTorch experiments organised. It provides:

- a `Writer` built on top of [tensorboardX](https://github.com/lanpa/tensorboardX) with handy helpers for scalars, models, CSV exports, and optional integrations with [Weights & Biases](https://forge.coreweave.com/wandb), [Pluto](https://github.com/trainy-ai/pluto), [Comet](https://www.comet.com), and [ClearML](https://clear.ml);
- automatic `info.json` metadata snapshots (command, Python version, host, SLURM job, Git details, and how the run ended) plus a `git.diff` of uncommitted changes, so you can replay runs later;
- a colour-aware `terminal` helper for structured CLI output with timestamps;
- a `Timer` for named code sections (context manager, decorator or start/stop) whose statistics go to the logs and to `timings.json`;
- optional console log capture so that your scripts remain reproducible.

The writer uses `tensorboardX` as its event backend, so you get native TensorBoard support while still being able to plug in third-party experiment trackers such as Weights & Biases, Pluto, ClearML, or Comet.

| Integration | Install | Status |
|---|---|---|
| [TensorBoardX](https://github.com/lanpa/tensorboardX) | included | Works (core backend) |
| [Weights & Biases](https://forge.coreweave.com/wandb) | `atalaya[wandb]` | Works (`Writer.with_wandb`) |
| [Pluto](https://github.com/trainy-ai/pluto) | `atalaya[pluto]` | Not validated against a server yet (`Writer.with_pluto`) |
| [ClearML](https://clear.ml) | `atalaya[clearml]` | Not fully validated yet (`Writer.with_clearml`) |
| [Comet ML](https://www.comet.com) | `atalaya[comet]` | Not fully validated yet (`Writer.with_comet`) |

## Installation

Install the core package from PyPI:

```bash
pip install atalaya
```

Extras are available if you want to enable third-party integrations:

```bash
# Enable Weights & Biases support
pip install atalaya[wandb]

# Install everything (Pluto, Comet, ClearML, matplotlib, seaborn)
pip install atalaya[all]
```

## Quick Start

There are two ways to log, and they write the same files: a current run, like `wandb.init` / `wandb.log`, or a `Writer` object you pass around. Everything is available from the package: `import atalaya as atl` gives `atl.init`, `atl.log`, `atl.Writer`, `atl.Timer`, `atl.read_scalars`, `atl.terminal`, ...

### Current run: `atl.init` / `atl.log`

```python
import atalaya as atl

atl.init(project="mnist", config={"lr": 1e-3})   # logs/<script>_<date>/, data.csv on

for epoch in range(10):
    with atl.timer("train"):                      # or @atl.timer("train") on a function
        loss = train_one_epoch()
    atl.log({"loss": loss})                       # steps 0, 1, 2, ... unless step= is given

atl.finish()                                      # info.json, timings.json, flushed data.csv
```

`atl.log` and `atl.timer` work from any module of the same process: a data module or a model file just does `import atalaya as atl` and logs into the run the training script started. `@atl.timer(...)` can decorate functions at import time, before `atl.init()`; without an active run it times nothing. `atl.run` is the current run (a `Writer`, or `None`), so everything below is available on it, e.g. `atl.run.with_wandb()` or `atl.run.add_video(...)`; `atl.config` is its config. `atl.init()` accepts every `Writer` option. Other processes (DataLoader workers, multiprocessing) do not share the run.

### Writer object

The same, without a global run — useful for several runs at once or library code that receives its writer:

```python
import atalaya as atl

writer = atl.Writer(
    name="baseline",
    project="my-awesome-project",
    logdir="logs",
    add_time=True,
    save_as_csv=True,
    output_catcher=True,
    log_git_info=True,
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
Every run writes an `info.json` file with the launch command, Python version, and other metadata in the log directory. When `log_git_info` is enabled and a Git repository is detected, the same file also includes Git details plus ready-to-run checkout/clone commands so experiments can be reproduced. You can merge your own metadata into the file via the `extra_info` parameter (for example `extra_info={"dataset": "cifar10"}`).

When the tree has uncommitted changes, they are saved to `git.diff` next to `info.json` (`save_git_diff=True`), and the replay commands apply it. Untracked files are listed in `info.json` but not saved. Pass `git_root=` when the repository cannot be found from the entry script.

When the run ends, `info.json` is updated with `status` (`finished`, `failed`, `interrupted` or `exited_without_close`), `end_time_utc`, `duration_seconds`, `last_step` and, for an uncaught exception, its traceback. Use the writer as a context manager (`with Writer(...) as writer:`) or call `close()`; otherwise it is closed at interpreter exit.

`add_scalars` keeps tensorboardX's meaning when its first argument is a tag string (`add_scalars("loss", {"train": a, "val": b}, step)`); with a dict first it logs each entry as its own scalar, like `log_scalars`. Every scalar goes to `data.csv`, including ones logged with `add_scalar`. `flush_on_log=False` leaves event-file flushing to tensorboardX's `flush_secs`.

### Optional integrations

```python
writer.with_wandb(group="experiments", entity="my-team", config=vars(args))
# or Pluto (pip install atalaya[pluto]); host= for a self-hosted server
writer.with_pluto(config=vars(args), host="pluto.my-lab.org")
```

Install the matching extras first (for example `poetry add atalaya[wandb]`).

Call integrations before logging anything: they re-create the event file. If `wandb.init` fails (no network, a proxy error, a timeout), `with_wandb` continues in offline mode by default (`fallback_offline=True`); local logging never depends on W&B. `with_clearml` does the same with ClearML's offline mode. Pluto has no offline mode (it must create the run on its server), so if `with_pluto` cannot reach it the run continues with local logs only, `data.csv` is switched on, and the recorded command is `python -m atalaya.upload pluto <logdir>`, which replays `data.csv` into Pluto with the original steps and times. Each integration's state is recorded in `info.json` under `integrations`, and when a run was logged offline the entry carries a `note` and the `command` to upload it later:

```json
"integrations": {
  "wandb": {
    "status": "offline_fallback",
    "error": "CommError: ...",
    "note": "wandb.init failed, so this run was logged to W&B offline. Local logs are complete. Run the command to upload it.",
    "command": "wandb sync logs/wandb/offline-run-20261110_120000-baseline"
  }
}
```

Notes on Pluto (`pluto-ml`): it forwards scalars only (histograms, images and videos stay in TensorBoard); credentials come from `PLUTO_API_KEY` or `pluto login`; the SDK sends anonymous error telemetry to Sentry unless `PLUTO_DISABLE_TELEMETRY=1`; and installing `pluto-ml` adds a `.pth` startup hook that, once you are logged in to Pluto, patches `import wandb` in that environment so W&B runs also log to Pluto. Use `with_wandb` or `with_pluto`, not both, in such an environment, or every scalar reaches Pluto twice.

### Run directory and options

A run directory holds the event file plus `info.json`, `git.diff` (dirty tree), `config.json` (`config=`), `requirements.txt` (installed packages, `save_requirements=True`) and `data.csv` (`save_as_csv=True`). `info.json` also records the torch/CUDA/cuDNN versions (only if your program imported torch) and the GPUs reported by `nvidia-smi` (`log_environment=True`).

| Option | Default | Effect |
|---|---|---|
| `config=` | `None` | saved to `config.json`, and passed to W&B / Pluto unless they get their own |
| `csv_compress=` | `False` | write `data.csv.gz` instead (about 9x smaller; readable by `gzip`, `zcat`, pandas) |
| `csv_flush_secs=` | `30` | also write CSV rows every N seconds, so a killed run loses at most that much |
| `handle_sigterm=` | `True` | on SIGTERM (SLURM sends it before SIGKILL) close the writer, status `terminated`; if your program has its own handler, flush and call it instead |
| `async_writes=` | `False` | run logging calls on one background thread, in order; `writer.wait()` blocks until they are written |
| `fast_video=` | `True` | `add_video` encodes GIFs with ffmpeg when installed (no moviepy, much faster) |

`OutputCatcher` (`output_catcher=True`) mirrors stdout and stderr to `log.txt` while they keep going to the terminal, so tqdm, rich and subprocesses behave as before.

### Reading runs back and copying them from a cluster

```python
import atalaya as atl

df = atl.read_scalars("logs/run")              # long format, one row per (step, tag)
wide = atl.read_scalars("logs/run", wide=True)  # one row per step, one column per tag
```

```bash
# rsync only the small per-run files (data.csv, info.json, config.json, git.diff, ...)
atalaya pull user@cluster:~/project/logs ./logs
atalaya pull user@cluster:~/project/logs ./logs --with-events --include 'scores.json'

# upload a run that could not reach Pluto during training
atalaya upload pluto ./logs/run
```

### Timing utilities

```python
import atalaya as atl

timer = writer.create_timer()            # timings.json written when the writer closes
                                         # create_timer(cuda_sync=True) for trustworthy GPU splits
                                         # or attach an existing one: writer.attach_timer(timer)

with timer("train/update_actor"):        # named section, as a context manager
    ...

@timer("rollout")                        # same section object, as a decorator
def rollout(...): ...

timer.start("eval"); ...; timer.stop("eval")   # when a `with` doesn't fit the code

if epoch % 50 == 0:                      # share of time and mean duration per section
    writer.log_timings(timer, global_step=epoch, steps=agent_steps)

with atl.Timer("training") as t:         # a single block
    train()
t.report()                               # table printed to the terminal
```

`timer.stats(steps=...)` returns, per section, the interval since the previous call (`count`, `sum`, `mean`, `min`, `max`, `frac`, `rate`) and the whole run (`total`, `total_frac`, `total_rate`); `summary()` gives the cumulative view that goes to `timings.json`. A section that raises is still stopped and recorded, sections can nest or recurse, and memory stays constant. GPU work is asynchronous: without `cuda_sync=True` a section's time can include work launched by an earlier one.

### Terminal helper

```python
import atalaya as atl

terminal = atl.terminal

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

Logs are saved in `example/logs/`, and the script shows how to combine `Writer`, `Timer` (with `attach_timer` and `log_timings`), and the `terminal` helper in a real training loop.

## Development and Publishing

This project is managed with [Poetry](https://python-poetry.org/).

```bash
# Run the tests (also run by GitHub Actions on every push)
poetry install --with dev && poe test

# Install dependencies (add --with dev to use Poe tasks)
poetry install --with dev

# Clean previous builds and publish to PyPI
poe publish

# Or run the Poetry command directly
# poetry publish --build
```

The package metadata (version, dependencies, classifiers) lives in `pyproject.toml`.
