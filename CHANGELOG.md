# Changelog

## 0.3.0

### Breaking changes

- `atalaya.time_manager` (`Timer`, `TimeManager`) is removed. Use `atalaya.timer.Timer`: one timer holds named sections, used as `with timer("x"):`, `@timer("x")`, `timer.start("x")` / `timer.stop("x")`, or `with Timer("x") as t:`. `report(report_type=...)` is now `report()`; statistics come from `stats()`, `flat()` and `summary()`.
- `Writer.with_neptune` and the `neptune` extra are removed: Neptune's hosted service shut down in March 2026.
- `output_catcher=True`: stderr now stays on stderr (it was sent to stdout); both still go to `log.txt`.
- `add_scalars("tag", {"a": ..., "b": ...}, step)` with a tag string first now has tensorboardX's meaning (several curves on one chart). `add_scalars({...}, step)` with a dict first logs each entry, as before; `log_scalars` is its explicit name.
- `info.json` `git.checkout_command` uses `git switch --detach <commit>` instead of `git reset --hard`.
- `format_time` zero-pads hours, minutes and seconds (`01:02:05`, was `1:2:5`).

### New defaults

Each can be turned off with the option in brackets.

- A SIGTERM handler closes the writer before the process ends (status `terminated`), or flushes it and calls the program's own handler if it has one (`handle_sigterm=False`).
- An uncaught exception is recorded in `info.json` (`capture_exceptions=False`).
- `requirements.txt` lists the installed packages (`save_requirements=False`).
- `info.json` records hostname, SLURM variables, `CUDA_VISIBLE_DEVICES`, torch/CUDA/cuDNN versions when torch is imported, and GPUs from `nvidia-smi` (`log_environment=False`).
- Uncommitted changes are saved to `git.diff`, and the replay commands apply it (`save_git_diff=False`).
- CSV rows are also written every 30 seconds, not only every 10 rows (`csv_flush_secs=None`).

### Added

- `atl.init()`, `atl.log()`, `atl.finish()`: a current run, like `wandb.init` / `wandb.log`, shared by every module of the process; `atl.run`, `atl.config`, `atl.timer("x")`, `atl.log_timings()`. `Writer` keeps working on its own.
- `import atalaya as atl` exposes the API (`atl.Writer`, `atl.Timer`, `atl.read_scalars`, `atl.terminal`, ...), imported lazily.
- `info.json` records how the run ended: `status` (`finished`, `failed`, `interrupted`, `terminated`, `exited_without_close`), end time, duration, last step and the traceback of an uncaught exception. The writer is a context manager and is closed at exit if `close()` was not called.
- `info.json` `integrations`: each tracker's state, and when it could not reach its server a `note` and the `command` that uploads the run later.
- `Writer.with_wandb(config=, init_timeout=, fallback_offline=True)`: continues offline when `wandb.init` fails.
- `Writer.with_pluto()` (extra `atalaya[pluto]`), with a local fallback and `atalaya upload pluto <logdir>` to replay `data.csv` into Pluto afterwards.
- `Writer.with_clearml(fallback_offline=True)`: continues in ClearML's offline mode when the server cannot be reached.
- `Writer(config=...)`: written to `config.json` and passed to W&B and Pluto.
- `Writer.create_timer()`, `attach_timer()`, `log_timings()`: section timings in the logs and in `timings.json`. `Timer(cuda_sync=True)` synchronises CUDA at section boundaries.
- `Writer(async_writes=True)`: logging calls run in order on one background thread; `writer.wait()`.
- `add_video` encodes GIFs with ffmpeg when it is installed (`fast_video=False` for tensorboardX's own path).
- `Writer(csv_compress=True)` writes `data.csv.gz`.
- `atalaya.read`: `read_scalars(logdir, wide=False)`, `read_info`, `read_config`, `read_timings`.
- `atalaya` command: `atalaya pull user@host:logs ./logs` copies only the small per-run files over rsync; `atalaya upload pluto <logdir>`.
- `git_root=` to point at the repository when it cannot be found from the entry script.
- `flush_on_log=False` leaves event-file flushing to tensorboardX's `flush_secs`.
- Tests (`poe test`) and a GitHub Actions workflow for Python 3.10–3.13.

### Fixed

- `with_comet` raised `TypeError`.
- New writer options did not override previous ones when the writer was re-created.
- `with_wandb()` / `with_comet()` after logging deleted the event file; it is kept, with a warning. Re-creating the writer within the same second no longer truncates the kept file.
- A resumed run repeated the CSV header in the middle of `data.csv`.
- Scalars logged with `add_scalar` were missing from `data.csv`.
- The ClearML task was never closed, so its offline session was never written.
- `OutputCatcher` replaced `sys.stdout` with an object without `isatty`, `fileno` or `encoding`, which broke tqdm, rich and subprocesses.
