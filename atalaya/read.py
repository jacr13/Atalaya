"""Read a run back from its directory, without TensorBoard.

    from atalaya.read import read_info, read_scalars, read_timings

    df = read_scalars("logs/run")             # long: global_step, tag, value, walltime, relative_time
    wide = read_scalars("logs/run", wide=True)  # one row per step, one column per tag
    info = read_info("logs/run")
    timings = read_timings("logs/run")   # Writer.attach_timer

Works on data.csv and data.csv.gz, including files from atalaya <= 0.2.8 that
repeat the header after a resume.
"""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

CSV_NAMES = ("data.csv", "data.csv.gz")


def find_csv(logdir) -> Path:
    d = Path(logdir)
    for name in CSV_NAMES:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"no data.csv or data.csv.gz in {d} (was it logged with save_as_csv?)")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def iter_rows(logdir, tags=None):
    """Yield (step, tag, value, walltime) for every numeric row."""
    wanted = set(tags) if tags is not None else None
    with _open_text(find_csv(logdir)) as f:
        for row in csv.DictReader(f):
            if row["global_step"] == "global_step":
                continue  # repeated header (atalaya <= 0.2.8, resumed run)
            if wanted is not None and row["tag"] not in wanted:
                continue
            try:
                value = float(row["value"])
            except ValueError:
                continue
            raw = row["global_step"]
            step = int(float(raw)) if raw not in ("", "None") else None
            yield step, row["tag"], value, float(row["walltime"])


def read_scalars(logdir, tags=None, wide=False):
    """Scalars as a pandas DataFrame (pandas is imported only here)."""
    import pandas as pd

    rows = list(iter_rows(logdir, tags))
    df = pd.DataFrame(rows, columns=["global_step", "tag", "value", "walltime"])
    if not wide:
        start = df["walltime"].min() if len(df) else 0.0
        return df.assign(relative_time=df["walltime"] - start)
    # last value wins if a tag was logged twice at one step
    return df.pivot_table(index="global_step", columns="tag", values="value", aggfunc="last").sort_index()


def read_scalars_by_step(logdir, tags=None):
    """-> ({step: {tag: value}}, {step: walltime}), without pandas."""
    by_step = defaultdict(dict)
    walltimes = {}
    for step, tag, value, walltime in iter_rows(logdir, tags):
        by_step[step][tag] = value
        walltimes[step] = walltime
    return by_step, walltimes


def read_info(logdir) -> dict:
    return json.loads((Path(logdir) / "info.json").read_text())


def read_config(logdir) -> dict | None:
    path = Path(logdir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_timings(logdir, name="timings.json") -> dict | None:
    path = Path(logdir) / name
    return json.loads(path.read_text()) if path.exists() else None
