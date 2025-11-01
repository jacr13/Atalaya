import csv
from pathlib import Path
from time import time


class CSVLogger:
    def __init__(
        self,
        logdir: str,
        filename: str = "data.csv",
        flush_interval: int = 10,
        initial_time: float | None = None,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.logdir / filename

        self._initial_time = time() if initial_time is None else initial_time
        self._flush_interval = flush_interval
        self._lines_to_write = [
            ["global_step", "tag", "value", "walltime", "relative_time"]
        ]

    def add(
        self,
        tag: str,
        value,
        global_step: int,
        walltime: float,
        initial_time: float | None = None,
    ):
        start_time = initial_time or self._initial_time
        relative_time = walltime - start_time
        self._lines_to_write.append(
            [global_step, tag, value, walltime, relative_time]
        )

        if len(self._lines_to_write) >= self._flush_interval:
            self.flush()

    def flush(self):
        with self.filepath.open("a+") as file:
            csv_writer = csv.writer(file)
            for line in self._lines_to_write:
                csv_writer.writerow(line)

        self._lines_to_write = []
