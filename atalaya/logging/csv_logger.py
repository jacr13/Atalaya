import csv
import gzip
import io
import threading
from pathlib import Path
from time import time

HEADER = ["global_step", "tag", "value", "walltime", "relative_time"]


class CSVLogger:
    """Scalars in long format: one row per (step, tag).

    Rows are buffered and appended when ``flush_interval`` rows are waiting or
    ``flush_secs`` seconds have passed since the last write, whichever comes
    first, so a run killed without warning loses at most that much.
    Appending to an existing file (a resumed run) does not repeat the header.

    With ``compress=True`` the file is ``data.csv.gz``. Each flush appends a
    gzip member; concatenated members are one valid gzip stream, which
    ``gzip``, ``zcat`` and ``pandas.read_csv`` read as a single file.
    """

    def __init__(
        self,
        logdir: str,
        filename: str | None = None,
        flush_interval: int = 10,
        flush_secs: float | None = 30.0,
        initial_time: float | None = None,
        compress: bool = False,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        if filename is None:
            filename = "data.csv.gz" if compress else "data.csv"
        self.filepath = self.logdir / filename

        self._initial_time = time() if initial_time is None else initial_time
        self._flush_interval = flush_interval
        self._flush_secs = flush_secs
        self._last_flush = time()
        self._lines_to_write = []
        self._lock = threading.Lock()  # add() may come from a writer thread
        self._needs_header = (
            not self.filepath.exists() or self.filepath.stat().st_size == 0
        )

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
        with self._lock:
            self._lines_to_write.append([global_step, tag, value, walltime, relative_time])
            due = len(self._lines_to_write) >= self._flush_interval or (
                self._flush_secs is not None and time() - self._last_flush >= self._flush_secs
            )
        if due:
            self.flush()

    def flush(self):
        with self._lock:
            lines, self._lines_to_write = self._lines_to_write, []
            self._last_flush = time()
            if not lines:
                return
            buffer = io.StringIO()
            csv_writer = csv.writer(buffer)
            if self._needs_header:
                csv_writer.writerow(HEADER)
                self._needs_header = False
            csv_writer.writerows(lines)
            data = buffer.getvalue()
            if self.compress:
                with self.filepath.open("ab") as file:
                    file.write(gzip.compress(data.encode()))
            else:
                with self.filepath.open("a", newline="") as file:
                    file.write(data)
