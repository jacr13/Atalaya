import os
import sys
import threading
from datetime import datetime
from pathlib import Path


class OutputCatcher:
    def __init__(
        self,
        logdir,
        filename: str = "log.txt",
        with_time: bool = False,
        catch_stdout: bool = True,
        catch_errors: bool = False,
    ):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.log_filename = self.logdir / filename
        self._with_time = with_time
        self._catch_stdout = catch_stdout
        self._catch_errors = catch_errors

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.pipe_r, self.pipe_w = os.pipe()
        self.pipe_writer = os.fdopen(self.pipe_w, "w")

        if self._catch_stdout:
            sys.stdout = self
        if self._catch_errors:
            sys.stderr = self

        self.running = True
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()

    def write(self, data):
        self.pipe_writer.write(data)
        self.pipe_writer.flush()

    def flush(self):
        self.pipe_writer.flush()

    def _reader_thread(self):
        pipe_reader = os.fdopen(self.pipe_r, "r")
        with self.log_filename.open("a") as logfile:
            while self.running:
                line = pipe_reader.readline()
                if not line:
                    break
                self.original_stdout.write(line)
                self.original_stdout.flush()

                if self._with_time:
                    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"
                logfile.write(line)
                logfile.flush()

    def stop(self):
        self.running = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.pipe_writer.close()
        self.thread.join()
