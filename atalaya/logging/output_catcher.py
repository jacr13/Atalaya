import io
import sys
import threading
from datetime import datetime
from pathlib import Path


class _Tee(io.TextIOBase):
    """Text stream that writes to the original stream and to a shared log file.

    Everything a stream consumer may ask about (isatty, fileno, encoding, ...)
    is answered by the original stream, so tqdm, rich, faulthandler and
    subprocess(stdout=sys.stdout) keep working.
    """

    def __init__(self, original, catcher):
        self._original = original
        self._catcher = catcher

    def write(self, data):
        n = self._original.write(data)
        self._catcher._write_to_file(data)
        return len(data) if n is None else n

    def flush(self):
        self._original.flush()
        self._catcher._flush_file()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._original, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._original, "errors", "strict")

    def __getattr__(self, name):  # buffer, mode, name, reconfigure, ...
        return getattr(self._original, name)


class OutputCatcher:
    """Mirror what the program prints into ``<logdir>/<filename>``.

    stdout keeps going to stdout and stderr to stderr; both are also appended
    to the log file, with a timestamp at the start of each line when
    ``with_time`` is set. Only output written through Python's ``sys.stdout``
    / ``sys.stderr`` is caught: output from C extensions or child processes
    writing to the file descriptors directly is not.
    """

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
        self._lock = threading.Lock()
        self._at_line_start = True
        self._file = self.log_filename.open("a", encoding="utf-8")

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        if catch_stdout:
            sys.stdout = _Tee(self.original_stdout, self)
        if catch_errors:
            sys.stderr = _Tee(self.original_stderr, self)
        self.running = True

    def _write_to_file(self, data):
        if not data:
            return
        with self._lock:
            if self._file.closed:
                return
            if self._with_time:
                stamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                lines = data.splitlines(keepends=True)
                out = []
                for line in lines:
                    if self._at_line_start:
                        out.append(stamp)
                    out.append(line)
                    self._at_line_start = line.endswith("\n")
                data = "".join(out)
            self._file.write(data)
            self._file.flush()

    def _flush_file(self):
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if isinstance(sys.stdout, _Tee) and sys.stdout._catcher is self:
            sys.stdout = self.original_stdout
        if isinstance(sys.stderr, _Tee) and sys.stderr._catcher is self:
            sys.stderr = self.original_stderr
        with self._lock:
            self._file.close()
