import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


def _run_git_command(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class GitInfo:
    path: str | None = None
    remote_url: str | None = None
    branch: str | None = None
    commit: str | None = None
    status: str | None = None

    @classmethod
    def collect(cls, start_path: str | Path | None = None) -> "GitInfo | None":
        root = cls._find_repo_root(start_path)
        if root is None:
            return None

        remote = _run_git_command(["config", "--get", "remote.origin.url"], root)
        branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], root)
        commit = _run_git_command(["rev-parse", "HEAD"], root)
        status = _run_git_command(["status", "--short"], root)

        return cls(
            path=str(root),
            remote_url=remote,
            branch=branch,
            commit=commit,
            status=status,
        )

    @staticmethod
    def _find_repo_root(start_path: str | Path | None = None) -> Path | None:
        start = Path(start_path or Path.cwd()).resolve()
        for path in [start, *start.parents]:
            if (path / ".git").exists():
                return path
        return None

    def dump(self, destination: str | Path) -> None:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("w") as fh:
            json.dump(self._dict_without_none(), fh, indent=2)

    def _dict_without_none(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}
