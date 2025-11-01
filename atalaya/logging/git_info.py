import json
import re
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
    checkout_command: str | None = None
    clone_command: str | None = None
    run_name: str | None = None

    @classmethod
    def collect(
        cls,
        start_path: str | Path | None = None,
        run_name: str | None = None,
    ) -> "GitInfo | None":
        root = cls._find_repo_root(start_path)
        if root is None:
            return None

        remote = _run_git_command(["config", "--get", "remote.origin.url"], root)
        branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], root)
        commit = _run_git_command(["rev-parse", "HEAD"], root)
        status = _run_git_command(["status", "--short"], root)
        checkout_cmd = None
        clone_cmd = None
        sanitized_run_name = cls._sanitize_branch_name(run_name) if run_name else None
        if commit:
            checkout_cmd = f"git fetch && git checkout {commit}"
            if branch and branch != "HEAD":
                checkout_cmd = (
                    f"git fetch && git checkout {branch} && git reset --hard {commit}"
                )
            clone_cmd = cls._build_clone_command(
                remote, branch, commit, sanitized_run_name
            )

        info = cls(
            path=str(root),
            remote_url=remote,
            branch=branch,
            commit=commit,
            status=status,
            checkout_command=checkout_cmd,
            clone_command=clone_cmd,
        )
        return info

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
            json.dump(self.to_dict(), fh, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @staticmethod
    def _build_clone_command(
        remote: str | None,
        branch: str | None,
        commit: str | None,
        run_branch: str | None,
    ) -> str | None:
        if remote is None or commit is None:
            return None

        repo_dir = GitInfo._extract_repo_dir(remote)
        if run_branch:
            branch_name = run_branch
        elif branch and branch != "HEAD":
            branch_name = f"{branch}-replay"
        else:
            branch_name = f"exp-{commit[:7]}"

        if repo_dir is None:
            # Fallback without explicit directory name
            return (
                f"git clone {remote} && "
                f"cd <repo-dir> && "
                f"git checkout -b {branch_name} {commit}"
            )

        return (
            f"git clone {remote} && "
            f"cd {repo_dir} && "
            f"git checkout -b {branch_name} {commit}"
        )

    @staticmethod
    def _extract_repo_dir(remote: str) -> str | None:
        if remote.endswith(".git"):
            remote = remote[:-4]

        if remote.startswith("git@"):
            path_part = remote.split(":", 1)[-1]
        elif remote.startswith("http://") or remote.startswith("https://"):
            path_part = remote.rstrip("/").split("/", maxsplit=3)[-1]
        else:
            path_part = remote

        if "/" in path_part:
            return path_part.split("/")[-1]
        if path_part:
            return path_part
        return None

    @staticmethod
    def _sanitize_branch_name(name: str) -> str | None:
        sanitized = re.sub(r"\s+", "-", name.strip())
        sanitized = re.sub(r"[^A-Za-z0-9._/-]", "-", sanitized)
        sanitized = re.sub(r"-{2,}", "-", sanitized)
        sanitized = sanitized.strip("-/")
        if not sanitized:
            return None
        # Git branch names cannot end with .lock etc, trim to safe length
        return sanitized[:80]
