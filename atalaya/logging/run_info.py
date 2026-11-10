import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _clean_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            nested = _clean_dict(value)
            if nested:
                cleaned[key] = nested
        else:
            cleaned[key] = value
    return cleaned


_SLURM_VARS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
)


def collect_run_info(
    *,
    run_name: str,
    project: str,
    logdir: Path,
    source_path: Path | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Gather baseline metadata for a run."""
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    info: dict[str, Any] = {
        "run_name": run_name,
        "project": project,
        "log_directory": str(logdir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "hostname": platform.node() or None,
        "command": command if command else None,
        "script_path": str(source_path) if source_path is not None else None,
    }

    # scheduler context, so a run can be traced back to its job and node
    info["slurm"] = {k: os.environ.get(k) for k in _SLURM_VARS}
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    if extra:
        info.update(extra)

    return _clean_dict(info)
