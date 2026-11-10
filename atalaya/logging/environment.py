"""Software and hardware context of a run: torch/CUDA, GPUs, installed packages."""

import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def collect_gpus(timeout: float = 5.0) -> list[dict[str, str]] | None:
    """GPUs as nvidia-smi reports them, or None when it is missing or fails.

    Asks nvidia-smi rather than torch.cuda, which would create a CUDA context
    in this process (a problem for code that forks data-loader workers later).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append(dict(zip(("index", "name", "driver", "memory"), parts)))
    return gpus or None


def collect_torch() -> dict[str, Any] | None:
    """torch build info, only if the program already imported torch.

    Importing torch here would cost seconds for programs that do not use it.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    version = getattr(torch, "version", None)
    info = {
        "version": getattr(torch, "__version__", None),
        "cuda": getattr(version, "cuda", None),
    }
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    try:
        info["cudnn"] = cudnn.version() if cudnn is not None else None
    except Exception:
        info["cudnn"] = None
    return info


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {}
    torch_info = collect_torch()
    if torch_info is not None:
        env["torch"] = torch_info
    gpus = collect_gpus()
    if gpus is not None:
        env["gpus"] = gpus
    return env


def installed_packages() -> list[str]:
    """``name==version`` for every distribution importable by this interpreter.

    Read from package metadata, so it works in environments without pip (uv,
    some conda setups) and takes milliseconds instead of a pip subprocess.
    """
    seen = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower() not in seen:
            seen[name.lower()] = f"{name}=={dist.version}"
    return [seen[k] for k in sorted(seen)]


def write_requirements(destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(installed_packages()) + "\n")
    return path
