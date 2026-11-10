"""Upload a finished run to a tracker from its local files.

    python -m atalaya.upload pluto <logdir> [--project P] [--name N] [--host H]

Reads <logdir>/data.csv or data.csv.gz (every scalar, as written by ``Writer(save_as_csv=True)``),
info.json (run name and project) and config.json if present, and replays the
scalars with their original steps and wall-clock times. This is the command
info.json points to when a tracker could not be reached during the run.
Histograms, images and videos are not uploaded: they live in the event file only.
"""

import argparse
import json
import sys
from pathlib import Path

from .read import find_csv, read_config, read_scalars_by_step


def _write_json(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def upload_pluto(logdir, project=None, name=None, host=None):
    """Create a Pluto run for ``logdir`` and replay its data.csv into it."""
    import pluto

    logdir = Path(logdir)
    find_csv(logdir)  # fail early, before creating a remote run
    info_path = logdir / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    config = read_config(logdir)

    by_step, walltimes = read_scalars_by_step(logdir)
    run = pluto.init(
        dir=str(logdir),
        project=project or info.get("project"),
        name=name or info.get("run_name"),
        config=config,
        settings={"host": host} if host else None,
    )
    steps = sorted(by_step, key=lambda s: (s is None, s if s is not None else 0))
    for step in steps:
        run.log(by_step[step], step=step, timestamp=walltimes[step])
    run.finish()

    if info_path.exists():
        entry = info.setdefault("integrations", {}).setdefault("pluto", {})
        entry.update(status="uploaded", id=getattr(run, "id", None), uploaded_steps=len(steps))
        _write_json(info_path, info)
    return run, len(steps)


def add_arguments(sub):
    p = sub.add_parser("pluto", help="replay data.csv into a new Pluto run")
    p.add_argument("logdir")
    p.add_argument("--project", help="default: project in info.json")
    p.add_argument("--name", help="default: run_name in info.json")
    p.add_argument("--host", help="self-hosted Pluto server; default: Pluto's cloud")


def run(args):
    if args.tracker == "pluto":
        _, n = upload_pluto(args.logdir, project=args.project, name=args.name, host=args.host)
        print(f"uploaded {n} steps from {find_csv(args.logdir)} to Pluto")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m atalaya.upload", description=__doc__.split("\n\n")[0])
    add_arguments(parser.add_subparsers(dest="tracker", required=True))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
