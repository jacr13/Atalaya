"""Copy runs from a remote machine (cluster) without their large files.

    python -m atalaya.pull user@cluster:~/project/logs ./logs
    python -m atalaya.pull user@cluster:~/project/logs ./logs --with-events --dry-run

rsync with a filter: only the small files atalaya writes per run are copied
(data.csv[.gz], info.json, config.json, timings.json, git.diff,
requirements.txt, log.txt), not TensorBoard event files, checkpoints or
tracker caches, unless asked.
Re-running only transfers what changed.
"""

import argparse
import shutil
import subprocess
import sys

RUN_FILES = (
    "data.csv",
    "data.csv.gz",
    "info.json",
    "config.json",
    "timings.json",
    "git.diff",
    "requirements.txt",
    "log.txt",
)
EVENT_FILES = ("events.out.tfevents.*",)


def rsync_command(source, destination, with_events=False, include=(), dry_run=False, compress=True):
    patterns = list(RUN_FILES) + list(include)
    if with_events:
        patterns += list(EVENT_FILES)
    cmd = ["rsync", "-a", "--prune-empty-dirs"]
    if compress:
        cmd.append("-z")
    if dry_run:
        cmd += ["--dry-run", "-v"]
    cmd += ["--include=*/"]
    cmd += [f"--include={p}" for p in patterns]
    cmd += ["--exclude=*"]
    # trailing slash: copy the contents of source into destination
    cmd += [source.rstrip("/") + "/", destination]
    return cmd


def pull(source, destination, with_events=False, include=(), dry_run=False, compress=True):
    if shutil.which("rsync") is None:
        raise RuntimeError("rsync not found on PATH")
    cmd = rsync_command(source, destination, with_events, include, dry_run, compress)
    return subprocess.run(cmd, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m atalaya.pull", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="remote logdir, e.g. user@host:~/project/logs (or a local path)")
    parser.add_argument("destination", help="local directory")
    parser.add_argument("--with-events", action="store_true", help="also copy TensorBoard event files")
    parser.add_argument("--include", action="append", default=[], metavar="PATTERN",
                        help="extra rsync include pattern, e.g. --include 'scores.json' (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="list what would be copied")
    parser.add_argument("--no-compress", action="store_true", help="do not compress during transfer")
    parser.add_argument("--print", action="store_true", dest="print_only", help="print the rsync command and exit")
    args = parser.parse_args(argv)

    if args.print_only:
        print(" ".join(rsync_command(args.source, args.destination, args.with_events, args.include,
                                     args.dry_run, not args.no_compress)))
        return 0
    pull(args.source, args.destination, args.with_events, args.include, args.dry_run, not args.no_compress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
