"""``atalaya`` command line.

    atalaya pull user@cluster:~/project/logs ./logs [--with-events] [--dry-run]
    atalaya upload pluto ./logs/run [--host H]
"""

import argparse
import sys

from . import pull, upload


def main(argv=None):
    parser = argparse.ArgumentParser(prog="atalaya")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="copy runs from a remote machine without their large files")
    p_pull.add_argument("source")
    p_pull.add_argument("destination")
    p_pull.add_argument("--with-events", action="store_true")
    p_pull.add_argument("--include", action="append", default=[], metavar="PATTERN")
    p_pull.add_argument("--dry-run", action="store_true")
    p_pull.add_argument("--no-compress", action="store_true")

    p_upload = sub.add_parser("upload", help="upload a finished run to a tracker from its local files")
    upload.add_arguments(p_upload.add_subparsers(dest="tracker", required=True))

    args = parser.parse_args(argv)
    if args.command == "pull":
        pull.pull(args.source, args.destination, args.with_events, args.include, args.dry_run, not args.no_compress)
        return 0
    return upload.run(args)


if __name__ == "__main__":
    sys.exit(main())
