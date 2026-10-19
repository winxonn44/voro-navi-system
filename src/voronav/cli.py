"""The `voronav` command. Subcommands:

    voronav demo <sequence-or-frames-dir> [--out docs/q4/demo.gif] ...
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "demo":
        from .demo import main as demo
        demo(rest)
    else:
        raise SystemExit(f"unknown command {cmd!r}; try `voronav --help`")


if __name__ == "__main__":
    main()
