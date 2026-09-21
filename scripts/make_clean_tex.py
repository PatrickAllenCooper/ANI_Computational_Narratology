"""
make_clean_tex.py -- write the comment-free `*_clean.tex` copy of each paper file.

Strips every comment that begins with an UNESCAPED `%` (a `\%` is a literal
percent sign and is kept), drops lines that were comment-only, and trims the
trailing whitespace a stripped comment leaves behind. Idempotent.

  python -m scripts.make_clean_tex papers/embodied_sensor/*.tex
  python -m scripts.make_clean_tex --check papers/embodied_sensor/methods.tex   # report only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COMMENT = re.compile(r"(?<!\\)%.*$")


def clean_text(src: str) -> str:
    out = []
    for line in src.splitlines():
        stripped = COMMENT.sub("", line).rstrip()
        if line.strip().startswith("%") and not line.strip().startswith("\\%"):
            continue                       # comment-only line
        if stripped == "" and line.strip() != "":
            continue                       # line was only a trailing comment
        out.append(stripped)
    text = "\n".join(out).lstrip("\n") + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    changed = 0
    for f in a.files:
        if f.name.endswith("_clean.tex"):
            continue
        dst = f.with_name(f.stem + "_clean.tex")
        new = clean_text(f.read_text())
        old = dst.read_text() if dst.exists() else None
        if new != old:
            changed += 1
            print(("would write " if a.check else "wrote ") + str(dst))
            if not a.check:
                dst.write_text(new)
    print(f"{changed} file(s) {'differ' if a.check else 'written'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
