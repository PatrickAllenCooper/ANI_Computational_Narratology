#!/usr/bin/env python3
"""Export the ICLR 2027 paper (papers/iclr2027) into the Overleaf project's flat layout.

The Overleaf project (NoT_ICLR_Submission.zip, 2026-09-24) is flat: the main document is
iclr2027_conference.tex, the bibliography iclr2027_conference.bib, the template files sit beside
them, and not_iclr.tex is a second, template-only document that is carried over unchanged.

This script reads the paper from a git ref (default HEAD, so a half-edited working tree is never
exported), inlines every \\input{sections/...} and \\input{figures/...} into one
iclr2027_conference.tex (math_commands.tex stays an \\input, as in the template), points
\\bibliography at iclr2027_conference, and writes iclr2027_conference.bib as our references.bib
plus any entry of the project's own bib whose key we lack (the template's example entries, which
not_iclr.tex cites), never the key anon24parent (PI order, 2026-09-23: no reference to the
authors' preprint). Template files and not_iclr.tex come from the extracted Overleaf project.

Usage:
  python3 scripts/export_overleaf.py --project /path/to/extracted_overleaf --out /path/to/out_dir [--ref HEAD] [--zip out.zip]
  python3 scripts/export_overleaf.py --selftest
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = "papers/iclr2027"
TEMPLATE_FILES = ("iclr2027_conference.sty", "iclr2027_conference.bst", "fancyhdr.sty", "natbib.sty",
                  "math_commands.tex")
CARRY_UNCHANGED = ("not_iclr.tex",)
DROP_KEYS = {"anon24parent"}
INPUT_RE = re.compile(r"\\input\{((?:sections|figures)/[^}]+)\}")


def read(ref: str | None, rel: str) -> str:
    if ref is None:
        return (REPO / PAPER / rel).read_text()
    return subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{PAPER}/{rel}"],
                          check=True, capture_output=True, text=True).stdout


def _code_part(line: str) -> str:
    """The part of a line before its first unescaped %."""
    m = re.search(r"(?<!\\)%", line)
    return line if m is None else line[:m.start()]


def flatten(text: str, ref: str | None, depth: int = 0) -> str:
    if depth > 5:
        raise RuntimeError("\\input nesting deeper than 5")
    out = []
    for line in text.split("\n"):
        code = _code_part(line)
        if not INPUT_RE.search(code):
            out.append(line)
            continue
        pieces, pos = [], 0
        for m in INPUT_RE.finditer(code):
            rel = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
            body = flatten(read(ref, rel), ref, depth + 1).rstrip("\n")
            pieces.append(code[pos:m.start()])
            pieces.append(f"%%% begin {rel}\n{body}\n%%% end {rel}\n")
            pos = m.end()
        pieces.append(line[pos:])
        out.append("".join(pieces))
    return "\n".join(out)


def bib_entries(text: str) -> dict[str, str]:
    """key -> entry text (from @type{key, to its closing brace at depth 0)."""
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        i, depth = m.end() - 1, 0
        j = text.index("{", m.start())
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    out.setdefault(m.group(2), text[m.start():k + 1])
                    break
    return out


def merge_bib(ours: str, theirs: str) -> tuple[str, list[str]]:
    o, t = bib_entries(ours), bib_entries(theirs)
    extra = [k for k in t if k not in o and k not in DROP_KEYS]
    kept = ours
    for k in DROP_KEYS & set(o):
        kept = kept.replace(o[k], f"% {k} removed on export (PI, 2026-09-23)")
    if extra:
        kept = kept.rstrip() + "\n\n% --- entries carried over from the Overleaf project's own bibliography ---\n\n" + \
            "\n\n".join(t[k] for k in extra) + "\n"
    return kept, extra


def export(project: Path, out: Path, ref: str | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    main = flatten(read(ref, "main.tex"), ref)
    main = main.replace("\\bibliography{references}", "\\bibliography{iclr2027_conference}")
    assert "\\bibliography{iclr2027_conference}" in main
    assert not INPUT_RE.search("\n".join(_code_part(l) for l in main.split("\n"))), "unflattened \\input left"
    assert "anon24parent" not in "\n".join(_code_part(l) for l in main.split("\n")), "preprint cited"
    (out / "iclr2027_conference.tex").write_text(main)
    bib, extra = merge_bib(read(ref, "references.bib"), (project / "iclr2027_conference.bib").read_text())
    (out / "iclr2027_conference.bib").write_text(bib)
    for f in TEMPLATE_FILES + CARRY_UNCHANGED:
        shutil.copy2(project / f, out / f)
    return {"main_lines": main.count("\n"), "bib_extra_from_project": extra}


def _selftest() -> int:
    ok = True

    def check(msg, cond):
        nonlocal ok
        print(("[ok] " if cond else "[FAIL] ") + msg)
        ok = ok and cond
    check("comment-only \\input is not expanded", _code_part("% by hand; \\input{figures/x}") == "")
    check("escaped percent is kept", _code_part("80\\% of \\input{x}") == "80\\% of \\input{x}")
    b = "@misc{a,\n title={x {y}}\n}\n@misc{anon24parent,\n title={p}\n}\n"
    t = "@misc{a,\n title={old}\n}\n@misc{Hinton06,\n title={h}\n}\n@misc{anon24parent,\n title={p}\n}\n"
    merged, extra = merge_bib(b, t)
    check("our entry wins on a shared key", "title={x {y}}" in merged and "title={old}" not in merged)
    check("project-only entries are carried", extra == ["Hinton06"] and "title={h}" in merged)
    check("anon24parent is removed", "@misc{anon24parent" not in merged)
    print("\nselftest: " + ("ALL OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, help="extracted Overleaf project directory")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ref", default="HEAD", help="git ref to export; 'WORKTREE' for the working tree")
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    ref = None if a.ref == "WORKTREE" else a.ref
    res = export(a.project, a.out, ref)
    print(f"exported {a.ref} to {a.out}: {res}")
    if a.zip:
        with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(a.out.iterdir()):
                if p.is_file() and p.suffix in (".tex", ".bib", ".sty", ".bst"):
                    z.write(p, p.name)
        print(f"wrote {a.zip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
