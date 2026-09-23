#!/usr/bin/env python3
"""Write comment-free copies of the ICLR 2027 draft.

For every sections/X.tex (and figures/X.tex, which the sections \\input) write X_clean.tex with all
LaTeX % comments removed, keeping escaped \\%. Removal is TeX-equivalent:
  * a line holding only a comment is dropped (TeX discards it, end of line included);
  * "text % note" keeps "text" and its end of line (TeX yields one space either way);
  * "text% note" (no space before %) is joined to the next line with its leading blanks stripped,
    which is what TeX does, except that after a control word a single space is kept so that
    "\\foo%" + "bar" does not become "\\foobar";
  * a %-ended line followed by a blank line keeps the blank line (the paragraph break).
In the _clean copies, \\input{sections/X} and \\input{figures/X} point to the _clean files.
main_clean.tex is main.tex, comment-free, inputting the _clean sections.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def comment_start(line):
    """Index of the first % that starts a comment (preceded by an even number of backslashes)."""
    for i, ch in enumerate(line):
        if ch == '%':
            n = 0
            j = i - 1
            while j >= 0 and line[j] == '\\':
                n += 1
                j -= 1
            if n % 2 == 0:
                return i
    return -1


def strip_comments(text):
    out = []
    join_pending = False  # previous emitted line ended in a comment with no space before it
    for line in text.split('\n'):
        idx = comment_start(line)
        if idx >= 0:
            before = line[:idx]
            if before.strip() == '':
                continue  # comment-only line; TeX discards it entirely
            if join_pending:
                prev = out[-1]
                sep = ' ' if re.search(r'\\[A-Za-z@]+$', prev) else ''
                out[-1] = prev + sep + before.lstrip()
            else:
                out.append(before)
            if before[-1] in ' \t':
                out[-1] = out[-1].rstrip()
                join_pending = False
            else:
                join_pending = True
        else:
            if join_pending:
                if line.strip() == '':
                    out.append('')
                else:
                    prev = out[-1]
                    sep = ' ' if re.search(r'\\[A-Za-z@]+$', prev) else ''
                    out[-1] = prev + sep + line.lstrip()
                join_pending = False
            else:
                out.append(line.rstrip())
    # collapse runs of more than one blank line (TeX treats them as one \par)
    cleaned = []
    for l in out:
        if l.strip() == '' and cleaned and cleaned[-1].strip() == '':
            continue
        cleaned.append(l)
    while cleaned and cleaned[0].strip() == '':
        cleaned.pop(0)
    return '\n'.join(cleaned).rstrip('\n') + '\n'


def redirect_inputs(text):
    def fix(m):
        path = m.group(1)
        if path.endswith('.tex'):
            path = path[:-4]
        if (path.startswith('sections/') or path.startswith('figures/')) and not path.endswith('_clean'):
            return '\\input{%s_clean}' % path
        return m.group(0)
    return re.sub(r'\\input\{([^}]*)\}', fix, text)


def main():
    written = []
    for sub in ('sections', 'figures'):
        for src in sorted((ROOT / sub).glob('*.tex')):
            if src.stem.endswith('_clean'):
                continue
            dst = src.with_name(src.stem + '_clean.tex')
            dst.write_text(redirect_inputs(strip_comments(src.read_text())))
            written.append(dst)
    main_src = ROOT / 'main.tex'
    main_dst = ROOT / 'main_clean.tex'
    main_dst.write_text(redirect_inputs(strip_comments(main_src.read_text())))
    written.append(main_dst)
    for w in written:
        t = w.read_text()
        left = [i for i, l in enumerate(t.split('\n'), 1) if comment_start(l) >= 0]
        if left:
            raise SystemExit(f'comment left in {w}: lines {left}')
        print(w.relative_to(ROOT))


if __name__ == '__main__':
    main()
