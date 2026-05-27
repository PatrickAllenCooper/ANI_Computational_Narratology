#!/usr/bin/env bash
# Build a LaTeX document in this directory (defaults to followup_paper.tex).
# Auto-detects local xelatex/latexmk in common Mac/Linux locations and falls
# back to a docker-based render via the texlive/texlive image.
#
# Usage:
#   ./build.sh                          # builds followup_paper.tex
#   ./build.sh ACL_paper.tex            # builds a different doc
#   ./build.sh followup_paper.tex clean # removes aux files first
#
# Requires either:
#   - latexmk + xelatex in PATH or in /Library/TeX/texbin, ~/Library/TinyTeX/bin/*,
#     /usr/local/texlive/*/bin/*, /opt/homebrew/bin
#   OR
#   - docker (daemon running) with the texlive/texlive image (pulled on demand)

set -euo pipefail

DOC="${1:-followup_paper.tex}"
ACTION="${2:-build}"
BASE="${DOC%.tex}"

cd "$(dirname "$0")"

if [[ ! -f "$DOC" ]]; then
  echo "error: $DOC not found in $(pwd)" >&2
  exit 1
fi

# ----- auto-detect a local TeX install ----------------------------------------
candidate_paths=(
  "/Library/TeX/texbin"
  "$HOME/Library/TinyTeX/bin/universal-darwin"
  "$HOME/Library/TinyTeX/bin/x86_64-darwin"
  "$HOME/Library/TinyTeX/bin/aarch64-darwin"
  "/usr/local/texlive/2025/bin/universal-darwin"
  "/usr/local/texlive/2024/bin/universal-darwin"
  "/usr/local/texlive/2023/bin/universal-darwin"
  "/usr/local/texlive/2025/bin/x86_64-linux"
  "/opt/homebrew/bin"
  "/usr/local/bin"
)

XELATEX=""
LATEXMK=""

if command -v latexmk >/dev/null 2>&1; then
  LATEXMK="$(command -v latexmk)"
fi
if command -v xelatex >/dev/null 2>&1; then
  XELATEX="$(command -v xelatex)"
fi

if [[ -z "$XELATEX" ]]; then
  for d in "${candidate_paths[@]}"; do
    if [[ -x "$d/xelatex" ]]; then
      XELATEX="$d/xelatex"
      [[ -z "$LATEXMK" && -x "$d/latexmk" ]] && LATEXMK="$d/latexmk"
      export PATH="$d:$PATH"
      break
    fi
  done
fi

clean_aux () {
  rm -f -- \
    "$BASE".aux "$BASE".log "$BASE".out "$BASE".bbl "$BASE".blg \
    "$BASE".toc "$BASE".xdv "$BASE".fls "$BASE".fdb_latexmk \
    "$BASE".synctex.gz
}

if [[ "$ACTION" == "clean" ]]; then
  clean_aux
  echo "cleaned aux files for $BASE"
  shift || true
  ACTION="build"
fi

# ----- run the build ----------------------------------------------------------
if [[ -n "$LATEXMK" && -n "$XELATEX" ]]; then
  echo "[build.sh] using latexmk + xelatex ($LATEXMK, $XELATEX)"
  "$LATEXMK" -xelatex -interaction=nonstopmode -halt-on-error -file-line-error "$DOC"
  echo "[build.sh] OK -> $BASE.pdf"
  exit 0
fi

if [[ -n "$XELATEX" ]]; then
  echo "[build.sh] using xelatex (no latexmk; running 2-pass + bibtex + 2-pass)"
  "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$DOC"
  if command -v bibtex >/dev/null 2>&1; then
    bibtex "$BASE" || true
  fi
  "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$DOC"
  "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$DOC"
  echo "[build.sh] OK -> $BASE.pdf"
  exit 0
fi

# ----- fall back to docker ----------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "[build.sh] no local xelatex found; using docker image texlive/texlive"
    docker run --rm \
      -v "$(pwd):/work" \
      -w /work \
      texlive/texlive:latest \
      bash -lc "latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error $DOC"
    echo "[build.sh] OK -> $BASE.pdf"
    exit 0
  else
    echo "error: docker is installed but the daemon is not running." >&2
    echo "       start Docker Desktop and retry, or install MacTeX/TinyTeX." >&2
    exit 2
  fi
fi

cat <<EOF >&2
error: could not find a TeX installation.
  Looked for xelatex/latexmk in PATH and in:
    /Library/TeX/texbin
    ~/Library/TinyTeX/bin/*
    /usr/local/texlive/*/bin/*
    /opt/homebrew/bin, /usr/local/bin
  And docker was not available as a fallback.

To install:
  - MacTeX (full):    brew install --cask mactex-no-gui
  - BasicTeX (small): brew install --cask basictex
                      then: sudo tlmgr install latexmk collection-xetex
  - TinyTeX (smallest): curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
EOF
exit 2
