#!/usr/bin/env bash
# Shared TeX path setup for paper project folders under papers/.

setup_tex_paths () {
  local shared
  shared="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export TEXINPUTS="${shared}//:${TEXINPUTS:-}"
  export BIBINPUTS="${shared}//:${BIBINPUTS:-}"
  export BSTINPUTS="${shared}//:${BSTINPUTS:-}"
}

detect_tex_tools () {
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
}

build_tex_document () {
  local doc="$1"
  local action="${2:-build}"
  local base="${doc%.tex}"

  if [[ ! -f "$doc" ]]; then
    echo "error: $doc not found in $(pwd)" >&2
    exit 1
  fi

  detect_tex_tools

  clean_aux () {
    rm -f -- \
      "$base".aux "$base".log "$base".out "$base".bbl "$base".blg \
      "$base".toc "$base".xdv "$base".fls "$base".fdb_latexmk \
      "$base".synctex.gz
  }

  if [[ "$action" == "clean" ]]; then
    clean_aux
    echo "cleaned aux files for $base"
    return 0
  fi

  if [[ -n "$LATEXMK" && -n "$XELATEX" ]]; then
    echo "[build.sh] using latexmk + xelatex ($LATEXMK, $XELATEX)"
    "$LATEXMK" -xelatex -interaction=nonstopmode -halt-on-error -file-line-error "$doc"
    echo "[build.sh] OK -> $base.pdf"
    return 0
  fi

  if [[ -n "$XELATEX" ]]; then
    echo "[build.sh] using xelatex (no latexmk; running 2-pass + bibtex + 2-pass)"
    "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$doc"
    if command -v bibtex >/dev/null 2>&1; then
      bibtex "$base" || true
    fi
    "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$doc"
    "$XELATEX" -interaction=nonstopmode -halt-on-error -file-line-error "$doc"
    echo "[build.sh] OK -> $base.pdf"
    return 0
  fi

  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "[build.sh] no local xelatex found; using docker image texlive/texlive"
    local shared_dir
    shared_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker run --rm \
      -v "$(pwd):/work" \
      -v "$shared_dir:/shared:ro" \
      -w /work \
      -e TEXINPUTS="/shared//:${TEXINPUTS:-}" \
      -e BIBINPUTS="/shared//:${BIBINPUTS:-}" \
      -e BSTINPUTS="/shared//:${BSTINPUTS:-}" \
      texlive/texlive:latest \
      bash -lc "latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error $doc"
    echo "[build.sh] OK -> $base.pdf"
    return 0
  fi

  cat <<EOF >&2
error: could not find a TeX installation.
  Install MacTeX/TinyTeX or start Docker with the texlive/texlive image.
EOF
  exit 2
}
