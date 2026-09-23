#!/usr/bin/env bash
# Build papers/iclr2027/main.tex (ICLR 2027 template, pdflatex via latexmk).
set -euo pipefail
cd "$(dirname "$0")"
source ../shared/build_lib.sh
setup_tex_paths
detect_tex_tools
"${LATEXMK:-latexmk}" -pdf -interaction=nonstopmode -halt-on-error main.tex
