#!/usr/bin/env bash
# Build papers/unified/main.tex with the shared ACL style on the TeX path.
set -euo pipefail
cd "$(dirname "$0")"
source ../shared/build_lib.sh
setup_tex_paths
detect_tex_tools
"${LATEXMK:-latexmk}" -pdf -interaction=nonstopmode -halt-on-error main.tex
