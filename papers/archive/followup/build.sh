#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../../shared/build_lib.sh
source "$PROJECT_DIR/../../shared/build_lib.sh"
setup_tex_paths
cd "$PROJECT_DIR"
DOC="${1:-followup_paper.tex}"
ACTION="${2:-build}"
build_tex_document "$DOC" "$ACTION"
