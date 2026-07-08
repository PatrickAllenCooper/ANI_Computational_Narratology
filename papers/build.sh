#!/usr/bin/env bash
# Build a paper project under papers/.
#
# Usage:
#   ./build.sh acl                          # papers/acl/ACL_paper.tex
#   ./build.sh sycophancy                   # papers/sycophancy/sycophancy_paper.tex
#   ./build.sh sycophancy sycophancy_slides.tex
#   ./build.sh acl ACL_paper.tex clean
#
# Archived drafts (e.g. the superseded followup paper) live under
# papers/archive/ and keep their own build.sh.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${1:?usage: ./build.sh acl|sycophancy [doc.tex] [clean]}"
shift

case "$PROJECT" in
  acl|sycophancy) ;;
  *)
    echo "error: unknown project '$PROJECT' (expected acl or sycophancy)" >&2
    exit 1
    ;;
esac

exec "$ROOT/$PROJECT/build.sh" "$@"
