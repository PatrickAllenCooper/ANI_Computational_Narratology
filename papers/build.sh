#!/usr/bin/env bash
# Build a paper project under papers/.
#
# Usage:
#   ./build.sh acl                          # papers/acl/ACL_paper.tex
#   ./build.sh followup                     # papers/followup/followup_paper.tex
#   ./build.sh sycophancy                   # papers/sycophancy/sycophancy_paper.tex
#   ./build.sh sycophancy sycophancy_slides.tex
#   ./build.sh acl ACL_paper.tex clean

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="${1:?usage: ./build.sh {acl|followup|sycophancy} [doc.tex] [clean]}"
shift

case "$PROJECT" in
  acl|followup|sycophancy) ;;
  *)
    echo "error: unknown project '$PROJECT' (expected acl, followup, or sycophancy)" >&2
    exit 1
    ;;
esac

exec "$ROOT/$PROJECT/build.sh" "$@"
