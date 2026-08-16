#!/usr/bin/env bash
#
# slurm/stage_weights.sh -- stage HuggingFace weights onto /scratch/alpine
# from a LOGIN or DTN node, and manage the 90-day purge clock.
#
#   bash slurm/stage_weights.sh Qwen3-8B Llama-3.1-8B-Instruct
#   bash slurm/stage_weights.sh --list
#   bash slurm/stage_weights.sh --due
#   bash slurm/stage_weights.sh --verify Qwen3-8B
#   bash slurm/stage_weights.sh --restage Qwen3-8B
#   bash slurm/stage_weights.sh --smoke
#
# WHY THIS EXISTS
# ---------------
# /scratch/alpine is 10 TB, is NOT backed up, and is purged 90 days after file
# CREATION -- not after last access.  `touch` does not help.  `ls` does not
# help.  Reading the file every day does not help.  The ONLY way to reset the
# clock is to make the bytes into new files.  So this script:
#
#   1. records the staging wall-clock time in a manifest (that time IS the
#      authoritative creation date, because we just created the files),
#   2. prints the resulting purge-due date every time you run it,
#   3. provides --restage, which re-creates the files (by local copy, or by
#      re-download) so the clock genuinely restarts.
#
# --restage defaults to a LOCAL COPY (`cp -R`, deliberately NOT `cp -a`: we do
# not want mtimes preserved, in case the purge sweep keys on mtime rather than
# birth time).  A copy is far faster than re-downloading 140 GB and needs no
# outbound internet.  Use --restage-mode download to re-fetch from the hub.
#
# Compute-node outbound internet is UNVERIFIED on CURC, which is exactly why
# staging happens here on a login/DTN node and jobs then run HF_HUB_OFFLINE=1.
#
# Nothing here hardcodes a username, an account, or a home path.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Staging must reach the hub, so ask env_setup for online mode.
export ANI_ALLOW_ONLINE=1
export ANI_QUIET="${ANI_QUIET:-1}"
# shellcheck source=./env_setup.sh
. "${SCRIPT_DIR}/env_setup.sh"

PURGE_DAYS="${ANI_PURGE_DAYS:-90}"
WARN_DAYS="${ANI_PURGE_WARN_DAYS:-15}"
DEST="${ANI_MODEL_ROOT}"
MANIFEST="${ANI_STAGE_MANIFEST}"

MODE="stage"
RESTAGE_MODE="copy"
DO_CHECKSUMS=0
DRY_RUN=0
REVISION=""
USE_DEFAULT_EXCLUDES=1
EXTRA_ARGS=()
MODELS=()

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
good() { printf '   [ok] %s\n' "$*"; }
bad()  { printf '   [!] %s\n' "$*" >&2; }
die()  { printf 'stage_weights: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    sed -n '3,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --dest DIR             stage root (default: $ANI_MODEL_ROOT)
  --list                 show every staged model and its purge-due date
  --due                  same as --list but exit non-zero if anything is due soon
  --verify [MODEL...]    re-check staged sizes/file counts against the manifest
  --verify-checksums     with --verify, also run full sha256 verification (slow)
  --checksums            compute SHA256SUMS while staging (slow; off by default)
  --restage [MODEL...]   re-create files to reset the 90-day purge clock
  --restage-mode MODE    copy (default, local, fast) | download (re-fetch)
  --revision REV         pin an HF revision/commit
  --warn-days N          days before purge to start warning (default 15)
  --no-default-excludes  keep *.pth / original/* / *.h5 / *.msgpack
  --dry-run              print what would happen, change nothing
  --smoke                self-check with a tiny model, then exit
  -h, --help             this text

MODEL may be an alias (Qwen3-8B) or a full HF repo id (Qwen/Qwen3-8B).
Aliases are read from scripts/local_backend.py LOCAL_MODELS when that module
exists, and otherwise from the built-in fallback table.
USAGE
}

epoch_now() { date -u '+%s'; }

# Portable epoch -> ISO-8601 UTC (GNU date and BSD/macOS date).
epoch_to_iso() {
    local e="$1"
    date -u -d "@${e}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
        || date -u -r "${e}" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null \
        || printf 'epoch:%s' "${e}"
}

epoch_to_date() {
    local e="$1"
    date -u -d "@${e}" '+%Y-%m-%d' 2>/dev/null \
        || date -u -r "${e}" '+%Y-%m-%d' 2>/dev/null \
        || printf 'epoch:%s' "${e}"
}

sha256_cmd() {
    if have sha256sum; then printf 'sha256sum'
    elif have shasum; then printf 'shasum -a 256'
    else return 1
    fi
}

# Payload bytes below a directory.  Counts exactly the files dir_files counts
# (bookkeeping files excluded) so staged size and verified size agree.
# `ls -ln` field 5 is the size on both GNU and BSD userlands; `du -sb` is
# GNU-only and `stat` flags differ between platforms, so neither is used.
dir_bytes() {
    find "$1" -type f ! -name '.ani_*' ! -name 'SHA256SUMS*' -exec ls -ln {} + 2>/dev/null \
        | awk '$5 ~ /^[0-9]+$/ {s += $5} END {printf "%.0f", s + 0}'
}

dir_files() {
    find "$1" -type f ! -name '.ani_*' ! -name 'SHA256SUMS*' 2>/dev/null | wc -l | tr -d ' '
}

human_bytes() {
    awk -v b="$1" 'BEGIN{
        split("B KiB MiB GiB TiB", u, " ")
        i=1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf "%.1f %s", b, u[i]
    }'
}

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

# ---------------------------------------------------------------------------
# alias resolution
# ---------------------------------------------------------------------------

# Fallback alias table: the declared Phase 19 / AGT local targets.
# Kept in sync with scripts/local_backend.py LOCAL_MODELS when that exists.
fallback_repo_for_alias() {
    case "$1" in
        Llama-3.1-8B-Instruct|llama-3.1-8b-instruct) printf 'meta-llama/Llama-3.1-8B-Instruct' ;;
        Llama-3.3-70B-Instruct|llama-3.3-70b-instruct) printf 'meta-llama/Llama-3.3-70B-Instruct' ;;
        Qwen3-8B|qwen3-8b)   printf 'Qwen/Qwen3-8B' ;;
        Qwen3-32B|qwen3-32b) printf 'Qwen/Qwen3-32B' ;;
        *) return 1 ;;
    esac
}

# Ask scripts/local_backend.py for the alias table if that module exists yet.
contract_repo_for_alias() {
    local a="$1"
    ani_python - "$a" <<'PY' 2>/dev/null || return 1
import sys
try:
    from scripts.local_backend import LOCAL_MODELS
except Exception:
    raise SystemExit(1)
repo = LOCAL_MODELS.get(sys.argv[1], "")
if not repo:
    raise SystemExit(1)
print(repo)
PY
}

resolve_repo() {
    local spec="$1" repo=""
    # A full repo id always contains exactly one slash and no spaces.
    case "${spec}" in
        */*) printf '%s' "${spec}"; return 0 ;;
    esac
    if repo="$(contract_repo_for_alias "${spec}")" && [ -n "${repo}" ]; then
        printf '%s' "${repo}"; return 0
    fi
    if repo="$(fallback_repo_for_alias "${spec}")" && [ -n "${repo}" ]; then
        printf '%s' "${repo}"; return 0
    fi
    return 1
}

# Local directory name for a repo id: org__name (filesystem-safe, reversible).
local_dir_for_repo() {
    printf '%s/%s' "${DEST}" "$(printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"
}

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dest) DEST="${2:?--dest needs a path}"; shift 2 ;;
        --list) MODE="list"; shift ;;
        --due) MODE="due"; shift ;;
        --verify) MODE="verify"; shift ;;
        --verify-checksums) MODE="verify"; DO_CHECKSUMS=1; shift ;;
        --checksums) DO_CHECKSUMS=1; shift ;;
        --restage) MODE="restage"; shift ;;
        --restage-mode) RESTAGE_MODE="${2:?}"; shift 2 ;;
        --revision) REVISION="${2:?}"; shift 2 ;;
        --warn-days) WARN_DAYS="${2:?}"; shift 2 ;;
        --purge-days) PURGE_DAYS="${2:?}"; shift 2 ;;
        --no-default-excludes) USE_DEFAULT_EXCLUDES=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --smoke) MODE="smoke"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; while [ "$#" -gt 0 ]; do EXTRA_ARGS+=("$1"); shift; done ;;
        -*) die "unknown option: $1 (see --help)" ;;
        *) MODELS+=("$1"); shift ;;
    esac
done

case "${RESTAGE_MODE}" in
    copy|download) : ;;
    *) die "--restage-mode must be 'copy' or 'download'" ;;
esac

MANIFEST="${DEST}/.ani_stage_manifest.jsonl"
mkdir -p "${DEST}"

# ---------------------------------------------------------------------------
# manifest access
# ---------------------------------------------------------------------------

manifest_append() {
    # $1 repo  $2 alias  $3 local_dir  $4 epoch  $5 bytes  $6 files  $7 tool  $8 revision
    local due=$(( $4 + PURGE_DAYS * 86400 ))
    printf '{"repo_id":"%s","alias":"%s","local_dir":"%s","staged_at":"%s","staged_epoch":%s,"bytes":%s,"files":%s,"purge_days":%s,"purge_due":"%s","purge_due_epoch":%s,"tool":"%s","revision":"%s","checksums":%s}\n' \
        "$(json_escape "$1")" "$(json_escape "$2")" "$(json_escape "$3")" \
        "$(epoch_to_iso "$4")" "$4" "$5" "$6" "${PURGE_DAYS}" \
        "$(epoch_to_iso "${due}")" "${due}" \
        "$(json_escape "$7")" "$(json_escape "$8")" \
        "$( [ "${DO_CHECKSUMS}" -eq 1 ] && printf 'true' || printf 'false' )" \
        >> "${MANIFEST}"
}

# Emit the newest manifest record per local_dir as TSV:
#   local_dir \t repo \t staged_epoch \t due_epoch \t bytes \t files
manifest_latest_tsv() {
    [ -r "${MANIFEST}" ] || return 0
    ani_python - "${MANIFEST}" <<'PY'
import json, sys
latest = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("local_dir", "")
        prev = latest.get(key)
        if prev is None or rec.get("staged_epoch", 0) >= prev.get("staged_epoch", 0):
            latest[key] = rec
for key in sorted(latest):
    r = latest[key]
    print("\t".join(str(r.get(f, "")) for f in
          ("local_dir", "repo_id", "staged_epoch", "purge_due_epoch", "bytes", "files")))
PY
}

# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------

download_tool() {
    if have hf; then printf 'hf'
    elif have huggingface-cli; then printf 'huggingface-cli'
    elif ani_python -c 'import huggingface_hub' >/dev/null 2>&1; then printf 'python'
    else return 1
    fi
}

default_excludes() {
    [ "${USE_DEFAULT_EXCLUDES}" -eq 1 ] || return 0
    # Llama repos ship consolidated .pth alongside safetensors; pulling both
    # doubles the transfer and the disk footprint for no benefit.
    printf '%s\n' 'original/*' '*.pth' '*.h5' '*.msgpack' '*.onnx'
}

do_download() {
    local repo="$1" dir="$2" tool="$3"
    local -a args
    mkdir -p "${dir}"

    case "${tool}" in
        hf|huggingface-cli)
            args=(download "${repo}" --local-dir "${dir}")
            [ -n "${REVISION}" ] && args+=(--revision "${REVISION}")
            while IFS= read -r pattern; do
                if [ -n "${pattern}" ]; then args+=(--exclude "${pattern}"); fi
            done <<EOF
$(default_excludes)
EOF
            [ "${#EXTRA_ARGS[@]}" -gt 0 ] && args+=("${EXTRA_ARGS[@]}")
            note "running: ${tool} ${args[*]}"
            if [ "${DRY_RUN}" -eq 1 ]; then return 0; fi
            "${tool}" "${args[@]}"
            ;;
        python)
            note "running: huggingface_hub.snapshot_download(${repo})"
            if [ "${DRY_RUN}" -eq 1 ]; then return 0; fi
            local excl
            excl="$(default_excludes | paste -sd '|' - 2>/dev/null || true)"
            ANI_REPO_ID="${repo}" ANI_DIR="${dir}" ANI_REV="${REVISION}" ANI_EXCL="${excl}" \
                ani_python - <<'PY'
import os
from huggingface_hub import snapshot_download
excl = [p for p in os.environ.get("ANI_EXCL", "").split("|") if p]
kw = dict(repo_id=os.environ["ANI_REPO_ID"],
          local_dir=os.environ["ANI_DIR"],
          ignore_patterns=excl or None)
rev = os.environ.get("ANI_REV") or None
if rev:
    kw["revision"] = rev
path = snapshot_download(**kw)
print(f"snapshot at {path}")
PY
            ;;
        *) die "unknown download tool ${tool}" ;;
    esac
}

write_checksums() {
    local dir="$1" cmd
    cmd="$(sha256_cmd)" || { bad "no sha256 tool; skipping checksums"; return 0; }
    note "computing SHA256SUMS (this is slow for large models)..."
    if [ "${DRY_RUN}" -eq 1 ]; then return 0; fi
    # The tmp file must be excluded from its own listing: the shell creates it
    # (via the redirect) before `find` runs, so a bare `! -name SHA256SUMS`
    # would record a checksum for SHA256SUMS.tmp, which then no longer exists
    # after the rename and makes every later --verify-checksums fail.
    ( cd "${dir}" \
        && find . -type f ! -name 'SHA256SUMS*' ! -name '.ani_*' -print0 \
        | sort -z \
        | xargs -0 ${cmd} > SHA256SUMS.tmp \
        && mv SHA256SUMS.tmp SHA256SUMS )
    good "wrote ${dir}/SHA256SUMS ($(wc -l < "${dir}/SHA256SUMS" | tr -d ' ') entries)"
}

write_dir_meta() {
    local dir="$1" repo="$2" epoch="$3" bytes="$4" files="$5"
    local due=$(( epoch + PURGE_DAYS * 86400 ))
    [ "${DRY_RUN}" -eq 1 ] && return 0
    cat > "${dir}/.ani_stage_meta.json" <<META
{
  "repo_id": "$(json_escape "${repo}")",
  "local_dir": "$(json_escape "${dir}")",
  "staged_at": "$(epoch_to_iso "${epoch}")",
  "staged_epoch": ${epoch},
  "bytes": ${bytes},
  "files": ${files},
  "purge_days": ${PURGE_DAYS},
  "purge_due": "$(epoch_to_iso "${due}")",
  "purge_due_epoch": ${due},
  "revision": "$(json_escape "${REVISION}")"
}
META
}

stage_one() {
    local spec="$1" repo dir tool epoch bytes files
    repo="$(resolve_repo "${spec}")" \
        || die "cannot resolve '${spec}' to an HF repo id (pass org/name explicitly)"
    dir="$(local_dir_for_repo "${repo}")"

    say ""
    say "== ${spec}  ->  ${repo}"
    note "target: ${dir}"

    if [ -d "${dir}" ] && [ "$(dir_files "${dir}")" -gt 0 ] && [ "${MODE}" != "restage" ]; then
        good "already staged ($(human_bytes "$(dir_bytes "${dir}")"), $(dir_files "${dir}") files)"
        note "use --restage to reset the 90-day purge clock"
        report_due_for_dir "${dir}"
        return 0
    fi

    tool="$(download_tool)" || die "no hf / huggingface-cli / huggingface_hub available"
    do_download "${repo}" "${dir}" "${tool}"

    epoch="$(epoch_now)"
    if [ "${DRY_RUN}" -eq 1 ]; then
        note "(dry run: no manifest record written)"
        return 0
    fi
    bytes="$(dir_bytes "${dir}")"
    files="$(dir_files "${dir}")"
    [ "${files}" -gt 0 ] || die "staging produced 0 files in ${dir}"

    [ "${DO_CHECKSUMS}" -eq 1 ] && write_checksums "${dir}"
    write_dir_meta "${dir}" "${repo}" "${epoch}" "${bytes}" "${files}"
    manifest_append "${repo}" "${spec}" "${dir}" "${epoch}" "${bytes}" "${files}" "${tool}" "${REVISION}"

    good "staged ${files} files, $(human_bytes "${bytes}")"
    report_due_for_dir "${dir}"
}

# ---------------------------------------------------------------------------
# restaging: genuinely re-create the bytes so the purge clock restarts
# ---------------------------------------------------------------------------
restage_one() {
    local spec="$1" repo dir tmp epoch bytes files avail_kb need_kb
    repo="$(resolve_repo "${spec}")" || die "cannot resolve '${spec}'"
    dir="$(local_dir_for_repo "${repo}")"

    say ""
    say "== RESTAGE ${repo}  (mode: ${RESTAGE_MODE})"

    if [ ! -d "${dir}" ]; then
        bad "${dir} does not exist -- nothing to restage; staging fresh instead"
        MODE="stage"
        stage_one "${spec}"
        MODE="restage"
        return 0
    fi

    tmp="${dir}.restage.$$"
    if [ "${RESTAGE_MODE}" = "copy" ]; then
        need_kb="$(du -sk "${dir}" 2>/dev/null | awk '{print $1}')"
        avail_kb="$(df -Pk "${DEST}" 2>/dev/null | awk 'NR==2{print $4}')"
        if [ -n "${need_kb}" ] && [ -n "${avail_kb}" ] && [ "${avail_kb}" -lt "${need_kb}" ]; then
            die "restage --restage-mode copy needs ~$(human_bytes $((need_kb * 1024))) transient free space on ${DEST}; only $(human_bytes $((avail_kb * 1024))) available. Use --restage-mode download."
        fi
        note "copying to ${tmp} (new inodes -> new creation time)"
        note "NOTE: cp -R, deliberately WITHOUT -a/-p: preserving mtimes could"
        note "      leave the purge sweep thinking these bytes are still old."
        if [ "${DRY_RUN}" -eq 1 ]; then note "(dry run)"; return 0; fi
        mkdir -p "${tmp}"
        cp -R "${dir}/." "${tmp}/"
    else
        note "re-downloading into ${tmp}"
        if [ "${DRY_RUN}" -eq 1 ]; then note "(dry run)"; return 0; fi
        local tool
        tool="$(download_tool)" || die "no download tool for --restage-mode download"
        do_download "${repo}" "${tmp}" "${tool}"
    fi

    files="$(dir_files "${tmp}")"
    [ "${files}" -gt 0 ] || { rm -rf "${tmp}"; die "restage produced 0 files; original left untouched"; }

    # Swap only after the replacement is known good.
    rm -rf "${dir}.old.$$"
    mv "${dir}" "${dir}.old.$$"
    mv "${tmp}" "${dir}"
    rm -rf "${dir}.old.$$"

    epoch="$(epoch_now)"
    bytes="$(dir_bytes "${dir}")"
    files="$(dir_files "${dir}")"
    [ "${DO_CHECKSUMS}" -eq 1 ] && write_checksums "${dir}"
    write_dir_meta "${dir}" "${repo}" "${epoch}" "${bytes}" "${files}"
    manifest_append "${repo}" "${spec}" "${dir}" "${epoch}" "${bytes}" "${files}" "restage-${RESTAGE_MODE}" "${REVISION}"

    good "restaged ${files} files, $(human_bytes "${bytes}") -- purge clock reset"
    report_due_for_dir "${dir}"
}

# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
verify_one_dir() {
    local dir="$1" rec_bytes="$2" rec_files="$3" now_bytes now_files rc=0
    if [ ! -d "${dir}" ]; then
        bad "MISSING: ${dir} (purged? never staged?)"
        return 1
    fi
    now_bytes="$(dir_bytes "${dir}")"
    now_files="$(dir_files "${dir}")"
    if [ -n "${rec_files}" ] && [ "${rec_files}" != "0" ] && [ "${now_files}" != "${rec_files}" ]; then
        bad "${dir}: file count ${now_files} != manifest ${rec_files}"
        rc=1
    fi
    # du can vary a little with block accounting; allow 2%.
    if [ -n "${rec_bytes}" ] && [ "${rec_bytes}" != "0" ]; then
        if ! awk -v a="${now_bytes}" -v b="${rec_bytes}" \
             'BEGIN{d=(a>b?a-b:b-a); exit !(b>0 && d/b<=0.02)}'; then
            bad "${dir}: size $(human_bytes "${now_bytes}") differs from manifest $(human_bytes "${rec_bytes}") by >2%"
            rc=1
        fi
    fi
    if [ "${DO_CHECKSUMS}" -eq 1 ]; then
        if [ -r "${dir}/SHA256SUMS" ]; then
            local cmd out crc=0
            if cmd="$(sha256_cmd)"; then
                note "verifying sha256 for ${dir} (slow)..."
                out="$( cd "${dir}" && ${cmd} -c SHA256SUMS 2>&1 )" || crc=$?
                if [ "${crc}" -eq 0 ]; then
                    good "checksums OK: ${dir}"
                else
                    bad "CHECKSUM MISMATCH: ${dir}"
                    printf '%s\n' "${out}" | grep -Ev ': OK$' | sed 's/^/       /' >&2
                    rc=1
                fi
            fi
        else
            bad "${dir}: no SHA256SUMS (stage with --checksums to create one)"
        fi
    fi
    [ "${rc}" -eq 0 ] && good "verified ${dir} ($(human_bytes "${now_bytes}"), ${now_files} files)"
    return "${rc}"
}

# ---------------------------------------------------------------------------
# purge reporting
# ---------------------------------------------------------------------------
report_due_for_dir() {
    local dir="$1" meta due now left
    meta="${dir}/.ani_stage_meta.json"
    [ -r "${meta}" ] || return 0
    due="$(ani_python -c 'import json,sys;print(json.load(open(sys.argv[1]))["purge_due_epoch"])' "${meta}" 2>/dev/null || true)"
    [ -n "${due}" ] || return 0
    now="$(epoch_now)"
    left=$(( (due - now) / 86400 ))
    if [ "${left}" -lt 0 ]; then
        bad "PURGE OVERDUE by $(( -left )) days -- these bytes may already be gone. Re-stage now."
    elif [ "${left}" -le "${WARN_DAYS}" ]; then
        bad "purge due $(epoch_to_date "${due}") -- only ${left} days left. Run: bash slurm/stage_weights.sh --restage"
    else
        note "next re-stage due by $(epoch_to_date "${due}") (${left} days from now)"
    fi
}

cmd_list() {
    local rows any=0 now soonest=""
    now="$(epoch_now)"
    rows="$(manifest_latest_tsv)"
    if [ -z "${rows}" ]; then
        note "no staged models recorded in ${MANIFEST}"
        return 0
    fi
    printf '   %-46s %-12s %-12s %8s %7s %s\n' \
        "LOCAL DIR" "STAGED" "PURGE DUE" "SIZE" "FILES" "STATE"
    printf '   %s\n' "---------------------------------------------------------------------------------------------------"
    while IFS="$(printf '\t')" read -r dir repo staged due bytes files; do
        [ -n "${dir}" ] || continue
        any=1
        local left state
        left=$(( (due - now) / 86400 ))
        if [ ! -d "${dir}" ]; then
            state="MISSING"
        elif [ "${left}" -lt 0 ]; then
            state="OVERDUE(${left}d)"
        elif [ "${left}" -le "${WARN_DAYS}" ]; then
            state="RESTAGE SOON(${left}d)"
        else
            state="ok(${left}d)"
        fi
        if [ -z "${soonest}" ] || [ "${left}" -lt "${soonest}" ]; then soonest="${left}"; fi
        printf '   %-46s %-12s %-12s %8s %7s %s\n' \
            "$(basename "${dir}")" "$(epoch_to_date "${staged}")" "$(epoch_to_date "${due}")" \
            "$(human_bytes "${bytes}")" "${files}" "${state}"
    done <<EOF
${rows}
EOF
    [ "${any}" -eq 1 ] || note "no records"
    say ""
    note "manifest: ${MANIFEST}"
    note "purge policy: ${PURGE_DAYS} days from file CREATION on ${DEST}"
    note "re-copying is the ONLY reset; touching/reading does nothing."
    if [ -n "${soonest}" ] && [ "${soonest}" -le "${WARN_DAYS}" ]; then
        return 3
    fi
    return 0
}

cmd_verify() {
    local rows rc=0
    rows="$(manifest_latest_tsv)"
    if [ -z "${rows}" ]; then
        note "nothing to verify (empty manifest)"
        return 0
    fi
    while IFS="$(printf '\t')" read -r dir repo staged due bytes files; do
        [ -n "${dir}" ] || continue
        if [ "${#MODELS[@]}" -gt 0 ]; then
            local want match=0 wrepo wdir
            for want in "${MODELS[@]}"; do
                wrepo="$(resolve_repo "${want}" || printf '%s' "${want}")"
                wdir="$(local_dir_for_repo "${wrepo}")"
                [ "${wdir}" = "${dir}" ] && match=1
            done
            [ "${match}" -eq 1 ] || continue
        fi
        verify_one_dir "${dir}" "${bytes}" "${files}" || rc=1
        report_due_for_dir "${dir}"
    done <<EOF
${rows}
EOF
    return "${rc}"
}

cmd_smoke() {
    say "== stage_weights smoke test (no large downloads) =="
    note "dest        ${DEST}"
    note "manifest    ${MANIFEST}"
    note "purge days  ${PURGE_DAYS} (warn at ${WARN_DAYS} days remaining)"
    local t
    t="$(download_tool)" && good "download tool: ${t}" || bad "no download tool available"
    sha256_cmd >/dev/null 2>&1 && good "sha256 tool: $(sha256_cmd)" || bad "no sha256 tool"

    local now due
    now="$(epoch_now)"
    due=$(( now + PURGE_DAYS * 86400 ))
    good "date maths: now=$(epoch_to_iso "${now}") due=$(epoch_to_iso "${due}")"

    local spec
    for spec in Qwen3-8B Llama-3.1-8B-Instruct Qwen3-32B Llama-3.3-70B-Instruct Qwen/Qwen3-4B; do
        if r="$(resolve_repo "${spec}")"; then
            good "resolve ${spec} -> ${r}  dir=$(basename "$(local_dir_for_repo "${r}")")"
        else
            bad "resolve ${spec} FAILED"
        fi
    done

    # Exercise the manifest round-trip in a sandbox, not the real one.
    local sandbox saved_manifest saved_dest
    sandbox="$(mktemp -d 2>/dev/null || printf '%s' "${TMPDIR:-/tmp}/ani_smoke_$$")"
    mkdir -p "${sandbox}/fake_model"
    printf 'weights\n' > "${sandbox}/fake_model/model.safetensors"
    saved_manifest="${MANIFEST}"; saved_dest="${DEST}"
    MANIFEST="${sandbox}/.ani_stage_manifest.jsonl"; DEST="${sandbox}"
    manifest_append "fake/Model" "fake" "${sandbox}/fake_model" "${now}" \
        "$(dir_bytes "${sandbox}/fake_model")" "$(dir_files "${sandbox}/fake_model")" "smoke" ""
    write_dir_meta "${sandbox}/fake_model" "fake/Model" "${now}" 1 1
    say ""
    cmd_list || true
    say ""
    verify_one_dir "${sandbox}/fake_model" "$(dir_bytes "${sandbox}/fake_model")" "1" \
        && good "verify round-trip OK"
    MANIFEST="${saved_manifest}"; DEST="${saved_dest}"
    rm -rf "${sandbox}"
    say ""
    good "smoke OK"
    return 0
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
case "${MODE}" in
    smoke)
        cmd_smoke
        ;;
    list)
        cmd_list || true
        ;;
    due)
        cmd_list
        ;;
    verify)
        cmd_verify
        ;;
    restage)
        [ "${#MODELS[@]}" -gt 0 ] || die "--restage needs at least one MODEL (or 'all')"
        if [ "${#MODELS[@]}" -eq 1 ] && [ "${MODELS[0]}" = "all" ]; then
            MODELS=()
            while IFS="$(printf '\t')" read -r dir repo _rest; do
                if [ -n "${repo}" ]; then MODELS+=("${repo}"); fi
            done <<EOF
$(manifest_latest_tsv)
EOF
            [ "${#MODELS[@]}" -gt 0 ] || die "--restage all: manifest ${MANIFEST} lists nothing"
        fi
        for m in "${MODELS[@]}"; do restage_one "${m}"; done
        say ""
        cmd_list || true
        ;;
    stage)
        if [ "${#MODELS[@]}" -eq 0 ]; then
            usage
            say ""
            die "no MODEL given"
        fi
        for m in "${MODELS[@]}"; do stage_one "${m}"; done
        say ""
        cmd_list || true
        ;;
    *)
        die "internal: unknown mode ${MODE}"
        ;;
esac
