#!/usr/bin/env bash
# shellcheck shell=bash
#
# slurm/env_setup.sh -- sourceable CURC environment for the ANI_Examination
# local-model harness.
#
#   source slurm/env_setup.sh
#
# Everything is overridable from the caller's environment; nothing here
# hardcodes a username, an account or an absolute home path.  Site paths
# (/scratch/alpine, /projects) are CURC facts and are themselves overridable.
#
# Variables this file honours as INPUT (all optional):
#   ANI_REPO             repo root            (default: parent of this script)
#   ANI_USER             login name           (default: $USER, else `id -un`)
#   ANI_SCRATCH          scratch root         (default: /scratch/alpine/$ANI_USER)
#   ANI_PROJECTS         projects root        (default: /projects/$ANI_USER)
#   ANI_VENV             python venv path     (default: $ANI_PROJECTS/ani-venv)
#   ANI_MODEL_ROOT       staged HF weights    (default: $ANI_SCRATCH/models)
#   ANI_HF_HOME          HF cache root        (default: $ANI_SCRATCH/hf)
#   ANI_OUT_DIR          per-cell JSON cache  (default: $ANI_REPO/divergence_study_outputs)
#   ANI_ENDPOINT_DIR     vLLM endpoint files  (default: $ANI_SCRATCH/ani_endpoints)
#   ANI_CKPT_DIR         array checkpoints    (default: $ANI_SCRATCH/ani_checkpoints)
#   NOT_LOCAL_BACKEND    vllm | hf | auto     (default: auto)
#   ANI_ALLOW_ONLINE     1 to leave HF_HUB_OFFLINE unset (staging only)
#   ANI_QUIET            1 to suppress the banner
#
# Variables this file EXPORTS for downstream code:
#   ANI_* (as above, resolved), HF_HOME, HF_HUB_OFFLINE, HF_DATASETS_OFFLINE,
#   TRANSFORMERS_CACHE, HUGGINGFACE_HUB_CACHE, TOKENIZERS_PARALLELISM,
#   OMP_NUM_THREADS, PYTHONPATH, NOT_LOCAL_BACKEND, ANI_LOCAL_BACKEND
#
# Backend switch
# --------------
# vLLM is NOT in the CURC Lmod module stack, so it must come from a
# self-installed venv (uv/pip) or an Apptainer image.  `NOT_LOCAL_BACKEND=auto`
# probes for an importable `vllm` and silently falls back to the HuggingFace
# `transformers` path when it is absent.  The HF path is slower (roughly 4-5x
# on decode) but needs only `module load hf-transformers` plus torch, so the
# harness still runs on day one.
#
# Functions defined for callers: ani_module_try, ani_resolve_backend,
# ani_python, ani_prepare_dirs, ani_env_summary.

# ---------------------------------------------------------------------------
# Option hygiene: honour `set -euo pipefail` while running, but do not leave a
# sourcing interactive shell in -e (a later typo would close their session).
# ---------------------------------------------------------------------------
_ani_sourced=0
(return 0 2>/dev/null) && _ani_sourced=1
_ani_saved_opts="$(set +o)"
set -euo pipefail

_ani_restore_opts() {
    if [ "${_ani_sourced}" = "1" ]; then
        eval "${_ani_saved_opts}"
    fi
}

_ani_say() {
    [ "${ANI_QUIET:-0}" = "1" ] && return 0
    printf 'env_setup: %s\n' "$*" >&2
}

_ani_warn() {
    printf 'env_setup: WARNING: %s\n' "$*" >&2
}

# ---------------------------------------------------------------------------
# Identity and roots -- discovered, never hardcoded
# ---------------------------------------------------------------------------
ANI_USER="${ANI_USER:-${USER:-$(id -un)}}"

_ani_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ANI_REPO="${ANI_REPO:-$(cd -- "${_ani_here}/.." && pwd -P)}"

# On CURC these are the real filesystems. Off-cluster (a laptop, CI, a dry-run
# rehearsal of the runbook) /scratch and /projects do not exist and are not
# creatable, so fall back to a local sandbox rather than dying on mkdir. This is
# what makes `ANI_DRY_RUN=1 bash slurm/array_generate.sbatch` runnable anywhere,
# which is how the coverage self-test is meant to be exercised before a sweep.
if [ -d /scratch/alpine ]; then
    ANI_SCRATCH="${ANI_SCRATCH:-/scratch/alpine/${ANI_USER}}"
else
    ANI_SCRATCH="${ANI_SCRATCH:-${TMPDIR:-/tmp}/ani_scratch_${ANI_USER}}"
fi
if [ -d /projects ]; then
    ANI_PROJECTS="${ANI_PROJECTS:-/projects/${ANI_USER}}"
else
    ANI_PROJECTS="${ANI_PROJECTS:-${ANI_REPO}}"
fi

# The venv lives on /projects, NOT /scratch: /scratch/alpine is purged 90 days
# after file CREATION, which would silently delete a working venv mid-study.
ANI_VENV="${ANI_VENV:-${ANI_PROJECTS}/ani-venv}"

# Weights are far too large for the 250 GB /projects quota (Llama-3.3-70B in
# bf16 is ~140 GB), so they live on /scratch under an explicit purge regime.
# See slurm/stage_weights.sh.
ANI_MODEL_ROOT="${ANI_MODEL_ROOT:-${ANI_SCRATCH}/models}"
ANI_HF_HOME="${ANI_HF_HOME:-${ANI_SCRATCH}/hf}"

ANI_OUT_DIR="${ANI_OUT_DIR:-${ANI_REPO}/divergence_study_outputs}"
ANI_ENDPOINT_DIR="${ANI_ENDPOINT_DIR:-${ANI_SCRATCH}/ani_endpoints}"
ANI_CKPT_DIR="${ANI_CKPT_DIR:-${ANI_SCRATCH}/ani_checkpoints}"
ANI_STAGE_MANIFEST="${ANI_STAGE_MANIFEST:-${ANI_MODEL_ROOT}/.ani_stage_manifest.jsonl}"

export ANI_USER ANI_REPO ANI_SCRATCH ANI_PROJECTS ANI_VENV
export ANI_MODEL_ROOT ANI_HF_HOME ANI_OUT_DIR ANI_ENDPOINT_DIR ANI_CKPT_DIR
export ANI_STAGE_MANIFEST

# ---------------------------------------------------------------------------
# Lmod bootstrap
# ---------------------------------------------------------------------------

# Make the `module` shell function available in a non-interactive shell.
ani_bootstrap_lmod() {
    if declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1; then
        return 0
    fi
    local init
    for init in \
        "${LMOD_PKG:-/nonexistent}/init/bash" \
        /etc/profile.d/lmod.sh \
        /etc/profile.d/z00_lmod.sh \
        /etc/profile.d/00-modulepath.sh \
        /curc/sw/lmod/lmod/init/bash \
        /usr/share/lmod/lmod/init/bash
    do
        if [ -r "${init}" ]; then
            # shellcheck disable=SC1090
            if . "${init}" 2>/dev/null; then
                declare -F module >/dev/null 2>&1 && return 0
                command -v module >/dev/null 2>&1 && return 0
            fi
        fi
    done
    return 1
}

# Load the first module in the argument list that actually loads.
# Never fails the caller; returns 1 if none loaded.
ani_module_try() {
    local m
    if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
        return 1
    fi
    for m in "$@"; do
        if module load "${m}" >/dev/null 2>&1; then
            _ani_say "loaded module ${m}"
            return 0
        fi
    done
    _ani_warn "no module loaded from candidates: $*"
    return 1
}

if ani_bootstrap_lmod; then
    _ani_say "Lmod available"
    # Purge only when explicitly asked -- a blind purge breaks nested jobs
    # that inherited a working environment.
    if [ "${ANI_MODULE_PURGE:-0}" = "1" ]; then
        module purge >/dev/null 2>&1 || true
    fi
    ani_module_try slurm/alpine slurm || true
    # CUDA: H200 needs >12.0, RTX Pro 6000 needs >12.8.  Prefer newest.
    ani_module_try "${ANI_CUDA_MODULE:-cuda/12.8}" cuda/12.8 cuda/12.6 cuda/12.4 cuda/12.3 cuda/12.1 cuda || true
    # Python: anaconda is the documented CURC path; miniforge/python are fallbacks.
    ani_module_try "${ANI_PYTHON_MODULE:-anaconda}" anaconda miniforge python/3.11 python || true
    # Only pull hf-transformers when we are NOT using a self-managed venv,
    # otherwise it can shadow the venv's torch/transformers.
    if [ "${ANI_USE_HF_MODULE:-0}" = "1" ]; then
        ani_module_try hf-transformers || true
    fi
else
    _ani_warn "Lmod not found; assuming modules are already loaded or unnecessary"
fi

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------
if [ -r "${ANI_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${ANI_VENV}/bin/activate"
    _ani_say "activated venv ${ANI_VENV}"
elif [ -r "${ANI_REPO}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "${ANI_REPO}/.venv/bin/activate"
    _ani_say "activated repo venv ${ANI_REPO}/.venv"
else
    _ani_warn "no venv at ${ANI_VENV} (or ${ANI_REPO}/.venv); using system python"
    _ani_warn "create one with: python3 -m venv '${ANI_VENV}' && '${ANI_VENV}/bin/pip' install -U pip"
fi

# ---------------------------------------------------------------------------
# HuggingFace cache + offline discipline
# ---------------------------------------------------------------------------
export HF_HOME="${ANI_HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${ANI_HF_HOME}/hub"
# TRANSFORMERS_CACHE is deprecated in recent transformers but still read by
# older pins; export it for compatibility and keep it consistent with HF_HOME.
export TRANSFORMERS_CACHE="${ANI_HF_HOME}/transformers"
export HF_DATASETS_CACHE="${ANI_HF_HOME}/datasets"

# Compute-node outbound internet is UNVERIFIED on CURC.  Default to offline so
# a job fails fast and loudly at model-load rather than hanging on a blocked
# hub call for the whole walltime.  Staging scripts set ANI_ALLOW_ONLINE=1.
if [ "${ANI_ALLOW_ONLINE:-0}" = "1" ]; then
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE || true
    _ani_say "online mode: HF hub calls permitted"
else
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

# Keep BLAS from oversubscribing the cores Slurm actually gave us.
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
else
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# Repo must be importable as `scripts.*` regardless of cwd.
if [ -n "${PYTHONPATH:-}" ]; then
    case ":${PYTHONPATH}:" in
        *":${ANI_REPO}:"*) : ;;
        *) export PYTHONPATH="${ANI_REPO}:${PYTHONPATH}" ;;
    esac
else
    export PYTHONPATH="${ANI_REPO}"
fi

# ---------------------------------------------------------------------------
# Backend switch: vLLM vs HuggingFace transformers
# ---------------------------------------------------------------------------

ani_python() {
    if command -v python3 >/dev/null 2>&1; then
        python3 "$@"
    else
        python "$@"
    fi
}

# Echo the backend that will actually be used: "vllm" or "hf".
# Honours NOT_LOCAL_BACKEND=vllm|hf|auto.
ani_resolve_backend() {
    local want="${NOT_LOCAL_BACKEND:-auto}"
    case "${want}" in
        vllm)
            if ani_python -c 'import vllm' >/dev/null 2>&1; then
                printf 'vllm'
            else
                _ani_warn "NOT_LOCAL_BACKEND=vllm but 'import vllm' failed; falling back to hf"
                printf 'hf'
            fi
            ;;
        hf)
            printf 'hf'
            ;;
        auto|"")
            if ani_python -c 'import vllm' >/dev/null 2>&1; then
                printf 'vllm'
            else
                printf 'hf'
            fi
            ;;
        *)
            _ani_warn "unrecognised NOT_LOCAL_BACKEND='${want}'; using hf"
            printf 'hf'
            ;;
    esac
}

NOT_LOCAL_BACKEND="${NOT_LOCAL_BACKEND:-auto}"
ANI_RESOLVED_BACKEND="$(ani_resolve_backend)"
# Mirror under a second name: `NOT_LOCAL_BACKEND` is the contracted switch,
# `ANI_LOCAL_BACKEND` carries the *resolved* value (never "auto") so
# scripts/local_backend.py can read a concrete backend without re-probing.
ANI_LOCAL_BACKEND="${ANI_RESOLVED_BACKEND}"
export NOT_LOCAL_BACKEND ANI_LOCAL_BACKEND ANI_RESOLVED_BACKEND

if [ "${ANI_RESOLVED_BACKEND}" = "hf" ]; then
    # Degraded path: HF transformers only.  Callers should shrink sweep sizes.
    export ANI_BACKEND_DEGRADED=1
else
    export ANI_BACKEND_DEGRADED=0
fi

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ani_prepare_dirs() {
    local d
    for d in "${ANI_MODEL_ROOT}" "${ANI_HF_HOME}" "${ANI_OUT_DIR}" \
             "${ANI_ENDPOINT_DIR}" "${ANI_CKPT_DIR}"; do
        mkdir -p "${d}" 2>/dev/null || _ani_warn "could not create ${d}"
    done
}
ani_prepare_dirs

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
ani_env_summary() {
    cat <<SUMMARY
--------------------------------------------------------------------------
 ANI_Examination CURC environment
--------------------------------------------------------------------------
  user            ${ANI_USER}
  repo            ${ANI_REPO}
  scratch         ${ANI_SCRATCH}
  projects        ${ANI_PROJECTS}
  venv            ${ANI_VENV}$( [ -n "${VIRTUAL_ENV:-}" ] && printf ' (active: %s)' "${VIRTUAL_ENV}" )
  python          $(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo '(none)')
  model root      ${ANI_MODEL_ROOT}
  HF_HOME         ${HF_HOME}
  HF_HUB_OFFLINE  ${HF_HUB_OFFLINE:-<unset: online>}
  out dir         ${ANI_OUT_DIR}
  checkpoints     ${ANI_CKPT_DIR}
  endpoints       ${ANI_ENDPOINT_DIR}
  backend         ${NOT_LOCAL_BACKEND} -> ${ANI_RESOLVED_BACKEND}$( [ "${ANI_BACKEND_DEGRADED}" = "1" ] && printf '  [DEGRADED: shrink sweeps 5-10x]' )
  OMP_NUM_THREADS ${OMP_NUM_THREADS}
--------------------------------------------------------------------------
SUMMARY
}

if [ "${ANI_QUIET:-0}" != "1" ]; then
    ani_env_summary >&2
fi

if [ "${ANI_RESOLVED_BACKEND}" = "hf" ] && [ "${ANI_QUIET:-0}" != "1" ]; then
    _ani_warn "vLLM unavailable -> HF transformers backend."
    _ani_warn "Expect ~4-5x slower decode; shrink paraphrase atlases 5-10x (see Guidance_Documents/curc_runbook.md)."
fi

_ani_restore_opts
unset _ani_here _ani_saved_opts _ani_sourced
