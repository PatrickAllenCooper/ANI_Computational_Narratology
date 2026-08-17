#!/usr/bin/env bash
#SBATCH --job-name=ani-vllm
#SBATCH --partition=ah200
#SBATCH --gres=gpu:h200:2
#SBATCH --qos=gpu-long
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# =============================================================================
# slurm/serve_vllm.sbatch -- persistent vLLM OpenAI-compatible endpoint.
# =============================================================================
#
# WHEN TO USE THIS INSTEAD OF array_generate.sbatch
#   array_generate.sbatch is the right tool for a fixed, enumerable sweep: it
#   loads the model, chews through its shard, and exits.  It is the wrong tool
#   for anything ITERATIVE -- the prompt optimizers in scripts/syco_loss.py,
#   scripts/run_phase14_*.py and the propensity probes decide what to ask next
#   based on what came back, so they cannot be flattened into a manifest up
#   front.  Those want a long-lived endpoint they can call in a loop.
#
#   This job provides exactly that: one server, held for its walltime, that any
#   number of client jobs (or an interactive session) can call concurrently.
#   32B and 70B live here rather than on MIG slices, because they need whole
#   GPUs and tensor parallelism.
#
# SUBMIT
#   mkdir -p slurm_logs
#   # See array_generate.sbatch's SUBMIT comment: set SBATCH_ACCOUNT to what
#   # your own preflight.sh (U1) recommends -- this account has three
#   # associations, not one default.
#   export SBATCH_ACCOUNT=ucb736_asc1
#   sbatch --export=ALL,ANI_SERVE_MODEL=qwen3-32b slurm/serve_vllm.sbatch
#
#   70B: the QOS caps force a choice, confirmed 2026-08-17 via `sacctmgr show
#   qos` on this account -- gpu-normal allows 4 concurrent whole h200 units
#   but MaxWall=24h; gpu-long allows up to 7 days but caps h200 at 2 units.
#   You cannot have both 4 GPUs AND >24h on a single job. Two honest options:
#
#     # (a) 4 whole H200s, <=24h (a long single-day run):
#     sbatch --qos=gpu-normal --gres=gpu:h200:4 --cpus-per-task=32 --mem=400G \
#            --time=24:00:00 \
#            --export=ALL,ANI_SERVE_MODEL=llama-3.3-70b slurm/serve_vllm.sbatch
#
#     # (b) 2 whole H200s (TP=2; 2x141GB is already enough for 70B bf16 +
#     # KV cache), up to 7 days:
#     sbatch --qos=gpu-long --gres=gpu:h200:2 --cpus-per-task=16 --mem=200G \
#            --time=48:00:00 --export=ALL,ANI_TP_SIZE=2,ANI_SERVE_MODEL=llama-3.3-70b \
#            slurm/serve_vllm.sbatch
#
#   8B smoke on the debug queue (1 h cap, billed at 10% -- confirmed via
#   TRESBillingWeights UsageFactor=0.100000; valid ONLY on aa100 and ami100).
#   CONFIRMED 2026-08-17 the hard way (a real submission failed with Slurm
#   error 7): gpu-testing on aa100 permits ONLY the a100_3g.20gb MIG slice as
#   a GRES type -- an 8B model in bf16 (~16GB) fits its 20GB with only ~4GB
#   left for KV cache, tight but workable for a short smoke test with modest
#   --max-model-len. ALSO CONFIRMED 2026-08-17 (a real submission failed with
#   Slurm error 12, "requesting 49152 MiB ... exceeds the maximum allowed of
#   38400 MiB"): this file's header default of --mem=200G (and the naive
#   48G guess below it used to have) is way past what a100_3g.20gb allows --
#   override --mem below 38400M:
#     sbatch --partition=aa100 --qos=gpu-testing --gres=gpu:a100_3g.20gb:1 \
#            --time=01:00:00 --cpus-per-task=4 --mem=32G \
#            --export=ALL,ANI_SERVE_MODEL=qwen3-8b,ANI_SERVE_MAX_MODEL_LEN=4096 \
#            slurm/serve_vllm.sbatch
#
# HOW CLIENTS FIND IT
#   The server writes "host:port" as the first line of
#       $ANI_ENDPOINT_DIR/vllm_<jobid>.txt
#   which is exactly where scripts/local_backend.py looks: its
#   endpoint_file_candidates() scans $ANI_ENDPOINT_DIR newest-mtime-first and
#   reads the first non-comment line.  So a client only has to do:
#
#       source slurm/env_setup.sh          # exports ANI_ENDPOINT_DIR
#       python -m scripts.run_brokenmath --generators qwen3-32b
#
#   The heartbeat below re-touches that file, so the newest file is always the
#   live server.  A rich JSON sidecar is written alongside as a DOTFILE
#   (.vllm_<jobid>.json) precisely so that the endpoint scan -- which ignores
#   dotfiles -- never mistakes it for a host:port file.
#
#   To bypass discovery entirely:  export NOT_VLLM_BASE_URL=http://host:port/v1
#
# ENVIRONMENT KNOBS
#   ANI_SERVE_MODEL            alias or HF repo id (default: qwen3-32b)
#   ANI_TP_SIZE                tensor-parallel size (default: GPUs on the node)
#   ANI_SERVE_PORT             fixed port (default: an OS-assigned free port)
#   ANI_SERVE_MAX_SECONDS      hard self-imposed lifetime (default: from the
#                              Slurm walltime minus ANI_SERVE_GRACE_S)
#   ANI_SERVE_GRACE_S          shutdown margin before walltime (default 120)
#   ANI_SERVE_STARTUP_TIMEOUT  seconds to wait for readiness (default 1800)
#   ANI_SERVE_HEALTH_INTERVAL  seconds between health probes (default 60)
#   ANI_SERVE_MAX_MODEL_LEN    --max-model-len passed to vLLM
#   ANI_SERVE_GPU_UTIL         --gpu-memory-utilization (default 0.90)
#   ANI_SERVE_ENFORCE_EAGER    --enforce-eager on by default (default 1); set
#                              0 to allow CUDA graph capture (more throughput,
#                              slower/riskier startup -- see the code comment)
#   ANI_SERVE_SWAP_SPACE_GB    --swap-space, CPU RAM for KV-cache offload
#                              under VRAM pressure (default 8)
#   ANI_VLLM_CONDA_ENV         override path to a pre-built vllm conda env,
#                              tried before failing (default:
#                              $ANI_PROJECTS/software/anaconda/envs/vllm-env,
#                              matching a sibling project's proven install)
#   ANI_SERVE_EXTRA            extra flags appended to the vllm command line
#
# Nothing here hardcodes a username, an account, or a home path.
# =============================================================================
set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}/slurm" ]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}/slurm"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
export ANI_REPO="${ANI_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"

export ANI_QUIET="${ANI_QUIET:-0}"
# This job IS the vLLM path; do not let env_setup silently resolve to hf.
export NOT_LOCAL_BACKEND="${NOT_LOCAL_BACKEND:-vllm}"
# shellcheck source=./env_setup.sh
. "${SCRIPT_DIR}/env_setup.sh"

cd "${ANI_REPO}"

JOB_ID="${SLURM_JOB_ID:-local$$}"
SERVE_MODEL="${ANI_SERVE_MODEL:-qwen3-32b}"
GRACE_S="${ANI_SERVE_GRACE_S:-120}"
STARTUP_TIMEOUT="${ANI_SERVE_STARTUP_TIMEOUT:-1800}"
HEALTH_INTERVAL="${ANI_SERVE_HEALTH_INTERVAL:-60}"
GPU_UTIL="${ANI_SERVE_GPU_UTIL:-0.90}"
# --enforce-eager and --swap-space are on by default: a sibling project's real
# vLLM deployment on this account's CURC allocation
# (/Users/pat/code/blanc/hpc/slurm_evaluate_curc_vllm.sh) credits both with
# specific, observed failure modes -- CUDA graph / Inductor capture causing a
# startup OOM on a large model, and tight VRAM without CPU KV-cache offload.
# --enforce-eager costs some steady-state throughput in exchange for a much
# more predictable, faster startup (skips capture entirely); disable it with
# ANI_SERVE_ENFORCE_EAGER=0 if profiling shows the throughput cost matters
# more than startup risk for a given model/GPU combination.
ENFORCE_EAGER="${ANI_SERVE_ENFORCE_EAGER:-1}"
SWAP_SPACE_GB="${ANI_SERVE_SWAP_SPACE_GB:-8}"

log() { printf '[serve %s] %s\n' "$(date -u '+%H:%M:%SZ')" "$*"; }
die() { printf '[serve] FATAL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
PYBIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
[ -n "${PYBIN}" ] || die "no python interpreter found"

if ! "${PYBIN}" -c 'import vllm' >/dev/null 2>&1; then
    # A sibling project on this account already built a working vLLM install
    # via a separate CURC-LLM-hoster project: a conda env at
    # /projects/$USER/software/anaconda/envs/vllm-env (see
    # /Users/pat/code/blanc/hpc/slurm_evaluate_curc_vllm.sh). Try it before
    # falling back to a from-scratch build on hardware nothing has verified
    # vLLM against yet.
    VLLM_ENV_DIR="${ANI_VLLM_CONDA_ENV:-${ANI_PROJECTS}/software/anaconda/envs/vllm-env}"
    if [ -d "${VLLM_ENV_DIR}" ] && [ -x "${VLLM_ENV_DIR}/bin/python" ]; then
        log "system python has no vllm; trying existing conda env: ${VLLM_ENV_DIR}"
        if "${VLLM_ENV_DIR}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
            PYBIN="${VLLM_ENV_DIR}/bin/python"
            export PATH="${VLLM_ENV_DIR}/bin:${PATH}"
            log "vLLM found in ${VLLM_ENV_DIR} -- using it"
        else
            log "vllm-env exists but does not import vllm either; giving up on it"
        fi
    fi
fi

if ! "${PYBIN}" -c 'import vllm' >/dev/null 2>&1; then
    cat >&2 <<NOVLLM
[serve] FATAL: vLLM is not importable in this environment.

  Checked: the active python (${PYBIN}), and ${VLLM_ENV_DIR:-<unset>} (a
  sibling project's known-working vllm-env, if present).

  vLLM is NOT in the CURC Lmod module stack -- it has to be self-installed into
  the venv, or run from an Apptainer image:

      python3 -m venv "\$ANI_VENV"
      "\$ANI_VENV/bin/pip" install -U pip
      "\$ANI_VENV/bin/pip" install vllm        # needs CUDA >12.0 (H200),
                                              #       CUDA >12.8 (RTX Pro 6000)

  Or reuse the proven conda recipe directly (see setup_curc_env.sh in
  /Users/pat/code/blanc/hpc/ for the working install-command sequence):
      module load anaconda
      conda create -n vllm-env python=3.11 -y && conda activate vllm-env
      pip install vllm openai tenacity

  Until then, use the HuggingFace fallback instead of this job:
      NOT_LOCAL_BACKEND=hf sbatch slurm/array_generate.sbatch
  and shrink the sweep accordingly (see Guidance_Documents/curc_runbook.md).
NOVLLM
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve alias -> repo id -> staged local directory
# ---------------------------------------------------------------------------
resolve_repo_id() {
    local spec="$1" out
    out="$("${PYBIN}" - "${spec}" <<'PY' 2>/dev/null || true
import sys
spec = sys.argv[1]
try:
    from scripts.local_backend import resolve_repo_id
    print(resolve_repo_id(spec))
except Exception:
    print(spec)
PY
)"
    printf '%s' "${out:-${spec}}"
}

REPO_ID="$(resolve_repo_id "${SERVE_MODEL}")"
# Same directory rule as slurm/stage_weights.sh: org/name -> org_name.
STAGED_DIR="${ANI_MODEL_ROOT}/$(printf '%s' "${REPO_ID}" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"

if [ -d "${STAGED_DIR}" ] && [ -n "$(ls -A "${STAGED_DIR}" 2>/dev/null)" ]; then
    MODEL_PATH="${STAGED_DIR}"
    log "serving staged weights: ${MODEL_PATH}"
else
    MODEL_PATH="${REPO_ID}"
    log "WARNING: ${STAGED_DIR} is missing or empty; falling back to repo id '${REPO_ID}'."
    log "WARNING: with HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} this will fail unless the"
    log "WARNING: compute node has outbound internet. Stage first:"
    log "WARNING:   bash slurm/stage_weights.sh ${SERVE_MODEL}"
fi

# ---------------------------------------------------------------------------
# Tensor parallel size
# ---------------------------------------------------------------------------
GPUS="${SLURM_GPUS_ON_NODE:-}"
if [ -z "${GPUS}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
fi
[ -n "${GPUS}" ] && [ "${GPUS}" -ge 1 ] 2>/dev/null || GPUS=1
TP_SIZE="${ANI_TP_SIZE:-${GPUS}}"
export ANI_TP_SIZE="${TP_SIZE}"

# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------
pick_port() {
    local p
    p="$("${PYBIN}" - <<'PY' 2>/dev/null || true
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
    if [ -n "${p}" ] && [ "${p}" -gt 1024 ] 2>/dev/null; then
        printf '%s' "${p}"
    else
        # Deterministic fallback derived from the job id.
        printf '%s' "$(( 8000 + (${JOB_ID##*_} % 20000) ))"
    fi
}
PORT="${ANI_SERVE_PORT:-$(pick_port)}"
HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
BASE_URL="http://${HOSTNAME_FQDN}:${PORT}/v1"

# ---------------------------------------------------------------------------
# Endpoint advertisement
# ---------------------------------------------------------------------------
mkdir -p "${ANI_ENDPOINT_DIR}"
ENDPOINT_FILE="${ANI_ENDPOINT_DIR}/vllm_${JOB_ID}.txt"
# Dotfile on purpose: local_backend's endpoint scan skips names starting with
# '.', so this metadata can never be misread as a host:port line.
META_FILE="${ANI_ENDPOINT_DIR}/.vllm_${JOB_ID}.json"

# ---------------------------------------------------------------------------
# Lifetime
# ---------------------------------------------------------------------------
MAX_SECONDS="${ANI_SERVE_MAX_SECONDS:-0}"
if [ "${MAX_SECONDS}" -le 0 ] 2>/dev/null; then
    if [ -n "${SLURM_JOB_END_TIME:-}" ] && [ "${SLURM_JOB_END_TIME}" -gt 0 ] 2>/dev/null; then
        MAX_SECONDS=$(( SLURM_JOB_END_TIME - $(date +%s) - GRACE_S ))
    else
        MAX_SECONDS=0
    fi
fi
[ "${MAX_SECONDS}" -gt 0 ] 2>/dev/null || MAX_SECONDS=0

cat <<BANNER
==============================================================================
 ani-vllm  job ${JOB_ID}
==============================================================================
  node            ${HOSTNAME_FQDN}
  model spec      ${SERVE_MODEL}
  repo id         ${REPO_ID}
  model path      ${MODEL_PATH}
  gpus on node    ${GPUS}
  tensor parallel ${TP_SIZE}
  port            ${PORT}
  base url        ${BASE_URL}
  endpoint file   ${ENDPOINT_FILE}
  max lifetime    $( [ "${MAX_SECONDS}" -gt 0 ] && printf '%ss' "${MAX_SECONDS}" || printf 'walltime only' )
  started         $(date -u '+%Y-%m-%dT%H:%M:%SZ')
==============================================================================

  Clients on this cluster:
      source slurm/env_setup.sh
      # ANI_ENDPOINT_DIR discovery picks this server up automatically

  Or pin it explicitly:
      export NOT_VLLM_BASE_URL=${BASE_URL}

BANNER

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
VLLM_PID=""
SHUTTING_DOWN=0

cleanup() {
    [ "${SHUTTING_DOWN}" -eq 1 ] && return 0
    SHUTTING_DOWN=1
    log "cleaning up"
    rm -f "${ENDPOINT_FILE}" "${META_FILE}" 2>/dev/null || true
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "stopping vLLM (pid ${VLLM_PID})"
        kill -TERM "${VLLM_PID}" 2>/dev/null || true
        # Give it a chance to release the GPUs before Slurm reaps the step.
        for _ in $(seq 1 30); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'log "SIGUSR1 -- shutting down"; cleanup; exit 0' USR1
trap 'log "SIGTERM -- shutting down"; cleanup; exit 0' TERM
trap 'log "SIGINT -- shutting down";  cleanup; exit 0' INT

VLLM_ARGS=(
    --model "${MODEL_PATH}"
    --served-model-name "${REPO_ID}" "${SERVE_MODEL}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_UTIL}"
    --disable-log-requests
    --swap-space "${SWAP_SPACE_GB}"
)
if [ "${ENFORCE_EAGER}" = "1" ]; then
    VLLM_ARGS+=(--enforce-eager)
fi
if [ -n "${ANI_SERVE_MAX_MODEL_LEN:-}" ]; then
    VLLM_ARGS+=(--max-model-len "${ANI_SERVE_MAX_MODEL_LEN}")
fi
if [ -n "${ANI_SERVE_EXTRA:-}" ]; then
    # Word-splitting is intended here: ANI_SERVE_EXTRA is a flag string.
    # shellcheck disable=SC2206
    EXTRA_ARR=(${ANI_SERVE_EXTRA})
    VLLM_ARGS+=("${EXTRA_ARR[@]}")
fi

log "launching: vllm serve ${VLLM_ARGS[*]}"
if command -v vllm >/dev/null 2>&1; then
    vllm serve "${VLLM_ARGS[@]}" &
else
    "${PYBIN}" -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
fi
VLLM_PID=$!
log "vLLM pid ${VLLM_PID}"

# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------
probe() {
    if command -v curl >/dev/null 2>&1; then
        curl -sf -m 10 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
    else
        "${PYBIN}" - "${PORT}" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=10).read()
PY
    fi
}

log "waiting up to ${STARTUP_TIMEOUT}s for readiness (large models load slowly)"
READY=0
WAITED=0
while [ "${WAITED}" -lt "${STARTUP_TIMEOUT}" ]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        wait "${VLLM_PID}" || true
        die "vLLM exited during startup -- see the log above (OOM? bad --max-model-len? weights missing?)"
    fi
    if probe; then
        READY=1
        break
    fi
    sleep 5
    WAITED=$(( WAITED + 5 ))
    if [ $(( WAITED % 60 )) -eq 0 ]; then
        log "still loading (${WAITED}s)"
    fi
done
[ "${READY}" -eq 1 ] || die "not ready after ${STARTUP_TIMEOUT}s; raise ANI_SERVE_STARTUP_TIMEOUT"

log "READY after ${WAITED}s"

# Advertise only once healthy, so no client ever sees a half-open endpoint.
{
    printf '%s:%s\n' "${HOSTNAME_FQDN}" "${PORT}"
    printf '# base_url=%s\n' "${BASE_URL}"
    printf '# job_id=%s model=%s tp=%s\n' "${JOB_ID}" "${REPO_ID}" "${TP_SIZE}"
} > "${ENDPOINT_FILE}"

cat > "${META_FILE}" <<META
{
  "host": "${HOSTNAME_FQDN}",
  "port": ${PORT},
  "base_url": "${BASE_URL}",
  "model_spec": "${SERVE_MODEL}",
  "repo_id": "${REPO_ID}",
  "model_path": "${MODEL_PATH}",
  "tensor_parallel_size": ${TP_SIZE},
  "job_id": "${JOB_ID}",
  "node": "${HOSTNAME_FQDN}",
  "ready_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "startup_seconds": ${WAITED},
  "max_seconds": ${MAX_SECONDS}
}
META

log "endpoint advertised: ${ENDPOINT_FILE}  ->  $(head -1 "${ENDPOINT_FILE}")"
if command -v curl >/dev/null 2>&1; then
    log "served models: $(curl -sf -m 10 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | head -c 400)"
fi

# ---------------------------------------------------------------------------
# Health / heartbeat loop
# ---------------------------------------------------------------------------
START_EPOCH="$(date +%s)"
FAILS=0
MAX_FAILS="${ANI_SERVE_MAX_FAILS:-3}"

while true; do
    sleep "${HEALTH_INTERVAL}"

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "vLLM process is gone -- exiting"
        break
    fi

    if probe; then
        FAILS=0
        # Re-touch so local_backend's newest-mtime-first scan keeps preferring
        # this (live) server over any stale file left by an earlier job.
        touch "${ENDPOINT_FILE}" 2>/dev/null || true
    else
        FAILS=$(( FAILS + 1 ))
        log "health probe FAILED (${FAILS}/${MAX_FAILS})"
        if [ "${FAILS}" -ge "${MAX_FAILS}" ]; then
            log "unhealthy ${MAX_FAILS} times in a row -- withdrawing endpoint and exiting"
            rm -f "${ENDPOINT_FILE}" "${META_FILE}" 2>/dev/null || true
            break
        fi
    fi

    if [ "${MAX_SECONDS}" -gt 0 ]; then
        ELAPSED=$(( $(date +%s) - START_EPOCH ))
        if [ "${ELAPSED}" -ge "${MAX_SECONDS}" ]; then
            log "hard lifetime ${MAX_SECONDS}s reached -- shutting down cleanly"
            break
        fi
        REMAIN=$(( MAX_SECONDS - ELAPSED ))
        if [ "${REMAIN}" -le 300 ] && [ $(( REMAIN % 60 )) -lt "${HEALTH_INTERVAL}" ]; then
            log "shutting down in ~${REMAIN}s"
        fi
    fi
done

cleanup
log "server job finished"
exit 0
