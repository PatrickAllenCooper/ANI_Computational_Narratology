#!/usr/bin/env bash
#
# slurm/preflight.sh -- one-shot CURC environment audit.
#
# RUN THIS FIRST, on an Alpine LOGIN node, before staging weights or
# submitting anything:
#
#     bash slurm/preflight.sh                 # login-node audit only
#     bash slurm/preflight.sh --probe-compute # also submit a 5-min gpu-testing
#                                             # job to test compute-node internet
#     bash slurm/preflight.sh --json out.json # machine-readable summary too
#
# The script NEVER fails hard: every probe is soft, so a missing command or a
# denied partition produces a recorded UNRESOLVED rather than an abort.  The
# point is to leave you with a summary of which of the six MUST-CONFIRM
# unknowns are now facts and which are still guesses.
#
# THE SIX MUST-CONFIRM UNKNOWNS
#   U1 ACCOUNT    Slurm account name + allocation balance
#   U2 GRES       Real per-partition GPU counts and GRES type strings
#                 (CURC docs are internally inconsistent: the quick-start says
#                  41 GPUs, the hardware table implies ~130)
#   U3 BILLING    TRESBillingWeights / SU acceleration factors for non-A100
#                 devices (only 108.6 SU per A100-GPU-hour is published)
#   U4 INTERNET   Whether COMPUTE nodes have outbound internet
#                 (documented as unverified; needs --probe-compute)
#   U5 RUNTIME    Apptainer availability + CUDA versions in the module stack
#                 (H200 needs CUDA >12.0, RTX Pro 6000 needs >12.8) + whether
#                 vLLM is importable anywhere
#   U6 STORAGE    Scratch quota, /projects quota, and writability
#
# Nothing here hardcodes a username, an account or a home path.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ANI_USER="${ANI_USER:-${USER:-$(id -un)}}"
ANI_SCRATCH="${ANI_SCRATCH:-/scratch/alpine/${ANI_USER}}"
ANI_PROJECTS="${ANI_PROJECTS:-/projects/${ANI_USER}}"

GPU_PARTITIONS=(ah200 artxpro6000 aa100 al40 ami100 gh200)
PROBE_COMPUTE=0
PROBE_WAIT=600
JSON_OUT=""
PROBE_PARTITION="${ANI_PROBE_PARTITION:-aa100}"
# CONFIRMED 2026-08-17: gpu-testing QOS on aa100 permits ONLY a100_3g.20gb as
# a GRES type (ami100's gpu-testing permits only mi100) -- NOT the full-size
# a100-40gb/a100_80gb cards. A real --probe-compute submission with
# --gres=gpu:a100-40gb:1 --qos=gpu-testing failed with Slurm error 7 ("Valid
# GRES types for this partition and QoS selection are: a100_3g.20gb"). This
# was not obvious from sacctmgr's MaxTRESPU listing alone (which reads like a
# concurrency cap, not an exhaustive allow-list) until the submission itself
# proved it.
PROBE_GRES="${ANI_PROBE_GRES:-a100_3g.20gb}"

# --- unknown tracking -------------------------------------------------------
U1_STATUS="UNRESOLVED"; U1_NOTE="Slurm account name / allocation balance"
U2_STATUS="UNRESOLVED"; U2_NOTE="Real per-partition GPU + GRES inventory"
U3_STATUS="UNRESOLVED"; U3_NOTE="SU billing weights for non-A100 devices"
U4_STATUS="UNRESOLVED"; U4_NOTE="Compute-node outbound internet"
U5_STATUS="UNRESOLVED"; U5_NOTE="Apptainer / CUDA / vLLM runtime path"
U6_STATUS="UNRESOLVED"; U6_NOTE="Scratch + projects quota and writability"

DISCOVERED_ACCOUNT=""
DISCOVERED_CUDA=""
DISCOVERED_BACKEND=""

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

hr()  { printf '\n%s\n' "======================================================================"; }
sec() { hr; printf ' %s\n' "$*"; printf '%s\n' "======================================================================"; }
sub() { printf '\n-- %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
bad()  { printf '   [!] %s\n' "$*"; }
good() { printf '   [ok] %s\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# Run a command, echo its output (stdout+stderr), never abort the script.
soft() {
    local out rc=0
    out="$("$@" 2>&1)" || rc=$?
    if [ -n "${out}" ]; then
        printf '%s\n' "${out}" | sed 's/^/   /'
    fi
    if [ "${rc}" -ne 0 ]; then
        printf '   (exit %d from: %s)\n' "${rc}" "$*"
    fi
    return 0
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --probe-compute) PROBE_COMPUTE=1; shift ;;
        --wait) PROBE_WAIT="${2:?--wait needs seconds}"; shift 2 ;;
        --json) JSON_OUT="${2:?--json needs a path}"; shift 2 ;;
        --probe-partition) PROBE_PARTITION="${2:?}"; shift 2 ;;
        --probe-gres) PROBE_GRES="${2:?}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

sec "ANI_Examination CURC preflight  --  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
note "host        $(hostname -f 2>/dev/null || hostname)"
note "user        ${ANI_USER}"
note "repo        ${REPO_ROOT}"
note "scratch     ${ANI_SCRATCH}"
note "projects    ${ANI_PROJECTS}"
note "shell       ${BASH_VERSION:-unknown}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    bad "you appear to be INSIDE a Slurm job (JOB_ID=${SLURM_JOB_ID});"
    bad "run this on a login node for the full picture."
fi

# ---------------------------------------------------------------------------
# Lmod / module stack
# ---------------------------------------------------------------------------
sec "MODULE STACK (Lmod)"

module_available=0
if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
    for init in "${LMOD_PKG:-/nonexistent}/init/bash" /etc/profile.d/lmod.sh \
                /etc/profile.d/z00_lmod.sh /curc/sw/lmod/lmod/init/bash \
                /usr/share/lmod/lmod/init/bash; do
        if [ -r "${init}" ]; then
            # shellcheck disable=SC1090
            . "${init}" 2>/dev/null || true
        fi
        if declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1; then
            break
        fi
    done
fi
if declare -F module >/dev/null 2>&1 || command -v module >/dev/null 2>&1; then
    module_available=1
    good "module command available"
else
    bad "module command NOT available -- Lmod bootstrap failed"
fi

MODULE_AVAIL_TXT=""
if [ "${module_available}" -eq 1 ]; then
    MODULE_AVAIL_TXT="$(module --terse avail 2>&1 || true)"
    if [ -z "${MODULE_AVAIL_TXT}" ]; then
        MODULE_AVAIL_TXT="$(module avail 2>&1 || true)"
    fi
    for want in anaconda miniforge cuda hf-transformers ollama apptainer singularity python nvhpc gcc openmpi; do
        matches="$(printf '%s\n' "${MODULE_AVAIL_TXT}" \
                   | tr ' ' '\n' | grep -i -- "${want}" | grep -v ':$' \
                   | sort -u | head -12 || true)"
        if [ -n "${matches}" ]; then
            sub "modules matching '${want}'"
            printf '%s\n' "${matches}" | sed 's/^/     /'
        else
            sub "modules matching '${want}'"
            note "  (none found)"
        fi
    done
else
    note "skipping module search"
fi

# ---------------------------------------------------------------------------
# U1: account + allocation
# ---------------------------------------------------------------------------
sec "U1  SLURM ACCOUNT AND ALLOCATION"

ACCOUNT_COUNT=0
if have sacctmgr; then
    sub "sacctmgr associations for ${ANI_USER}"
    soft sacctmgr -nP show associations user="${ANI_USER}" \
        format=Account,Cluster,Partition,QOS,GrpTRESMins
    ALL_ACCOUNTS="$(sacctmgr -nP show associations user="${ANI_USER}" format=Account 2>/dev/null \
                    | grep -v '^$' | sort -u || true)"
    ACCOUNT_COUNT="$(printf '%s\n' "${ALL_ACCOUNTS}" | grep -c . || true)"
    DISCOVERED_ACCOUNT="$(printf '%s\n' "${ALL_ACCOUNTS}" | head -1)"
else
    bad "sacctmgr not found"
fi

if have sshare; then
    sub "sshare (fairshare + usage)"
    soft sshare -U -u "${ANI_USER}" -o Account,User,RawShares,RawUsage,EffectvUsage,FairShare
fi

if have sacctmgr; then
    sub "QOS definitions (walltime / job limits / usage factor)"
    soft sacctmgr -P show qos format=Name,MaxWall,MaxSubmitJobsPU,MaxTRESPU,UsageFactor,Priority
fi

# CURC ships a helper for allocation balance; name varies by site generation.
for cmd in suacct sbalance alloc_balance curc-allocation; do
    if have "${cmd}"; then
        sub "${cmd}"
        soft "${cmd}"
    fi
done

if [ -n "${DISCOVERED_ACCOUNT}" ]; then
    good "discovered Slurm account: ${DISCOVERED_ACCOUNT}"
    note "export SBATCH_ACCOUNT='${DISCOVERED_ACCOUNT}'   # sbatch honours this"
    U1_STATUS="RESOLVED"
    U1_NOTE="account=${DISCOVERED_ACCOUNT}"
else
    bad "could not discover a Slurm account automatically."
    bad "ask RC support, then: export SBATCH_ACCOUNT=<your-account>"
    U1_NOTE="no account discovered; set SBATCH_ACCOUNT manually"
fi
if [ "${ACCOUNT_COUNT}" -gt 1 ] 2>/dev/null; then
    note "sacctmgr found ${ACCOUNT_COUNT} associations for ${ANI_USER}, not one"
    note "default -- SET SBATCH_ACCOUNT explicitly (the export printed above)."
    note "Confirmed in practice on 2026-08-17: an account with multiple"
    note "associations (ucb-general/ucb736_asc1/ucb738_asc1) needed one picked"
    note "explicitly; check 'sshare' above for fairshare/usage across them"
    note "before choosing which to bill against."
elif [ "${ACCOUNT_COUNT}" -eq 1 ] 2>/dev/null; then
    note "sacctmgr found exactly one association -- an explicit --account= may"
    note "not even be required (a single default association is common); try"
    note "submitting without SBATCH_ACCOUNT before assuming you need to set it,"
    note "and fall back to the export above if sbatch rejects the submission."
fi

# ---------------------------------------------------------------------------
# U2: real GRES inventory
# ---------------------------------------------------------------------------
sec "U2  REAL GPU / GRES INVENTORY PER PARTITION"
note "CURC docs are internally inconsistent here (quick-start says 41 GPUs,"
note "the hardware table implies ~130).  The numbers below are the cluster's"
note "own answer and supersede both."

gres_total_seen=0
if have sinfo; then
    for part in "${GPU_PARTITIONS[@]}"; do
        sub "partition ${part}"
        # Exactly the documented spot-check first.
        soft sinfo -p "${part}" -o "%N %G"
        # Then an aggregate the docs do not give us.
        agg="$(sinfo -h -p "${part}" -N -o '%n|%G|%T' 2>/dev/null | awk -F'|' '
            !seen[$1"|"$2]++ {
                nodes[$1]=1
                g=$2
                m=split(g, parts, ",")
                for (i=1;i<=m;i++) {
                    p=parts[i]
                    sub(/\(.*/, "", p)
                    k=split(p, f, ":")
                    if (f[1]=="gpu" && k>=3 && f[3]+0 > 0) { cnt[f[2]] += f[3]; total += f[3] }
                }
            }
            { state[$3] += 1 }
            END {
                n=0; for (x in nodes) n++
                if (n==0) { print "     (no nodes visible in this partition)"; exit }
                printf "     nodes visible: %d   total GPU units: %d\n", n, total
                for (t in cnt) printf "       gres type %-26s x %d\n", t, cnt[t]
                for (s in state) printf "       node state %-26s x %d\n", s, state[s]
            }' || true)"
        if [ -n "${agg}" ]; then
            printf '%s\n' "${agg}"
            gres_total_seen=1
        fi
    done
    if [ "${gres_total_seen}" -eq 1 ]; then
        U2_STATUS="RESOLVED"
        U2_NOTE="sinfo answered for $(printf '%s ' "${GPU_PARTITIONS[@]}")"
    else
        U2_NOTE="sinfo returned nothing for the GPU partitions"
    fi

    sub "partition limits (scontrol)"
    for part in "${GPU_PARTITIONS[@]}"; do
        line="$(scontrol show partition "${part}" 2>/dev/null \
                | tr ' ' '\n' \
                | grep -E '^(MaxTime|DefaultTime|TotalNodes|TotalCPUs|State|QoS|AllowQos|AllowAccounts)=' \
                | tr '\n' ' ' || true)"
        if [ -n "${line}" ]; then
            note "${part}: ${line}"
        else
            note "${part}: (not visible)"
        fi
    done
    note "QoS= above is the partition DEFAULT; AllowQos= is the full permitted"
    note "list -- that is what --qos= must name. CONFIRMED 2026-08-17 on this"
    note "account: it is 'gpu-normal'/'gpu-long' (add 'gpu-testing' on aa100/"
    note "ami100) -- bare 'normal'/'long'/'testing' do NOT appear in any GPU"
    note "partition's AllowQos= here, contrary to an earlier inference this repo"
    note "made from a sibling project's aa100 usage. array_generate.sbatch and"
    note "serve_vllm.sbatch default to the gpu-* names now, marked confirmed."
    note "If YOUR AllowQos= above disagrees with that, trust what you just read"
    note "over this note -- it is account-specific and can differ from ours."
    note ""
    note "ALSO CONFIRMED THE HARD WAY: gpu-testing's MaxTRESPU grant"
    note "(gres/gpu:a100_3g.20gb=1,gres/gpu:mi100=1) is not just a concurrency"
    note "cap on those two GRES types -- a real submission with"
    note "--gres=gpu:a100-40gb:1 --qos=gpu-testing failed (Slurm error 7: "
    note "'Valid GRES types ... are: a100_3g.20gb'). Under gpu-testing on aa100,"
    note "a100_3g.20gb is the ONLY legal GRES type (mi100 on ami100)."
else
    bad "sinfo not found -- are you on a login node with the Slurm client?"
fi

# ---------------------------------------------------------------------------
# U3: billing weights
# ---------------------------------------------------------------------------
sec "U3  SU BILLING WEIGHTS (the unpublished acceleration factors)"
note "Only 108.6 SU per A100-GPU-hour is published.  H200 / RTX Pro 6000 /"
note "GH200 factors are NOT.  Slurm itself knows, via TRESBillingWeights."

billing_seen=0
if have scontrol; then
    for part in "${GPU_PARTITIONS[@]}"; do
        w="$(scontrol show partition "${part}" 2>/dev/null \
             | tr ' ' '\n' | grep -i '^TRESBillingWeights=' || true)"
        if [ -n "${w}" ] && [ "${w}" != "TRESBillingWeights=(null)" ]; then
            note "${part}: ${w}"
            billing_seen=1
        else
            note "${part}: TRESBillingWeights not set/visible"
        fi
    done
fi
if have sacctmgr; then
    sub "TRES types tracked by the accounting DB"
    soft sacctmgr -nP show tres format=Type,Name,ID
fi
if [ "${billing_seen}" -eq 1 ]; then
    U3_STATUS="RESOLVED"
    U3_NOTE="TRESBillingWeights read from scontrol -- transcribe into slurm/su_estimate.py"
else
    U3_NOTE="no TRESBillingWeights visible; ask RC support for the H200/RTXPro factors"
    bad "Until this resolves, budget with slurm/su_estimate.py's 'high' (3x) column."
fi

# ---------------------------------------------------------------------------
# U4: compute-node outbound internet
# ---------------------------------------------------------------------------
sec "U4  OUTBOUND INTERNET"

probe_urls=(https://huggingface.co https://pypi.org)
sub "from THIS node ($(hostname -s 2>/dev/null || echo '?'))"
login_net_ok=1
if have curl; then
    for u in "${probe_urls[@]}"; do
        if curl -sS -o /dev/null -m 15 -w '   %{http_code}  '"${u}"'\n' "${u}" 2>/dev/null; then
            :
        else
            bad "no route to ${u}"
            login_net_ok=0
        fi
    done
else
    bad "curl not found; cannot test"
    login_net_ok=0
fi

if [ "${PROBE_COMPUTE}" -eq 1 ]; then
    sub "from a COMPUTE node (submitting a short ${PROBE_PARTITION} gpu-testing job)"
    if ! have sbatch; then
        bad "sbatch not found; cannot probe"
    else
        probe_dir="${ANI_SCRATCH}/ani_preflight"
        if ! mkdir -p "${probe_dir}" 2>/dev/null; then
            probe_dir="${TMPDIR:-/tmp}/ani_preflight_${ANI_USER}"
            mkdir -p "${probe_dir}"
        fi
        stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
        result="${probe_dir}/netprobe_${stamp}.txt"
        # Seeded with a harmless flag so the array is never empty: expanding an
        # empty array under `set -u` is an error on bash <= 4.3.
        acct_args=(--comment=ani-preflight-net)
        if [ -n "${DISCOVERED_ACCOUNT}" ]; then
            acct_args+=(--account="${DISCOVERED_ACCOUNT}")
        elif [ -n "${SBATCH_ACCOUNT:-}" ]; then
            acct_args+=(--account="${SBATCH_ACCOUNT}")
        fi
        wrap_cmd="{
  echo \"node=\$(hostname -f)\"
  echo \"date=\$(date -u +%FT%TZ)\"
  for u in https://huggingface.co https://pypi.org; do
    code=\$(curl -sS -o /dev/null -m 20 -w '%{http_code}' \"\$u\" 2>&1) || code=FAIL
    echo \"url=\$u code=\$code\"
  done
  echo \"nvidia_smi=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tr '\\n' ';')\"
  echo \"cuda_driver=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)\"
  echo \"apptainer=\$(command -v apptainer || command -v singularity || echo none)\"
  echo DONE
} > '${result}' 2>&1"
        jid=""
        if jid="$(sbatch --parsable \
                    --job-name=ani-preflight-net \
                    --partition="${PROBE_PARTITION}" \
                    --qos=gpu-testing \
                    --gres="gpu:${PROBE_GRES}:1" \
                    --time=00:05:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
                    --output="${probe_dir}/netprobe_${stamp}.slurm.out" \
                    "${acct_args[@]}" \
                    --wrap="${wrap_cmd}" 2>&1)"; then
            note "submitted probe job ${jid}; waiting up to ${PROBE_WAIT}s"
            waited=0
            while [ "${waited}" -lt "${PROBE_WAIT}" ]; do
                if [ -s "${result}" ] && grep -q '^DONE$' "${result}" 2>/dev/null; then
                    break
                fi
                sleep 10
                waited=$((waited + 10))
            done
            if [ -s "${result}" ] && grep -q '^DONE$' "${result}" 2>/dev/null; then
                sed 's/^/   /' "${result}"
                if grep -qE 'code=(200|30[0-9])' "${result}"; then
                    good "COMPUTE NODES HAVE OUTBOUND INTERNET"
                    U4_STATUS="RESOLVED"
                    U4_NOTE="compute node reached the hub (see ${result})"
                else
                    good "compute node reachable; hub calls BLOCKED"
                    U4_STATUS="RESOLVED"
                    U4_NOTE="compute node has NO hub access -- stage weights from login/DTN, run HF_HUB_OFFLINE=1"
                fi
                if grep -q 'apptainer=none' "${result}"; then
                    note "apptainer NOT present on the compute node"
                else
                    note "apptainer present on the compute node"
                fi
            else
                bad "probe job did not finish within ${PROBE_WAIT}s"
                note "check later:  cat ${result}"
                note "queue state:  squeue -j ${jid}"
                U4_NOTE="probe ${jid} still pending; re-check ${result}"
            fi
        else
            bad "probe submission failed:"
            printf '%s\n' "${jid}" | sed 's/^/     /'
            U4_NOTE="probe submission failed (account/QOS?)"
        fi
    fi
else
    note "compute-node internet NOT tested (re-run with --probe-compute)."
    note "CURC documents this as unverified; login-node reachability does NOT imply"
    note "compute-node reachability.  Until confirmed, treat compute nodes as offline:"
    note "stage weights from a login/DTN node and run with HF_HUB_OFFLINE=1."
    U4_NOTE="not tested (needs --probe-compute); assuming compute nodes are OFFLINE"
fi
if [ "${login_net_ok}" -eq 1 ]; then
    good "login node can reach the hub -- staging from here will work"
fi

# ---------------------------------------------------------------------------
# U5: runtime (Apptainer, CUDA, vLLM)
# ---------------------------------------------------------------------------
sec "U5  RUNTIME: APPTAINER / CUDA / vLLM"

apptainer_ok=0
for c in apptainer singularity; do
    if have "${c}"; then
        good "${c} found: $(command -v "${c}")"
        soft "${c}" --version
        apptainer_ok=1
    fi
done
if [ "${apptainer_ok}" -eq 0 ]; then
    if [ "${module_available}" -eq 1 ] && printf '%s' "${MODULE_AVAIL_TXT}" | grep -qi -e apptainer -e singularity; then
        note "not on PATH, but a module exists -- try: module load apptainer"
    else
        bad "not on PATH on THIS (login) node, and no module found here."
        note "CONFIRMED 2026-08-17: this check only runs on the login node, and"
        note "apptainer was found at /usr/bin/apptainer on a COMPUTE node during"
        note "a real --probe-compute job (netprobe log). Login-node absence does"
        note "NOT mean compute-node absence -- if U5's pip/conda install path"
        note "fails, an Apptainer image is a real, available fallback; verify"
        note "with 'which apptainer' inside an interactive compute allocation"
        note "before assuming it is unavailable."
    fi
fi

sub "CUDA"
if have nvcc; then
    DISCOVERED_CUDA="$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' | head -1 || true)"
    good "nvcc release ${DISCOVERED_CUDA:-unknown}"
else
    note "nvcc not on PATH (expected on a login node without a cuda module loaded)"
fi
if have nvidia-smi; then
    soft nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    note "nvidia-smi absent (normal on a login node)"
fi
note "Requirement check: H200 needs CUDA >12.0; RTX Pro 6000 needs CUDA >12.8."
if [ "${module_available}" -eq 1 ]; then
    cuda_mods="$(printf '%s\n' "${MODULE_AVAIL_TXT}" | tr ' ' '\n' | grep -iE '^cuda/' | sort -u || true)"
    if [ -n "${cuda_mods}" ]; then
        note "cuda modules on offer:"
        printf '%s\n' "${cuda_mods}" | sed 's/^/     /'
    fi
fi

sub "vLLM"
py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [ -n "${py}" ]; then
    note "python: ${py}  ($("${py}" --version 2>&1))"
    if "${py}" -c 'import vllm, sys; sys.stdout.write(vllm.__version__)' >/dev/null 2>&1; then
        good "vLLM importable: $("${py}" -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
        DISCOVERED_BACKEND="vllm"
    else
        bad "vLLM NOT importable in this python."
        note "vLLM is NOT in the CURC module stack; it must be self-installed"
        note "(uv/pip in a venv) or containerised via Apptainer."
        if "${py}" -c 'import transformers' >/dev/null 2>&1; then
            good "transformers importable -- the HF fallback backend will work"
            DISCOVERED_BACKEND="hf"
        else
            bad "transformers NOT importable either; neither backend is ready"
            DISCOVERED_BACKEND="none"
        fi
    fi
    if "${py}" -c 'import torch' >/dev/null 2>&1; then
        note "torch: $("${py}" -c 'import torch;print(torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null)"
    else
        note "torch not importable in this python"
    fi
else
    bad "no python found"
    DISCOVERED_BACKEND="none"
fi

if [ -n "${DISCOVERED_BACKEND}" ] && [ "${DISCOVERED_BACKEND}" != "none" ]; then
    U5_STATUS="RESOLVED"
    U5_NOTE="backend=${DISCOVERED_BACKEND}; apptainer=$( [ "${apptainer_ok}" -eq 1 ] && echo yes || echo no ); cuda=${DISCOVERED_CUDA:-unknown}"
else
    U5_NOTE="no usable inference backend yet (install vllm, or transformers+torch)"
fi

# ---------------------------------------------------------------------------
# U6: storage
# ---------------------------------------------------------------------------
sec "U6  FILESYSTEMS AND QUOTA"
note "CURC facts: /home 2 GB; /projects 250 GB (backed up);"
note "/scratch/alpine 10 TB, NOT backed up, purged 90 days after file CREATION."
note "Touching a file does NOT reset the purge clock -- only re-copying does."

quota_seen=0
for c in curc-quota curcquota quota; do
    if have "${c}"; then
        sub "${c}"
        soft "${c}"
        quota_seen=1
        break
    fi
done
[ "${quota_seen}" -eq 0 ] && bad "no quota command found (curc-quota / quota)"

sub "df"
for p in "${ANI_SCRATCH}" "${ANI_PROJECTS}" "${HOME}"; do
    if [ -d "${p}" ]; then
        soft df -h "${p}"
    else
        bad "${p} does not exist"
    fi
done

sub "writability"
writable_ok=1
for p in "${ANI_SCRATCH}" "${ANI_PROJECTS}"; do
    probe="${p}/.ani_write_probe.$$"
    if mkdir -p "${p}" 2>/dev/null && : > "${probe}" 2>/dev/null; then
        rm -f "${probe}" 2>/dev/null || true
        good "writable: ${p}"
    else
        bad "NOT writable: ${p}"
        writable_ok=0
    fi
done
if [ "${writable_ok}" -eq 1 ] && [ "${quota_seen}" -eq 1 ]; then
    U6_STATUS="RESOLVED"
    U6_NOTE="scratch+projects writable; quota reported above"
elif [ "${writable_ok}" -eq 1 ]; then
    U6_STATUS="PARTIAL"
    U6_NOTE="paths writable but no quota command; check usage manually"
else
    U6_NOTE="scratch and/or projects not writable -- fix before staging weights"
fi

sub "existing staged weights"
stage_manifest="${ANI_MODEL_ROOT:-${ANI_SCRATCH}/models}/.ani_stage_manifest.jsonl"
if [ -r "${stage_manifest}" ]; then
    good "stage manifest: ${stage_manifest}"
    soft bash "${SCRIPT_DIR}/stage_weights.sh" --list
else
    note "none yet (run slurm/stage_weights.sh from a login/DTN node)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
badge() {
    case "$1" in
        RESOLVED) printf 'RESOLVED  ' ;;
        PARTIAL)  printf 'PARTIAL   ' ;;
        *)        printf 'UNRESOLVED' ;;
    esac
}

sec "MUST-CONFIRM SUMMARY"
printf '   %-3s %-10s %-8s %s\n' "ID" "STATUS" "TOPIC" "DETAIL"
printf '   %s\n' "----------------------------------------------------------------------"
printf '   %-3s %s %-8s %s\n' "U1" "$(badge "${U1_STATUS}")" "ACCOUNT"  "${U1_NOTE}"
printf '   %-3s %s %-8s %s\n' "U2" "$(badge "${U2_STATUS}")" "GRES"     "${U2_NOTE}"
printf '   %-3s %s %-8s %s\n' "U3" "$(badge "${U3_STATUS}")" "BILLING"  "${U3_NOTE}"
printf '   %-3s %s %-8s %s\n' "U4" "$(badge "${U4_STATUS}")" "INTERNET" "${U4_NOTE}"
printf '   %-3s %s %-8s %s\n' "U5" "$(badge "${U5_STATUS}")" "RUNTIME"  "${U5_NOTE}"
printf '   %-3s %s %-8s %s\n' "U6" "$(badge "${U6_STATUS}")" "STORAGE"  "${U6_NOTE}"

resolved=0
for s in "${U1_STATUS}" "${U2_STATUS}" "${U3_STATUS}" "${U4_STATUS}" "${U5_STATUS}" "${U6_STATUS}"; do
    [ "${s}" = "RESOLVED" ] && resolved=$((resolved + 1))
done
printf '\n   %d / 6 resolved.\n' "${resolved}"

if [ "${resolved}" -lt 6 ]; then
    sub "NEXT ACTIONS"
    [ "${U1_STATUS}" != "RESOLVED" ] && note "U1: ask RC support for your account, then export SBATCH_ACCOUNT=<acct>"
    [ "${U2_STATUS}" != "RESOLVED" ] && note "U2: re-run on a login node where sinfo can see the GPU partitions"
    [ "${U3_STATUS}" != "RESOLVED" ] && note "U3: ask RC support for the H200/RTXPro SU factors; budget at 3x A100 meanwhile"
    [ "${U4_STATUS}" != "RESOLVED" ] && note "U4: re-run with --probe-compute (uses ~0.2 SU on gpu-testing)"
    [ "${U5_STATUS}" != "RESOLVED" ] && note "U5: create the venv and pip install vllm (or transformers+torch); see the runbook"
    [ "${U6_STATUS}" != "RESOLVED" ] && note "U6: confirm scratch/projects paths exist and are writable"
fi

if [ -n "${JSON_OUT}" ]; then
    mkdir -p "$(dirname -- "${JSON_OUT}")" 2>/dev/null || true
    cat > "${JSON_OUT}" <<JSON
{
  "generated_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "host": "$(hostname -f 2>/dev/null || hostname)",
  "user": "${ANI_USER}",
  "scratch": "${ANI_SCRATCH}",
  "projects": "${ANI_PROJECTS}",
  "discovered_account": "${DISCOVERED_ACCOUNT}",
  "discovered_cuda": "${DISCOVERED_CUDA}",
  "discovered_backend": "${DISCOVERED_BACKEND}",
  "resolved_count": ${resolved},
  "unknowns": {
    "U1_ACCOUNT":  {"status": "${U1_STATUS}", "note": "${U1_NOTE}"},
    "U2_GRES":     {"status": "${U2_STATUS}", "note": "${U2_NOTE}"},
    "U3_BILLING":  {"status": "${U3_STATUS}", "note": "${U3_NOTE}"},
    "U4_INTERNET": {"status": "${U4_STATUS}", "note": "${U4_NOTE}"},
    "U5_RUNTIME":  {"status": "${U5_STATUS}", "note": "${U5_NOTE}"},
    "U6_STORAGE":  {"status": "${U6_STATUS}", "note": "${U6_NOTE}"}
  }
}
JSON
    printf '\n   wrote %s\n' "${JSON_OUT}"
fi

hr
exit 0
