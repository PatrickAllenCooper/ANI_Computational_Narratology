# CURC / Alpine Operator Runbook

Status: operational source of truth for running the open-weights half of this study on CU Boulder Research Computing (Alpine). It covers the harness in `slurm/`. The scientific design lives in `study_design.md` and `tier3_preregistration.md`; this file is only about getting jobs onto GPUs without wasting an allocation.

Everything in `slurm/` reads paths, accounts and usernames from the environment. Nothing is hardcoded, so this runbook does not need editing per-user.

---

## 0. The one-paragraph version

Run `slurm/preflight.sh` on a login node and resolve the six unknowns. Create a venv on `/projects`. Stage weights to `/scratch/alpine` with `slurm/stage_weights.sh` and write down the purge date. Build a JSONL work manifest. Estimate the burn with `slurm/su_estimate.py --plan`. Submit `slurm/array_generate.sbatch` as a resumable array on MIG slices. If a task dies, resubmit the identical array — the per-cell JSON cache makes it idempotent. Use `slurm/serve_vllm.sbatch` only for iterative workloads that cannot be flattened into a manifest.

---

## 1. First-time setup, in order

The order matters. Each step depends on a fact established by the one before it.

### 1.1 Preflight

```bash
git clone <this repo> && cd ANI_Examination
bash slurm/preflight.sh --json preflight.json
```

Read the `MUST-CONFIRM SUMMARY` block at the bottom. It never fails hard and never changes anything; it only reports. Re-run it whenever the cluster changes under you.

### 1.2 The six must-confirm unknowns

The published CURC documentation is incomplete or self-contradictory on six points that materially change how this study should be run. Preflight resolves what it can from the cluster itself; the rest need a support ticket.

| ID | Topic | Why it matters | How preflight resolves it | If it stays UNRESOLVED |
|---|---|---|---|---|
| **U1** | Slurm account | Every `sbatch` needs one; there is no sensible default | `sacctmgr show associations`, `sshare` | Ask RC support, then `export SBATCH_ACCOUNT=<acct>` |
| **U2** | Real GRES inventory | The docs are internally inconsistent — the quick-start says 41 GPUs, the hardware table implies ~130. Array width and queue expectations depend on which is true | `sinfo -p <part> -o "%N %G"` per partition, plus a per-GRES-type aggregate | Re-run on a login node where `sinfo` can see the GPU partitions |
| **U3** | SU billing weights | Only **108.6 SU per A100-GPU-hour** is published. H200 / RTX Pro 6000 / GH200 factors are not. A 3x surprise is the difference between a comfortable study and a spent allocation | `scontrol show partition <p>` → `TRESBillingWeights` | Budget against `su_estimate.py`'s **high (3x)** column and ask RC support |
| **U4** | Compute-node internet | Documented as unverified. If compute nodes cannot reach the hub, every job must run `HF_HUB_OFFLINE=1` against pre-staged weights or it will hang for its whole walltime | `--probe-compute` submits a 5-minute `gpu-testing` job that curls the hub | **Assume offline.** Stage from a login/DTN node. This is the default the harness already takes |
| **U5** | Runtime path | vLLM is **not** in the Lmod module stack. Whether you get vLLM or the HF fallback decides how large the sweeps can be | checks `apptainer`/`singularity`, `nvcc`, `import vllm`, `import transformers` | Install vLLM into the venv, or accept the HF path and shrink the design (§6) |
| **U6** | Storage | `/home` is 2 GB; `/projects` is 250 GB and backed up; `/scratch/alpine` is 10 TB, not backed up, and purged | `curc-quota`, `df`, write probes | Confirm the paths exist and are writable before staging 140 GB |

`--probe-compute` costs roughly 0.2 SU (a 5-minute `gpu-testing` job billed at 10%). Run it once; the answer is worth far more than that.

### 1.3 Environment

```bash
export SBATCH_ACCOUNT=<from U1>        # sbatch honours this; no --account needed
python3 -m venv "/projects/$USER/ani-venv"
"/projects/$USER/ani-venv/bin/pip" install -U pip
"/projects/$USER/ani-venv/bin/pip" install -r requirements.txt   # if present
"/projects/$USER/ani-venv/bin/pip" install vllm                  # optional; see §6
source slurm/env_setup.sh
```

**The venv goes on `/projects`, not `/scratch`.** `/scratch/alpine` is purged 90 days after file creation, and a purged venv fails in a confusing way weeks into a study. Weights go on `/scratch` because 140 GB does not fit in the 250 GB `/projects` quota alongside everything else.

`slurm/env_setup.sh` is sourceable and idempotent. It bootstraps Lmod, loads modules with fallbacks, activates the venv, points `HF_HOME`/`TRANSFORMERS_CACHE` at scratch, sets `HF_HUB_OFFLINE=1`, pins `OMP_NUM_THREADS` to the cores Slurm actually gave you, and resolves the backend switch. It deliberately restores your shell's original option flags on the way out, so sourcing it interactively does not leave you in `set -e`.

### 1.4 Stage weights

From a **login or DTN node** (compute nodes may have no route to the hub):

```bash
bash slurm/stage_weights.sh qwen3-8b llama-3.1-8b
bash slurm/stage_weights.sh --list
```

Aliases come from `scripts/local_backend.py`'s `LOCAL_MODELS` when that module is present, with a built-in fallback table otherwise. Full `org/name` repo ids always work.

Use `--checksums` when you want a `SHA256SUMS` file written for later `--verify-checksums` runs. It is off by default because hashing 140 GB is slow and `hf download` already validates etags; turn it on for the models you care most about.

### 1.5 Log directory

Slurm will not create the directory its `--output` path lives in, and the job dies at submit time if it is missing:

```bash
mkdir -p slurm_logs
```

To put logs on scratch instead, set `SBATCH_OUTPUT` in your shell — your shell expands `$USER`, Slurm expands `%x`/`%A`/`%a`:

```bash
export SBATCH_OUTPUT="/scratch/alpine/$USER/slurm_logs/%x_%A_%a.out"
export SBATCH_ERROR="/scratch/alpine/$USER/slurm_logs/%x_%A_%a.err"
```

---

## 2. The work manifest

`array_generate.sbatch` is driven by a JSONL file, one cell per line.

```json
{"cache_kind":"bm_gen","model":"qwen3-8b","arm":"raw","item_id":"bm_0007",
 "system":"","user":"Prove that ...","max_tokens":2048,"sample_idx":0}
```

| Field | Required | Meaning |
|---|---|---|
| `model` | yes | local alias or hosted model name; routed by `scripts.local_backend.generate_any` |
| `user` | yes | the user turn |
| `system` | no | system prompt; `""` means the raw arm (no system turn) |
| `arm` | no | arm label, default `raw`; goes into the cache filename and the record |
| `item_id` | no | item identifier (`problem_id`/`id` also accepted) |
| `cache_kind` | no | filename prefix, default `bm_gen` |
| `cache_path` | no | explicit cache path, overriding the naming rule |
| `max_tokens` | no | default 2048 |
| `sample_idx` | no | default 0; retries increment it |
| `temperature`, `seed` | no | forwarded to the local backend only |
| `instrument` + `append_verdict_instruction` | no | opt in to `scripts.verdict_format` (see below) |
| `record_extra` | no | dict merged into the cached record for downstream aggregation |

The cache path follows the repo's existing convention exactly:

```
divergence_study_outputs/{cache_kind}_{safe(model)}_{arm}_{safe(item_id)}.json
```

which for BrokenMath reproduces `bm_gen_{safe(model)}_{arm}_{safe(problem_id)}.json` — the same files `scripts/run_brokenmath.py` reads. The cached record keeps `response` and `arm` as its first keys with unchanged meaning, so every existing aggregator reads these files without modification; additional keys (token counts, latency, backend, `verdict`) are purely additive.

**Prompts are never silently modified.** The verdict instruction from `scripts/verdict_format.py` is appended only when a cell explicitly sets `"append_verdict_instruction": true`. This is deliberate: the committed results were produced without it, and a silently altered prompt would invalidate them.

---

## 3. Estimating the burn before you spend it

```bash
python slurm/su_estimate.py --list-gres
python slurm/su_estimate.py --gres h200_2g.35gb --gpus 1 --cores 4 --hours 8 --array-tasks 16
python slurm/su_estimate.py --plan work_manifest.jsonl --gres h200_2g.35gb --array-tasks 16
```

`--plan` discounts cells whose cache already exists, so re-running it mid-sweep tells you what is *left* to spend.

Read the output with two things in mind:

- **The low column is a floor, not a forecast.** It assumes an H200 bills the same as an A100. That is the optimistic bound, and it is almost certainly wrong. Budget against the high column until U3 is resolved.
- **The headline total assumes MIG slices bill at the full-GPU rate.** Whether CURC prorates a `2g.35gb` slice to 2/7 is unverified; both figures are printed. Assume the worse one.

Once `scontrol show partition ah200` gives you real `TRESBillingWeights`, transcribe them into the constants at the top of `slurm/su_estimate.py` and delete the guesswork.

---

## 4. Running the array

### 4.1 Launch

```bash
source slurm/env_setup.sh
mkdir -p slurm_logs
sbatch --array=0-15%8 --export=ALL,ANI_MANIFEST=$PWD/work_manifest.jsonl \
       slurm/array_generate.sbatch
```

Defaults baked into the file: `--partition=ah200 --gres=gpu:h200_2g.35gb:1 --qos=gpu-normal --time=08:00:00`, `--requeue`, `--open-mode=append`, `--signal=B:USR1@300`, logs at `slurm_logs/%x_%A_%a.{out,err}`. Override anything on the command line; the header comment lists ready-made alternates for the aa100 debug queue, RTX Pro 6000 MIG, and whole-GPU tensor-parallel runs.

Always smoke first, on the cheap queue:

```bash
sbatch --partition=aa100 --qos=gpu-testing --gres=gpu:a100-40gb:1 --time=01:00:00 \
       --array=0-1 --export=ALL,ANI_MANIFEST=$PWD/work_manifest.jsonl,ANI_MAX_CELLS=5 \
       slurm/array_generate.sbatch
```

`gpu-testing` is capped at 1 hour and 5 concurrent jobs, is billed at 10%, and is valid **only on `aa100` and `ami100`**.

### 4.2 Sharding

Each task takes a round-robin stripe: manifest cell *i* goes to task `i % nshards`. Round-robin rather than contiguous blocks because manifests are usually grouped by model and arm, and contiguous blocks would hand one task all the cheap cells and another all the expensive ones.

Shard index and count are derived from `SLURM_ARRAY_TASK_ID`, `_MIN` and `_MAX`. **This has one sharp edge:** re-running a single task with `--array=3` makes `_MIN == _MAX == 3`, so that task computes "shard 0 of 1" and processes the *entire* manifest. It is idempotent, so nothing is corrupted, but it is not what you meant. To re-run one task of a sixteen-way array:

```bash
sbatch --array=3 --export=ALL,ANI_MANIFEST=...,ANI_SHARD=3,ANI_SHARDS=16 \
       slurm/array_generate.sbatch
```

The banner prints the resolved `shard N of M` on every task — check it.

### 4.3 Monitoring

```bash
squeue -u "$USER" -o "%.10i %.9P %.14j %.2t %.10M %.6D %R"
tail -f slurm_logs/ani-gen_<arrayjobid>_0.out
cat "$ANI_CKPT_DIR"/<run>_<arrayjobid>/task_0.json
```

Every checkpoint carries `assigned`, `generated`, `skipped`, `empty`, `errors`, `remaining`, `traversed`, `complete`, `resumable`, `stopped_by`, plus a `cells_per_s` rate. Aggregate progress across an array:

```bash
python - <<'PY'
import json, pathlib, os
d = pathlib.Path(os.environ["ANI_CKPT_DIR"])
for run in sorted(d.iterdir()):
    tot = {"generated":0,"skipped":0,"empty":0,"errors":0,"remaining":0}
    done = 0
    for f in sorted(run.glob("task_*.json")):
        s = json.loads(f.read_text())
        for k in tot: tot[k] += s.get(k, 0)
        done += bool(s.get("complete"))
    print(run.name, tot, f"{done} shards complete")
PY
```

`empty` and `errors` mean different things and want different fixes. `errors` means every attempt raised — a dead endpoint, an OOM, a missing model. `empty` means the backend answered successfully with no text — usually `max_tokens` too small for a reasoning model, or a chat-template problem.

### 4.4 Resuming

**Resubmit the identical array.** That is the whole procedure. Cells whose cache file exists with a non-empty `response` are skipped, so a resubmitted task walks past finished work in seconds and picks up exactly where it stopped.

This works because:

- The cache is written atomically (temp file + `os.replace`), so a job killed mid-write never leaves a half-written JSON.
- Failed and empty cells are **not** cached, so they stay retryable instead of poisoning the cache with `""`.
- Existing cache files are never rewritten, so committed results cannot be invalidated by a re-run.

Three ways a task stops early, all of which checkpoint cleanly:

1. `--signal=B:USR1@300` fires 5 minutes before walltime. The batch script forwards `SIGUSR1` to the worker, which finishes the current cell, checkpoints, and exits 0.
2. A self-imposed deadline computed from `SLURM_JOB_END_TIME` minus `ANI_GRACE_S` (default 240 s) stops the worker even if Slurm's signal is late or lost.
3. `ANI_MAX_CELLS` for smoke runs.

An incomplete shard exits **0**, not non-zero — stopping at a walltime cap is a normal outcome, not a failure, and marking it failed would poison every `sacct` summary you look at afterwards.

`ANI_AUTO_REQUEUE=1` makes a task requeue itself, but **only** when it stopped for a clock reason (`resumable: true` in the checkpoint). A shard that traversed its whole stripe and still has uncached cells has cells that genuinely fail; requeueing that would spin forever and burn SU on a bug.

---

## 5. Why MIG slices, and why not one long job

An 8B model in bf16 needs about 16 GB. An `h200_2g.35gb` slice gives 35 GB — the weights plus a comfortable KV cache. So the 8B production runs go on MIG slices, as many single-GPU resumable array tasks of 4–8 hours each.

The alternative — one standing 4×H200 `gpu-long` job that everything talks to — is much worse:

- At the A100 factor, 4 GPUs × 168 hours ≈ **73,000 SU per week**, and that is the *optimistic* bound; if the H200 factor is 2–3x it is 150k–220k. A large fraction of an allocation, much of it spent idle between calls.
- It is a single point of failure with no resume story. A node failure at hour 100 loses everything not already cached.
- Wide, long jobs schedule badly. Many small single-GPU tasks backfill into gaps and start almost immediately.

Array tasks bill only for work actually done, and the per-cell cache means a lost task costs one task's worth of progress.

**Do not quantize the 8B models.** They fit in bf16, and quantization perturbs logprobs enough to contaminate the small propensity shifts this study is trying to measure. The compute saved is not worth the measurement risk.

32B and 70B are the exception: they need whole GPUs and tensor parallelism, and their workloads tend to be iterative. Those go through `serve_vllm.sbatch` under `gpu-long`.

---

## 6. Backend branch: vLLM or HuggingFace

vLLM is **not in the CURC module stack**. It must be pip-installed into the venv (CUDA >12.0 for H200, >12.8 for RTX Pro 6000) or run from an Apptainer image. `slurm/env_setup.sh` therefore treats it as optional:

```bash
export NOT_LOCAL_BACKEND=auto   # default: use vLLM if importable, else hf
export NOT_LOCAL_BACKEND=vllm   # require vLLM (warns and degrades if absent)
export NOT_LOCAL_BACKEND=hf     # force the transformers path
```

`env_setup.sh` exports the *resolved* backend as `ANI_LOCAL_BACKEND` (never `auto`) so `scripts/local_backend.py` does not have to re-probe, and sets `ANI_BACKEND_DEGRADED=1` on the HF path.

### What shrinks if you end up on HF

The HF `transformers` path has no paged attention and no continuous batching. Expect roughly **4–5x slower decode** for the same model on the same GPU. That is not a small inconvenience — it changes what the study can afford:

- **Paraphrase atlases shrink 5–10x.** A full `PARAPHRASE_DIMENSIONS` sweep across every channel × direction × register is a vLLM-only luxury. On HF, sample the atlas: keep every *channel* and *direction* (they carry the hypotheses) and subsample paraphrase indices within each family.
- **Nuisance percentiles need fewer perturbations.** `NUISANCE_PERTURBATIONS` has 20+ entries; on HF, use a fixed random subsample of 6–8 per item and say so in the writeup. The percentile is noisier but still calibrates the effect.
- **Prefer propensity over sampling.** `scripts/propensity.py`'s `verdict_propensity` reads logprobs from a single forward pass instead of sampling many completions. On the HF path this is the single biggest saving available, and it is also lower-variance. Use it wherever a verdict distribution is what you actually want.
- **Ladder later.** Get the 8B results complete on HF before spending anything on 32B/70B.

Record which backend produced every result: the cached record carries a `backend` field for exactly this reason, and vLLM and HF are not guaranteed to be bitwise identical even at temperature 0.

---

## 7. The persistent endpoint

Use `serve_vllm.sbatch` when the workload cannot be flattened into a manifest — the prompt optimizers in `scripts/syco_loss.py` and `scripts/run_phase14_*.py` decide what to ask next based on what came back, so there is nothing to enumerate up front.

```bash
sbatch --export=ALL,ANI_SERVE_MODEL=qwen3-32b slurm/serve_vllm.sbatch
sbatch --gres=gpu:h200:4 --cpus-per-task=32 --mem=400G --time=48:00:00 \
       --export=ALL,ANI_SERVE_MODEL=llama-3.3-70b slurm/serve_vllm.sbatch
```

The server writes `host:port` as the first line of `$ANI_ENDPOINT_DIR/vllm_<jobid>.txt`, which is exactly where `scripts/local_backend.py`'s `endpoint_file_candidates()` looks (it scans that directory newest-mtime-first and reads the first non-comment line). Clients need only:

```bash
source slurm/env_setup.sh          # exports ANI_ENDPOINT_DIR
python -m scripts.run_brokenmath --generators qwen3-32b
```

Details worth knowing:

- The endpoint file is written **only after** the health probe passes, so no client ever sees a half-open server.
- A rich JSON sidecar is written as a **dotfile** (`.vllm_<jobid>.json`) on purpose — the endpoint scan ignores dotfiles, so it can never be misread as a `host:port` line.
- The heartbeat re-touches the endpoint file every 60 s, so when several files exist the newest is always the live server.
- Cleanup on exit removes both files, and the server is served under **both** the repo id and the alias, so either spelling resolves.
- To bypass discovery: `export NOT_VLLM_BASE_URL=http://host:port/v1`.
- A hard self-imposed lifetime (from `SLURM_JOB_END_TIME` minus `ANI_SERVE_GRACE_S`) shuts the server down cleanly before Slurm kills it, so the GPUs are released properly.

---

## 8. Scratch purge discipline

**`/scratch/alpine` is purged 90 days after file CREATION, not last access.** `touch` does not help. Reading the files does not help. `ls` does not help. The only reset is making the bytes into new files.

```bash
bash slurm/stage_weights.sh --list          # states and days remaining
bash slurm/stage_weights.sh --due           # exits 3 if anything is due soon
bash slurm/stage_weights.sh --restage all   # reset the clock
```

`--restage` defaults to a local `cp -R` into a sibling directory, then an atomic swap. Two deliberate choices there:

- `cp -R`, **not** `cp -a` or `cp -p`. Preserving mtimes would defeat the whole exercise if the purge sweep keys on mtime rather than birth time.
- Copy, not re-download. A local copy is far faster than pulling 140 GB again and needs no outbound internet, which matters if U4 came back negative. It needs transient free space equal to the model size; the script checks this and refuses rather than filling the filesystem. Use `--restage-mode download` when space is tight.

Put `stage_weights.sh --due` in a monthly reminder. Losing 140 GB of weights halfway through a study is a bad week, and it happens silently.

Nothing on `/scratch` is backed up. Results belong in `divergence_study_outputs/` in the repo (on `/projects`), which is where the harness writes them by default.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: ... Invalid account` | U1 unresolved | `export SBATCH_ACCOUNT=<acct>`; `sacctmgr show associations user=$USER` |
| Job dies instantly, no log file | `slurm_logs/` does not exist | `mkdir -p slurm_logs`, or set `SBATCH_OUTPUT` |
| `Requested node configuration is not available` | Wrong GRES string, or that QOS is invalid on that partition | `sinfo -p <part> -o "%N %G"`; `gpu-testing` is aa100/ami100 only |
| Job hangs for the whole walltime at model load | Compute node has no route to the hub (U4) and `HF_HUB_OFFLINE` was unset | Stage weights first; keep `HF_HUB_OFFLINE=1` (the default) so it fails fast instead |
| `OSError: ... is not a local folder` | Weights not staged, or purged | `bash slurm/stage_weights.sh --list`, then re-stage |
| Task processes the whole manifest instead of its stripe | Re-ran a single task, so `_MIN == _MAX` | Pass `ANI_SHARD` and `ANI_SHARDS` explicitly (§4.2) |
| `[worker] ... errors=N` | Every attempt raised — dead endpoint, OOM, missing model | Read the traceback in the log; check `nvidia-smi` output in the banner |
| `[worker] ... empty=N` | Backend answered with no text | Raise `max_tokens` (reasoning models need thousands); check the chat template |
| Array re-runs generate nothing and exit at once | Working as designed — everything is cached | Confirm with `skipped=` in the checkpoint |
| Requeue loop burning SU | Should be impossible: auto-requeue requires `resumable: true` | Check the checkpoint; if `traversed: true` the cells genuinely fail |
| vLLM job exits at startup | vLLM not installed, or CUDA too old | See the FATAL message it prints; H200 needs >12.0, RTX Pro 6000 >12.8 |
| Client cannot find the endpoint | `ANI_ENDPOINT_DIR` not exported, or a stale file | `source slurm/env_setup.sh`; stale entries fail the health probe and fall through |
| SU burn much higher than estimated | U3 — the H200 factor is not the A100 factor | Read the real `TRESBillingWeights` and update `su_estimate.py` |
| `env_setup.sh` left my shell in a weird state | Should be impossible — it restores your option flags | Report it; check `ANI_QUIET=1` if the banner is the only complaint |

### Useful accounting queries

```bash
sacct -u "$USER" -S $(date -d '-7 days' +%F) \
      -o JobID%18,JobName%16,State,Elapsed,AllocTRES%48,Priority
sreport cluster AccountUtilizationByUser start=$(date -d '-30 days' +%F) -t hours
scontrol show partition ah200 | tr ' ' '\n' | grep -i TRESBillingWeights
```

---

## 10. File map

| File | Role |
|---|---|
| `slurm/preflight.sh` | One-shot login-node audit; resolves the six unknowns. Run first, never destructive. |
| `slurm/env_setup.sh` | Sourceable environment: modules, venv, HF cache, offline flags, backend switch. |
| `slurm/stage_weights.sh` | Stage/verify/re-stage weights; owns the 90-day purge clock. |
| `slurm/array_generate.sbatch` | The core harness: resumable single-GPU generation array. |
| `slurm/serve_vllm.sbatch` | Persistent vLLM endpoint for iterative workloads and 32B/70B. |
| `slurm/su_estimate.py` | SU calculator, single-shape and `--plan` modes. |

All shell scripts are `set -euo pipefail` and safe to run repeatedly. `preflight.sh`, `stage_weights.sh --list/--due/--verify`, and `su_estimate.py` change nothing and can be run at any time.
