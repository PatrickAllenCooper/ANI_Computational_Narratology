"""
scripts/detach_run.py -- launch a long run that survives its launcher.

WHY THIS EXISTS
---------------
Long `run_stance_factorial` jobs were dying silently at 95%+ completion,
three times, always shortly after the shell that launched them was torn down.
`nohup ... &` was not enough: nohup only ignores SIGHUP, while the harness
reaps the whole PROCESS GROUP, which takes the child with it regardless.

`start_new_session=True` calls setsid(2) in the child, putting it in a fresh
session and process group so a kill aimed at the launcher's group cannot reach
it. macOS has no `setsid` binary, hence doing it from Python.

Also forces unbuffered output (`-u`): under nohup, Python buffers stdout when it
is not a tty, which made one run look hung for 18 minutes with an empty log when
it was in fact working.

USAGE
    python3 scripts/detach_run.py <logfile> <command> [args...]

    python3 scripts/detach_run.py /tmp/run.log \
        python3 -m scripts.run_stance_factorial --instrument brokenmath ...

Prints the detached PID. Poll the logfile for progress; the process outlives
this script, the shell, and the agent session.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip())
        return 2
    log_path, cmd = argv[1], argv[2:]

    # -u on any python child: unbuffered, so the log reflects live progress.
    if cmd[0].startswith("python") and "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,      # setsid(2): survives process-group kills
            env=env,
            cwd=os.getcwd(),
        )
    print(f"detached pid={proc.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
