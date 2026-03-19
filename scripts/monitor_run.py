#!/usr/bin/env python3
"""
Monitor a live training run and append periodic snapshots to a log file.

Usage:
  python scripts/monitor_run.py --pid 520361 --log-file sessions/run26_6hr.log
  python scripts/monitor_run.py --pid 520361 --log-file sessions/run26_6hr.log \
      --output sessions/run26_hourly_checks.log --interval 3600

Each snapshot records:
  - current process state from `ps`
  - one-line GPU summary from `nvidia-smi` (if available)
  - tail of the training log

The monitor stops automatically once the target PID exits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append periodic process/GPU/log snapshots for a training run."
    )
    parser.add_argument("--pid", type=int, required=True, help="PID of the process to monitor.")
    parser.add_argument(
        "--log-file",
        required=True,
        help="Training log to tail, e.g. sessions/run26_6hr.log",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Snapshot log path. Default: <log-file stem>_checks.log next to the log file.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between snapshots. Default: 3600.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=12,
        help="How many log lines to include per snapshot. Default: 12.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human label to include in the snapshot header.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write one snapshot and exit.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_command(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return f"<command not found: {argv[0]}>"

    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip()
    if output:
        return output
    return f"<no output; exit_code={completed.returncode}>"


def format_ps(pid: int) -> str:
    return run_command(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,pgid=,stat=,etime=,pcpu=,pmem=,cmd="]
    )


def format_gpu() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ]
    )


def tail_lines(path: Path, count: int) -> str:
    if not path.exists():
        return f"<log file does not exist: {path}>"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = deque(handle, maxlen=count)
    except OSError as exc:
        return f"<failed to read log: {exc}>"
    text = "".join(lines).rstrip()
    return text if text else "<log file is empty>"


def write_snapshot(
    out_handle,
    *,
    pid: int,
    label: str | None,
    alive: bool,
    log_path: Path,
    tail_count: int,
) -> None:
    now = dt.datetime.now().isoformat()
    label_suffix = f" label={label}" if label else ""
    out_handle.write(f"\n=== snapshot {now} alive={alive} pid={pid}{label_suffix} ===\n")
    out_handle.write("[ps]\n")
    out_handle.write(format_ps(pid) + "\n")
    out_handle.write("[nvidia-smi]\n")
    out_handle.write(format_gpu() + "\n")
    out_handle.write("[log tail]\n")
    out_handle.write(tail_lines(log_path, tail_count) + "\n")
    out_handle.flush()


def default_output_path(log_path: Path) -> Path:
    return log_path.with_name(f"{log_path.stem}_checks.log")


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2
    if args.tail_lines <= 0:
        print("--tail-lines must be positive", file=sys.stderr)
        return 2

    log_path = resolve_path(args.log_file)
    out_path = resolve_path(args.output) if args.output else default_output_path(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as out:
        start = dt.datetime.now().isoformat()
        label_suffix = f" label={args.label}" if args.label else ""
        out.write(f"=== monitor start {start} pid={args.pid}{label_suffix} ===\n")
        out.flush()

        while True:
            alive = process_alive(args.pid)
            write_snapshot(
                out,
                pid=args.pid,
                label=args.label,
                alive=alive,
                log_path=log_path,
                tail_count=args.tail_lines,
            )
            if args.once or not alive:
                stop = dt.datetime.now().isoformat()
                reason = "once" if args.once else "pid not alive"
                out.write(f"=== monitor stop {stop} pid={args.pid} reason={reason} ===\n")
                out.flush()
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
