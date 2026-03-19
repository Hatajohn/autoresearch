#!/usr/bin/env python3
"""
Launch a named training profile and optionally start log monitoring.

Examples:
  uv run python3 scripts/run_stage2.py --profile backbone_min --run-id 0
  uv run python3 scripts/run_stage2.py --profile pc_min --run-id 1
  uv run python3 scripts/run_stage2.py --profile pc_min --run-id 2.5 --train-time-budget 3600
  uv run python3 scripts/run_stage2.py --profile pc_min --run-id 3a --depth 8 --width 256
  uv run python3 scripts/run_stage2.py --profile pc_min --run-id 4 --max-epochs 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "sessions"


PROFILES = {
    "backbone_min": {
        "log_suffix": "stage2_backbone_min",
        "monitor_suffix": "stage2_backbone_min_checks",
        "env": {
            "PC_WEIGHT": "0",
            "USE_STOCHASTIC_LAYERS": "0",
            "USE_TORCH_COMPILE": "regional",
            "TRAIN_TIME_BUDGET": "86400",
            "MAX_EPOCHS": "1",
            "VAL_INTERVAL": "300",
            "CHECKPOINT_STEPS": "300",
            "EARLY_STOP_MIN_STEPS": "20",
        },
    },
    "pc_min": {
        "log_suffix": "stage2_pc_min",
        "monitor_suffix": "stage2_pc_min_checks",
        "env": {
            "PC_WEIGHT": "0.02",
            "USE_STOCHASTIC_LAYERS": "0",
            "USE_TORCH_COMPILE": "regional",
            "TRAIN_TIME_BUDGET": "86400",
            "MAX_EPOCHS": "1",
            "VAL_INTERVAL": "300",
            "CHECKPOINT_STEPS": "300",
            "EARLY_STOP_MIN_STEPS": "20",
            "PC_DIAG_INTERVAL": "10",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a named training run.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        required=True,
        help="Named training launch profile.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier used in log/check filenames, e.g. 2, 2.5, pc_min_repeat1.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=1800,
        help="Seconds between monitor snapshots. Default: 1800.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=20,
        help="Lines of log tail to include in each monitor snapshot. Default: 20.",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Launch training only; do not start monitor_run.py.",
    )
    parser.add_argument(
        "--train-time-budget",
        type=int,
        default=None,
        help="Override TRAIN_TIME_BUDGET failsafe for this launch without editing the base profile.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override MAX_EPOCHS for this launch without editing the base profile.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override DEPTH for this launch without editing train.py.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override target model width before HEAD_DIM rounding; sets MODEL_WIDTH for this launch.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment override. Repeat as needed.",
    )
    return parser.parse_args()


def train_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "train.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    own_pid = str(os.getpid())
    return [line for line in lines if not line.startswith(own_pid)]


def make_paths(profile: str, run_id: str) -> tuple[Path, Path]:
    spec = PROFILES[profile]
    log_path = SESSIONS_DIR / f"run{run_id}_{spec['log_suffix']}.log"
    monitor_path = SESSIONS_DIR / f"run{run_id}_{spec['monitor_suffix']}.log"
    return log_path, monitor_path


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env override '{item}'; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env override '{item}'; key is empty.")
        overrides[key] = value
    return overrides


def start_training(env_overrides: dict[str, str], log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(env_overrides)
    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["uv", "run", "train.py"],
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_monitor(
    *,
    pid: int,
    profile: str,
    run_id: str,
    log_path: Path,
    monitor_path: Path,
    interval: int,
    tail_lines: int,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "uv",
            "run",
            "python3",
            "scripts/monitor_run.py",
            "--pid",
            str(pid),
            "--log-file",
            str(log_path),
            "--output",
            str(monitor_path),
            "--interval",
            str(interval),
            "--tail-lines",
            str(tail_lines),
            "--label",
            f"run{run_id}:{profile}",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main() -> int:
    args = parse_args()
    active = train_processes()
    if active:
        print("Refusing to launch: train.py already appears to be running.", file=sys.stderr)
        for line in active:
            print(f"  {line}", file=sys.stderr)
        return 1

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_path, monitor_path = make_paths(args.profile, args.run_id)
    if log_path.exists():
        print(f"Refusing to overwrite existing log: {log_path}", file=sys.stderr)
        return 1
    if monitor_path.exists():
        print(f"Refusing to overwrite existing monitor log: {monitor_path}", file=sys.stderr)
        return 1

    spec = PROFILES[args.profile]
    env_overrides = dict(spec["env"])
    if args.train_time_budget is not None:
        env_overrides["TRAIN_TIME_BUDGET"] = str(args.train_time_budget)
    if args.max_epochs is not None:
        env_overrides["MAX_EPOCHS"] = str(args.max_epochs)
    if args.depth is not None:
        env_overrides["DEPTH"] = str(args.depth)
    if args.width is not None:
        env_overrides["MODEL_WIDTH"] = str(args.width)
    try:
        env_overrides.update(parse_env_overrides(args.env))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    train_proc = start_training(env_overrides, log_path)
    print(f"Started run{args.run_id} profile={args.profile} pid={train_proc.pid}")
    print(f"log: {log_path.relative_to(ROOT)}")
    print("env:")
    for key, value in env_overrides.items():
        print(f"  {key}={value}")

    if args.no_monitor:
        return 0

    monitor_proc = start_monitor(
        pid=train_proc.pid,
        profile=args.profile,
        run_id=args.run_id,
        log_path=log_path,
        monitor_path=monitor_path,
        interval=args.monitor_interval,
        tail_lines=args.tail_lines,
    )
    print(f"monitor_pid: {monitor_proc.pid}")
    print(f"monitor_log: {monitor_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
