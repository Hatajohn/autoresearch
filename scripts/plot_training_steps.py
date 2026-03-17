#!/usr/bin/env python3
"""
Plot training step data from a run log: loss, pc_loss, grad_norm vs time.

Usage:
  python scripts/plot_training_steps.py [LOG_FILE]
  python scripts/plot_training_steps.py sessions/run23_2hr.log

If LOG_FILE is omitted, defaults to sessions/run23_2hr.log.
Log lines must match: step NNNNN (...) | loss: X | pc_loss: X | ... | elapsed: Ns
or ... | remaining: Ns (older logs). Outputs PNGs to reports/ with basename from the log.
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; avoids blank plots in some environments
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SESSIONS = ROOT / "sessions"
REPORTS = ROOT / "reports"


def parse_log(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Parse step lines from the log. Returns (x, loss, pc_loss, grad_norm, x_label)
    where x is the time axis (elapsed or remaining in seconds).
    """
    text = log_path.read_text()
    x_vals, loss, pc_loss, grad_norm = [], [], [], []
    x_label = "Time (s)"

    # Split by step lines: "step 00043 (25.4%) | ..."
    for block in re.split(r"(?=step\s+\d{5}\s+\()", text):
        if "loss:" not in block:
            continue
        m_loss = re.search(r"loss:\s+([\d.]+)", block)
        m_pc = re.search(r"pc_loss:\s+([\d.]+)", block)
        m_gn = re.search(r"grad_norm:\s+([\d.]+)", block)
        if not (m_loss and m_pc and m_gn):
            continue

        # Support both "elapsed: Ns" and "remaining: Ns" (older logs)
        m_el = re.search(r"elapsed:\s+([\d.]+)s", block)
        m_rem = re.search(r"remaining:\s+(\d+)s", block)
        if m_el:
            t = int(float(m_el.group(1)))
            if x_label == "Time (s)":
                x_label = "Time elapsed (s)"
        elif m_rem:
            t = int(m_rem.group(1))
            if x_label == "Time (s)":
                x_label = "Time remaining (s)"
        else:
            continue

        x_vals.append(t)
        loss.append(float(m_loss.group(1)))
        pc_loss.append(float(m_pc.group(1)))
        grad_norm.append(float(m_gn.group(1)))

    if not x_vals:
        return np.array([]), np.array([]), np.array([]), np.array([]), x_label

    x_vals = np.array(x_vals)
    loss = np.array(loss)
    pc_loss = np.array(pc_loss)
    grad_norm = np.array(grad_norm)
    # Sort by x so the plot goes left-to-right in time order
    order = np.argsort(x_vals)
    return x_vals[order], loss[order], pc_loss[order], grad_norm[order], x_label


def main():
    parser = argparse.ArgumentParser(description="Plot loss, pc_loss, grad_norm from a training log.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default=str(SESSIONS / "run23_2hr.log"),
        help="Path to log file (e.g. sessions/run23_2hr.log)",
    )
    parser.add_argument(
        "-o", "--out-dir",
        default=str(REPORTS),
        help="Output directory for PNGs (default: reports/)",
    )
    parser.add_argument(
        "-b", "--basename",
        default=None,
        help="Basename for output PNGs (default: log file stem, e.g. run23_2hr → run23_2hr_loss.png). Use e.g. run23 to get run23_loss.png.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basename for output files (default: log stem, or --basename)
    base = args.basename if args.basename is not None else log_path.stem

    x, loss, pc_loss, grad_norm, x_label = parse_log(log_path)
    if len(x) == 0:
        print(
            "Error: no step lines parsed. Log must contain lines with "
            "loss:, pc_loss:, grad_norm:, and either elapsed: Ns or remaining: Ns",
            file=sys.stderr,
        )
        sys.exit(1)

    def save_plot(y, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(x, y, "o-", color="C0", markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)

    save_plot(loss, "Loss (nats)", f"{base} — Loss vs {x_label.replace(' (s)', '')}", f"{base}_loss.png")
    save_plot(pc_loss, "pc_loss", f"{base} — PC loss vs {x_label.replace(' (s)', '')}", f"{base}_pc_loss.png")
    save_plot(grad_norm, "Grad norm", f"{base} — Grad norm vs {x_label.replace(' (s)', '')}", f"{base}_grad_norm.png")

    print(f"Parsed {len(x)} steps. Saved 3 plots to {out_dir}")


if __name__ == "__main__":
    main()
