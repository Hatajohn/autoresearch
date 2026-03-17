#!/usr/bin/env python3
"""Convenience wrapper: plot run23_2hr.log and write run23_*.png for report links."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Invoke the generalized script
sys.path.insert(0, str(ROOT))
from scripts.plot_training_steps import main

if __name__ == "__main__":
    sys.argv = ["plot_run23_steps.py", str(ROOT / "sessions" / "run23_2hr.log"), "--basename", "run23"]
    main()
