#!/usr/bin/env python
"""Unified training entry point for release runs."""
import argparse
import runpy
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TASK_TO_SCRIPT = {
    "text": SCRIPT_DIR / "internal" / "train.py",
    "llava": SCRIPT_DIR / "internal" / "train_llava.py",
    "gemma": SCRIPT_DIR / "internal" / "train_gemma.py",
}

parser = argparse.ArgumentParser(description="Train Latent Memory models")
parser.add_argument("--task", choices=TASK_TO_SCRIPT, required=True)
args, rest = parser.parse_known_args()

sys.argv = [str(TASK_TO_SCRIPT[args.task])] + rest
runpy.run_path(str(TASK_TO_SCRIPT[args.task]), run_name="__main__")