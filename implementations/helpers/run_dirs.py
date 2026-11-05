"""Timestamped run folders: every producing script writes into <base>/<YYYY-MM-DD_HHMMSS>/, nothing is overwritten.

Rule for all templates:
    producing scripts (extraction, grouping, fitting)  new_run_dir(base) -> a new folder per run
    follow-up scripts (copy, plots, comparisons)       run_dir(base, RUN) -> the chosen run, None = the newest;
                                                       their outputs go into a subfolder of that run
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Timestamped output folders of the templates)

from __future__ import annotations

from datetime import datetime
from pathlib import Path

RUN_FORMAT = "%Y-%m-%d_%H%M%S"


def new_run_dir(base: Path) -> Path:
    """New folder base/<now> for one run (raises if it already exists, i.e. two runs in the same second)."""
    run = Path(base) / datetime.now().strftime(RUN_FORMAT)
    run.mkdir(parents=True)
    return run


def run_dir(base: Path, run: str | None = None) -> Path:
    """Folder of a run: base/run, or the newest run in base for run = None (names sort chronologically).

    Raises:
        FileNotFoundError: If the run (or any run) does not exist.
    """
    base = Path(base)
    if run is not None:
        path = base / run
        if not path.is_dir():
            raise FileNotFoundError(f"run {path} does not exist")
        return path
    runs = sorted(p for p in base.glob("*") if p.is_dir()) if base.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no runs in {base}, run the producing script first")
    return runs[-1]