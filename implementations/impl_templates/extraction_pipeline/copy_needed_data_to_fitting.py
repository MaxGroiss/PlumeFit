"""Copies the extraction outputs the fitting needs into fitting_pipeline/data_needed_for_fitting.

Only what changes with every extraction / grouping run is copied (the vehicle mapping and the merged data are read
in place from data_needed_for_extraction):
    results/<run>/extraction/*.csv                  -> data_needed_for_fitting/extraction/  (peak windows)
    results/<run>/grouping/<grouping run>/<group>/  -> data_needed_for_fitting/templates/   (templates per group)
    source.json                                     which runs were copied (the fitting adds it to its run_config.json)
The two target folders are replaced completely, they only ever hold copies.
"""
# This file contains code/docs created with AI assistance;
# unless stated otherwise, Anthropic models were used
# Individual uses are marked by inline comments stating purpose and extent: AI-Assisted: <Model> ; (Cause)
# AI-Assisted: <Opus 5.5> ; (Copy step between extraction and fitting pipeline)

from __future__ import annotations

import json
import shutil
from pathlib import Path

from implementations.impl_templates.extraction_pipeline.extraction_template import RESULTS_DIR

from implementations.helpers.run_dirs import run_dir

# ============================================================================== SETTINGS (adjust here)
EXTRACTION_RUN: str | None = None   # e.g. "2026-10-18_153000"; None: the newest extraction run
GROUPING_RUN: str | None = None     # grouping run inside the extraction run; None: the newest
TARGET_DIR = Path(__file__).parents[1] / "fitting_pipeline" / "data_needed_for_fitting"

# ============================================================================== AUTOMATIC (nothing to adjust below)

def copy_needed_data(extraction_run: Path, grouping_run: Path, target: Path) -> None:
    """Replace target/extraction and target/templates by the CSVs of the given runs, write target/source.json."""
    for name, src in (("extraction", extraction_run / "extraction"), ("templates", grouping_run)):
        shutil.rmtree(target / name, ignore_errors=True)
        # group_summary.csv stays with the grouping run, only the group folders are templates
        shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("group_summary.csv"))
    source = {"extraction_run": extraction_run.name, "grouping_run": grouping_run.name}
    (target / "source.json").write_text(json.dumps(source, indent=1), encoding="utf-8")


if __name__ == "__main__":
    extraction_run = run_dir(RESULTS_DIR, EXTRACTION_RUN)
    grouping_run = run_dir(extraction_run / "grouping", GROUPING_RUN)
    copy_needed_data(extraction_run, grouping_run, TARGET_DIR)
    groups = sorted(p.name for p in (TARGET_DIR / "templates").iterdir() if p.is_dir())
    print(f"Copied extraction {extraction_run.name}, grouping {grouping_run.name} -> {TARGET_DIR}\n"
          f"  groups: {', '.join(groups)}")