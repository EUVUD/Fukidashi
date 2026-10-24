"""Paths shared by the API and the scripts. The DB and the ledger live under gitignored runs/."""

from __future__ import annotations

import os
from pathlib import Path

from .datasets.openmantra import DEFAULT_ROOT as DATASET_ROOT

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("FUKIDASHI_DB") or ROOT / "runs" / "fukidashi.db")
LEDGER_PATH = ROOT / "runs" / "usage.jsonl"

__all__ = ["DATASET_ROOT", "DB_PATH", "LEDGER_PATH", "ROOT"]
