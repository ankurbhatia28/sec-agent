"""Repo paths and environment loading. Import this before anything reads env vars."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
EVALS_DIR = ROOT / "evals"

load_dotenv(ROOT / ".env")

CACHE_DIR = Path(os.environ.get("SECRAG_CACHE_DIR", ROOT / ".cache"))
