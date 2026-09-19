"""Resolve local outputs relative to this checkout, independent of shell cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
