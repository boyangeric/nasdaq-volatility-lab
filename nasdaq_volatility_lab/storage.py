"""Read the shared feature artifact under one schema and integrity contract."""

import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .features import FEATURE_COLUMNS, validate_feature_table


def load_feature_table(path: Path) -> tuple[pd.DataFrame, dict]:
    """Reject stale bytes and unexpected model inputs before any split or fit.

    A file hash alone cannot stop a modified schema from including the target
    among model inputs. Check the explicit feature whitelist as well.
    """
    path = Path(path)
    if not path.exists() or not path.with_name("feature_schema.json").exists():
        raise FileNotFoundError(
            "Feature table not found. Run nasdaq-volatility-lab fetch-data first."
        )
    schema = json.loads(path.with_name("feature_schema.json").read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or not schema.get("table_sha256"):
        raise ValueError(
            "Invalid feature schema. Run nasdaq-volatility-lab fetch-data to rebuild it."
        )
    if sha256(path) != schema["table_sha256"]:
        raise ValueError("Feature table does not match its recorded hash.")
    if schema.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Feature schema must contain the ordered 12-feature whitelist.")
    if schema.get("target_column") != "target_vol_5d":
        raise ValueError("Unexpected target column in feature schema.")
    frame = pd.read_parquet(path)
    expected = ["feature_date", *FEATURE_COLUMNS, "label_end_date", "target_vol_5d"]
    if frame.columns.tolist() != expected:
        raise ValueError("Saved table columns do not match the research schema.")
    table = frame.set_index("feature_date")
    validate_feature_table(table)
    return table, schema


def new_run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Publish a complete JSON file, never partially written content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def workspace_lock(root):
    """Serialise data publication and run/log writes in one workspace."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".nasdaq-volatility-lab.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            f"Workspace is locked: {path}. Wait for the active command; remove a stale lock only after it has stopped."
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def git_provenance(root):
    def run(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        commit, status = run("rev-parse", "HEAD"), run("status", "--porcelain")
        return {"commit": commit, "dirty": status != "" if status is not None else None}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def load_run(root, run_id=None):
    """Never combine models from different runs or return incomplete runs."""
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run ID; use the ID printed by train.")
    directory = root / "results"
    paths = (
        [directory / run_id / "run.json"]
        if run_id
        else sorted(directory.glob("*/run.json"), reverse=True)
    )
    for path in paths:
        if path.parent.name.startswith(".") or not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            if run_id:
                raise ValueError(f"Cannot read run {run_id}: {exc}") from exc
            continue
        if isinstance(record, dict) and record.get("status") == "completed":
            return record
    raise FileNotFoundError(
        "No completed matching run. Run 'nasdaq-volatility-lab train' first or check --run-id."
    )
