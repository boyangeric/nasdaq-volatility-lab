"""Read the shared feature artifact under one schema and integrity contract."""

from pathlib import Path
import hashlib
import json

import pandas as pd

from .build_table import FEATURE_COLUMNS, validate_feature_table


def load_feature_table(path: Path) -> tuple[pd.DataFrame, dict]:
    """Reject stale bytes and unexpected model inputs before any split or fit.

    A file hash alone cannot stop a modified schema from including the target
    among model inputs. Check the explicit feature whitelist as well.
    """
    path = Path(path)
    schema = json.loads(path.with_name("feature_schema.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != schema["table_sha256"]:
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
