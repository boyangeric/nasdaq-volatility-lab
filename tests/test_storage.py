"""Reject corrupted artifacts and schemas that could introduce target leakage."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nasdaq_volatility_lab.features import (
    FEATURE_COLUMNS,
    build_feature_table,
    validate_feature_table,
)
from nasdaq_volatility_lab.storage import load_feature_table


class StorageTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range(end="2005-01-31", periods=100)
        prices = pd.DataFrame(
            {
                "Close": 100 * np.exp(np.arange(100) * 0.001),
                "Volume": 1000.0,
            },
            index=dates,
        )
        self.table = build_feature_table(prices, prices.copy())
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "feature_table.parquet"
        self.table.reset_index().to_parquet(self.path, index=False)
        self.schema = {
            "feature_columns": FEATURE_COLUMNS.copy(),
            "target_column": "target_vol_5d",
            "table_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }
        self.write_schema()

    def write_schema(self):
        self.path.with_name("feature_schema.json").write_text(json.dumps(self.schema))

    def test_round_trip_and_corruption(self):
        actual, _ = load_feature_table(self.path)
        # Parquet retains dates, not pandas' optional inferred frequency metadata.
        pd.testing.assert_frame_equal(actual, self.table, check_freq=False)
        with self.path.open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_feature_table(self.path)

    def test_rejects_target_in_inputs_and_wrong_target(self):
        self.schema["feature_columns"] = [*FEATURE_COLUMNS[:-1], "target_vol_5d"]
        self.write_schema()
        with self.assertRaisesRegex(ValueError, "whitelist"):
            load_feature_table(self.path)
        self.schema["feature_columns"] = FEATURE_COLUMNS.copy()
        self.schema["target_column"] = "qqq_vol_20d"
        self.write_schema()
        with self.assertRaisesRegex(ValueError, "target"):
            load_feature_table(self.path)

    def test_rejects_wrong_label_horizon_and_negative_target(self):
        invalid = self.table.copy()
        invalid.loc[invalid.index[0], "label_end_date"] = invalid.index[4]
        with self.assertRaisesRegex(ValueError, "fifth"):
            validate_feature_table(invalid)
        invalid = self.table.copy()
        invalid.loc[invalid.index[0], "target_vol_5d"] = -0.1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_feature_table(invalid)
