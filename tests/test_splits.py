"""Synthetic boundary tests; no network or market data required."""

import unittest

import pandas as pd

from nasdaq_volatility_lab.splits import chronological_split


class ChronologicalSplitTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        self.table = pd.DataFrame(
            {
                "target_vol_5d": [0.2] * 25 + [float("nan")] * 5,
                "label_end_date": pd.Series(dates).shift(-5).to_numpy(),
            },
            index=dates,
        )

    def split(self, table=None, **kwargs):
        return chronological_split(
            self.table if table is None else table,
            self.table.index[15],
            self.table.index[-1] + pd.Timedelta(days=1),
            **kwargs,
        )

    def test_exact_gap_and_strict_label_boundary(self):
        split = self.split()
        self.assertTrue(split.train.index.equals(self.table.index[:10]))
        self.assertTrue(split.gap.index.equals(self.table.index[10:15]))
        self.assertEqual(split.train.label_end_date.max(), self.table.index[14])
        self.assertEqual(self.table.iloc[10].label_end_date, split.evaluation.index[0])
        self.assertNotIn(self.table.index[10], split.train.index)

    def test_missing_labels_excluded_without_mutating_input(self):
        original = self.table.copy(deep=True)
        split = self.split()
        self.assertTrue(split.excluded_evaluation.index.equals(self.table.index[-5:]))
        pd.testing.assert_frame_equal(original, self.table)

    def test_label_cutoff_excludes_equal_and_later_outcomes(self):
        cutoff = self.table.index[23]
        split = self.split(label_cutoff=cutoff)
        self.assertTrue(split.evaluation.index.equals(self.table.index[15:18]))
        self.assertTrue((split.evaluation.label_end_date < cutoff).all())

    def test_rejects_leaking_training_label_even_with_gap(self):
        invalid = self.table.copy()
        invalid.loc[invalid.index[9], "label_end_date"] = invalid.index[15]
        with self.assertRaisesRegex(ValueError, "reach the evaluation"):
            self.split(invalid)

    def test_rejects_zero_gap_unsorted_and_duplicate_dates(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            self.split(gap_sessions=0)
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            self.split(self.table.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            self.split(pd.concat([self.table, self.table.iloc[-1:]]))

    def test_nested_split_and_insufficient_history(self):
        outer = chronological_split(self.table, self.table.index[23], "2021-01-01")
        inner = chronological_split(outer.train, self.table.index[10], self.table.index[23])
        self.assertTrue(inner.train.index.isin(outer.train.index).all())
        self.assertTrue(inner.evaluation.index.isin(outer.train.index).all())
        self.assertLess(inner.train.label_end_date.max(), inner.evaluation.index.min())
        with self.assertRaisesRegex(ValueError, "Insufficient training"):
            chronological_split(outer.train, outer.train.index[4], self.table.index[23])


if __name__ == "__main__":
    unittest.main()
