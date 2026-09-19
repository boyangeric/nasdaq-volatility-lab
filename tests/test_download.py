"""Exercise snapshot creation and failure handling without network access."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from nasdaq_volatility_lab import download_data


class DownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.frame = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2005-01-03"]))
        root_patch = patch.object(download_data, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_download_writes_both_tickers_and_refuses_overwrite(self):
        with patch.object(download_data.yf, "download", return_value=self.frame) as download:
            download_data.main()
            self.assertEqual(download.call_count, 2)
            self.assertTrue(download.call_args.kwargs["auto_adjust"])
        snapshot = self.root / "data/snapshot"
        self.assertEqual({p.name for p in snapshot.iterdir()}, {"QQQ.parquet", "SPY.parquet", "metadata.json"})
        with patch.object(download_data.yf, "download") as download:
            with self.assertRaises(FileExistsError):
                download_data.main()
            download.assert_not_called()

    def test_failed_second_download_leaves_no_snapshot(self):
        with patch.object(download_data.yf, "download", side_effect=[self.frame, pd.DataFrame()]):
            with self.assertRaisesRegex(RuntimeError, "SPY"):
                download_data.main()
        self.assertFalse((self.root / "data/snapshot").exists())
