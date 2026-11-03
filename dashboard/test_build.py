"""Checks for publication identity and numerical-data integrity."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from build import HERE, build, catalog, read_json, validate_report


class PublishedReports(unittest.TestCase):
    def test_root_and_archived_report_use_identical_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "site"
            build(output)
            index = catalog()
            for item in index["snapshots"]:
                archive = output / "versions" / item["id"]
                self.assertEqual((archive / "results.json").read_bytes(),
                                 (output / "reports" / item["file"]).read_bytes())
                manifest = read_json(archive / "manifest.json")
                for report in manifest["snapshots"]:
                    self.assertTrue((archive / report["path"]).is_file())
            latest = next(i for i in index["snapshots"] if i["id"] == index["latest"])
            self.assertEqual((output / "results.json").read_bytes(),
                             (output / "reports" / latest["file"]).read_bytes())

    def test_modified_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            shutil.copytree(HERE / "reports", source / "reports")
            index = read_json(source / "reports/index.json")
            path = source / "reports" / index["snapshots"][0]["file"]
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                catalog(source)

    def test_missing_bin_cannot_be_published(self):
        index = catalog()
        data = read_json(HERE / "reports" / index["snapshots"][0]["file"])
        damaged = copy.deepcopy(data)
        damaged["examples"][0]["traces"][0]["bins"].pop()
        with self.assertRaisesRegex(ValueError, "edges"):
            validate_report(damaged)

    def test_registry_entry_cannot_disappear_silently(self):
        index = catalog()
        data = read_json(HERE / "reports" / index["snapshots"][0]["file"])
        damaged = copy.deepcopy(data)
        damaged["ideas"].pop()
        with self.assertRaisesRegex(ValueError, "registry"):
            validate_report(damaged)


if __name__ == "__main__":
    unittest.main()
