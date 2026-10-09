import csv
import hashlib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "MANIFEST.csv"


class ManifestHashTests(unittest.TestCase):
    def test_committed_artifact_hashes_match_manifest(self):
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))

        self.assertTrue(records, "manifest must contain artifact records")
        for record in records:
            expected = record["sha256"].strip()
            status = record["status"].strip()
            if not expected or status.startswith("retired_") or status.startswith("not_in_repository"):
                continue

            artifact = REPO_ROOT / record["file"]
            self.assertTrue(artifact.is_file(), f"missing manifest artifact: {record['file']}")
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"hash mismatch: {record['file']}")


if __name__ == "__main__":
    unittest.main()
