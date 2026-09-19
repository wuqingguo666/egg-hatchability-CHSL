import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Hatchability.csv"


class DatasetContractTest(unittest.TestCase):
    def test_cleaned_dataset_contract(self):
        with DATA_PATH.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)

        self.assertEqual(len(rows), 1737)
        self.assertEqual(len(reader.fieldnames or []), 125)
        self.assertIn("Hatchability", reader.fieldnames or [])
        self.assertEqual(
            sum(float(row["Hatchability"]) < 60 for row in rows),
            180,
        )


if __name__ == "__main__":
    unittest.main()

