import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_python_files():
    for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")]:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        if path.name == "check_project.py":
            continue
        if ":\\" in source:
            raise AssertionError(f"Absolute Windows path found in {path}")


def check_notebooks():
    for path in ROOT.glob("notebooks/*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if ":\\" in source:
                raise AssertionError(f"Absolute Windows path found in {path}")


def check_dataset():
    path = ROOT / "data" / "Hatchability.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 1737 or len(reader.fieldnames or []) != 125:
        raise AssertionError("Unexpected dataset dimensions")
    if "Hatchability" not in (reader.fieldnames or []):
        raise AssertionError("Target column is missing")
    low_count = sum(float(row["Hatchability"]) < 60 for row in rows)
    if low_count != 180:
        raise AssertionError(f"Expected 180 low-hatchability rows, found {low_count}")


if __name__ == "__main__":
    check_python_files()
    check_notebooks()
    check_dataset()
    print("Project checks passed")
