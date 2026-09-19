from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Hatchability.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_COLUMN = "Hatchability"
LOW_HATCHABILITY_THRESHOLD = 60
N_SYNTHETIC_SAMPLES = 30

# Final evaluation: five outer folds, each approximately 80% train / 20% test.
OUTER_N_SPLITS = 5
OUTER_SHUFFLE = True
OUTER_RANDOM_STATE = 42

# HSL construction: out-of-fold stage-0 predictions for the stage-1 learner.
STACKING_INTERNAL_CV = 5

