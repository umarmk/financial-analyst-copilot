from pathlib import Path

# Project root is the parent of the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw and processed data directories
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

