# utils/file_handling.py  – 2025-05-15 portable rewrite
import json
import os
from pathlib import Path

from platformdirs import PlatformDirs, user_documents_path

APP_NAME = "VirtualElevationRecipes"
APP_AUTHOR = "hannes"
DIRS = PlatformDirs(APP_NAME, APP_AUTHOR)


# ────────────────────────────────────────────────────────────────────────
# Public helpers
# ────────────────────────────────────────────────────────────────────────
def get_config_dir() -> Path:
    p = Path(DIRS.user_config_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_results_dir() -> Path:
    try:
        docs_root = Path(user_documents_path())
    except AttributeError:
        docs_root = Path.home() / "Documents"
    p = docs_root / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p


# ────────────────────────────────────────────────────────────────────────
# Generic helpers
# ────────────────────────────────────────────────────────────────────────
def ensure_dir(directory):
    return Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None
