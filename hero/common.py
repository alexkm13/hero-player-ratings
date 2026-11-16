"""Paths and identifiers shared by the standalone model commands."""
import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HERO_DATA_ROOT", ROOT / "data")).expanduser()
ARTIFACTS = Path(os.environ.get("HERO_ARTIFACTS_ROOT", ROOT / "artifacts")).expanduser()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def team(value):
    name = str(value).replace(".", "").upper()
    return {"ARI": "UTA", "UTAH": "UTA", "LA": "LAK", "NJ": "NJD",
            "SJ": "SJS", "TB": "TBL"}.get(name, name)
