"""Local layout for optional archived experiment inputs and generated results."""
import os
from pathlib import Path

from hero.common import ROOT, DATA, ARTIFACTS

ARCHIVE = Path(os.environ.get("HERO_ARCHIVE_ROOT", DATA / "research_archive")).expanduser()
PCV = Path(os.environ.get("HERO_PCV_ROOT", ROOT / "pcv")).expanduser()


def output(name):
    path = ARTIFACTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
