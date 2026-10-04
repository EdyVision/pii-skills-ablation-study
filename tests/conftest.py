"""Pytest conftest: add lib to path so ablation_harness is importable."""

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
lib_path = repo_root / "lib"
if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))
