# tests/conftest.py
import sys
from pathlib import Path
import pytest

# Project root = folder that contains "src" and "tests"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))