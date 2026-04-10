"""Run the test suite in a way that makes the `src` package importable when
invoking `python run_tests.py` from the project root.

Usage:
    python run_tests.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
# Ensure project root is on sys.path so `import src` works when running directly
sys.path.insert(0, str(ROOT))

import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
