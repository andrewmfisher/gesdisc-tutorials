"""Compatibility launcher for the canonical benchmark script.

The maintained implementation is ``../benchmark_access_methods.py``. Keeping
this thin launcher preserves the old command path without duplicating logic.
"""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    benchmark_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(benchmark_dir))
    benchmark_script = benchmark_dir / "benchmark_access_methods.py"
    runpy.run_path(str(benchmark_script), run_name="__main__")
