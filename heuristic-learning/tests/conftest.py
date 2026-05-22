"""Put `src/` on sys.path so tests can `import hl_core` / `import hl_lander`.

Mirrors the sys.path bootstrap that experiments/hl-lunar-lander/run.py uses, so
tests and the experiment runner resolve the same package layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
