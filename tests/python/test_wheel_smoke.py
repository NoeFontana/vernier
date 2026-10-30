"""Run the wheel smoke script against the development build.

CI runs ``tools/dev/wheel_smoke.py`` against every built wheel it can
execute; running it here keeps the script working between releases and
puts its fork-while-evaluating check (ADR-0070) on every pull request.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "dev" / "wheel_smoke.py"


def test_wheel_smoke_script() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "smoke OK" in result.stdout
