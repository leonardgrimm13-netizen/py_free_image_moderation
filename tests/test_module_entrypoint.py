from __future__ import annotations

import subprocess
import sys


def test_python_m_modimg_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "modimg", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    help_text = f"{proc.stdout}\n{proc.stderr}"
    assert "--no-apis" in help_text
