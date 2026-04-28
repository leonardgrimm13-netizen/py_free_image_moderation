from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image


def test_offline_no_apis_with_generated_image(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), color=(120, 80, 200)).save(img_path)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis"],
        check=False,
        capture_output=True,
        text=True,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode in (0, 2)
    assert "Traceback (most recent call last)" not in combined
    assert "FINAL:" in combined
    assert "[" in combined and "]" in combined
