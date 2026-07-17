from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from PIL import Image


def test_offline_no_apis_with_generated_image(tmp_path: Path) -> None:
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (24, 24), color=(120, 80, 200)).save(img_path)
    env = os.environ.copy()
    env.update(
        {
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "OCR_ENABLE": "0",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_BACKEND": "disabled",
            "YOLO_WEAPON_MODEL": str(tmp_path / "missing-weapons.pt"),
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
        }
    )

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis"],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0
    assert "Traceback (most recent call last)" not in combined
    assert "FINAL:" in combined
    assert "[" in combined and "]" in combined
