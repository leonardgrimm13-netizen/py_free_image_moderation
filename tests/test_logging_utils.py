from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("level", ["BASIC_FORMAT", "NOT_A_LEVEL"])
def test_invalid_log_level_does_not_prevent_cli_startup(level: str) -> None:
    env = os.environ.copy()
    env["MODIMG_LOG_LEVEL"] = level
    result = subprocess.run(
        [sys.executable, "-m", "modimg.cli", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--no-apis" in result.stdout
