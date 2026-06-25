from __future__ import annotations

from pathlib import Path

import pytest

from modimg import utils


class FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self._data = data
        self.headers = headers or {"Content-Type": "image/png"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int) -> bytes:
        return self._data


def test_download_url_to_temp_mocked_success(monkeypatch) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    monkeypatch.setattr(utils.urllib.request, "urlopen", lambda *a, **k: FakeResponse(png))

    path, display = utils.download_url_to_temp("https://example.test/a%20b.png")
    try:
        assert Path(path).exists()
        assert Path(path).suffix == ".png"
        assert display == "a%20b.png"
    finally:
        Path(path).unlink(missing_ok=True)


def test_download_url_to_temp_rejects_non_image(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda *a, **k: FakeResponse(b"hello", {"Content-Type": "text/plain"}),
    )

    with pytest.raises(RuntimeError, match="URL did not return an image"):
        utils.download_url_to_temp("https://example.test/not-image.txt")


def test_download_url_to_temp_cleans_partial_file_on_write_error(monkeypatch, tmp_path) -> None:
    target = tmp_path / "partial.png"
    target.write_bytes(b"partial")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    monkeypatch.setattr(utils.urllib.request, "urlopen", lambda *a, **k: FakeResponse(png))

    class BrokenTemp:
        name = str(target)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data: bytes) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(utils.tempfile, "NamedTemporaryFile", lambda *a, **k: BrokenTemp())

    with pytest.raises(OSError, match="disk full"):
        utils.download_url_to_temp("https://example.test/image.png")

    assert target.exists() is False
