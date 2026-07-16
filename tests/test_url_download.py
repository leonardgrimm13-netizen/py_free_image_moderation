from __future__ import annotations

import socket
from pathlib import Path

import pytest

from modimg import utils


class FakeResponse:
    def __init__(
        self,
        data: bytes,
        headers: dict[str, str] | None = None,
        *,
        status: int = 200,
    ) -> None:
        self._data = data
        self._offset = 0
        self.headers = headers or {"Content-Type": "image/png"}
        self.status = status
        self.closed = False

    def getheader(self, name: str):
        return self.headers.get(name)

    def close(self) -> None:
        self.closed = True

    def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class FakeConnection:
    sock = None

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _mock_response(monkeypatch, response: FakeResponse) -> FakeConnection:
    connection = FakeConnection()
    monkeypatch.setattr(
        utils,
        "_request_url_once",
        lambda *args, **kwargs: (connection, response),
    )
    return connection


def test_download_url_to_temp_mocked_success(monkeypatch) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    response = FakeResponse(png)
    connection = _mock_response(monkeypatch, response)

    path, display = utils.download_url_to_temp("https://example.test/a%20b.png")
    try:
        assert Path(path).exists()
        assert Path(path).suffix == ".png"
        assert display == "a%20b.png"
    finally:
        Path(path).unlink(missing_ok=True)
    assert response.closed is True
    assert connection.closed is True


def test_download_url_to_temp_rejects_non_image(monkeypatch) -> None:
    _mock_response(monkeypatch, FakeResponse(b"hello", {"Content-Type": "text/plain"}))

    with pytest.raises(RuntimeError, match="URL did not return an image"):
        utils.download_url_to_temp("https://example.test/not-image.txt")


def test_download_url_to_temp_cleans_partial_file_on_write_error(monkeypatch, tmp_path) -> None:
    target = tmp_path / "partial.png"
    target.write_bytes(b"partial")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    _mock_response(monkeypatch, FakeResponse(png))

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


def test_download_url_to_temp_enforces_streaming_byte_limit(monkeypatch) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 20
    _mock_response(monkeypatch, FakeResponse(png))

    with pytest.raises(RuntimeError, match="URL too large"):
        utils.download_url_to_temp("https://example.test/image.png", max_bytes=12)


def test_download_url_to_temp_revalidates_redirect_targets(monkeypatch) -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    requested: list[str] = []
    responses = [
        FakeResponse(b"", {"Location": "https://cdn.example.test/final.png"}, status=302),
        FakeResponse(png),
    ]

    def fake_request(url: str, **kwargs):
        requested.append(url)
        return FakeConnection(), responses.pop(0)

    monkeypatch.setattr(utils, "_request_url_once", fake_request)
    path, _display = utils.download_url_to_temp("https://example.test/start.png")
    try:
        assert requested == [
            "https://example.test/start.png",
            "https://cdn.example.test/final.png",
        ]
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"])
def test_resolve_url_addresses_rejects_non_public_targets(monkeypatch, address: str) -> None:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
    monkeypatch.setattr(
        utils.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(family, socket.SOCK_STREAM, 6, "", sockaddr)],
    )

    with pytest.raises(RuntimeError, match="not publicly routable"):
        utils._resolve_url_addresses("example.test", 443, allow_private=False)


def test_resolve_url_addresses_private_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )

    addresses = utils._resolve_url_addresses("localhost", 80, allow_private=True)

    assert addresses[0][3] == ("127.0.0.1", 80)


def test_url_validation_rejects_embedded_credentials() -> None:
    with pytest.raises(RuntimeError, match="embedded credentials"):
        utils._validated_url("https://user:secret@example.test/image.png")
