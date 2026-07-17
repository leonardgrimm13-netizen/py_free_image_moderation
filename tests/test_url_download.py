from __future__ import annotations

import socket
import ssl
import threading
import time
from pathlib import Path

import pytest

from modimg import utils
from tests.avif_helpers import make_avif_bytes


VALID_SVG = b"""<svg xmlns="http://www.w3.org/2000/svg" width="24" height="12">
  <rect width="24" height="12" fill="red"/>
</svg>"""


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


@pytest.mark.parametrize(
    ("url", "content_type", "expected_display"),
    [
        ("https://example.test/vector.svg", "image/svg+xml", "vector.svg"),
        ("https://example.test/vector.svg", "application/octet-stream", "vector.svg"),
        ("https://example.test/extensionless", "image/svg+xml; charset=utf-8", "extensionless"),
        ("https://example.test/wrong-mime", "text/plain", "wrong-mime"),
    ],
)
def test_download_url_to_temp_accepts_valid_svg_by_content(
    monkeypatch,
    url: str,
    content_type: str,
    expected_display: str,
) -> None:
    response = FakeResponse(VALID_SVG, {"Content-Type": content_type})
    connection = _mock_response(monkeypatch, response)

    path, display = utils.download_url_to_temp(url)
    try:
        downloaded = Path(path)
        assert downloaded.suffix == ".svg"
        assert downloaded.read_bytes() == VALID_SVG
        assert display == expected_display
    finally:
        Path(path).unlink(missing_ok=True)
    assert response.closed is True
    assert connection.closed is True


def test_download_url_to_temp_rejects_html_with_svg_content_type(monkeypatch) -> None:
    html = b"<!doctype html><html><body>not an svg</body></html>"
    _mock_response(monkeypatch, FakeResponse(html, {"Content-Type": "image/svg+xml"}))

    with pytest.raises(RuntimeError, match="supported image format"):
        utils.download_url_to_temp("https://example.test/fake.svg")


def test_download_url_to_temp_uses_stricter_svg_byte_limit(monkeypatch) -> None:
    monkeypatch.setenv("MODIMG_MAX_SVG_BYTES", "64")
    _mock_response(
        monkeypatch,
        FakeResponse(
            VALID_SVG,
            {"Content-Type": "image/svg+xml", "Content-Length": str(len(VALID_SVG))},
        ),
    )

    with pytest.raises(RuntimeError, match="URL SVG too large"):
        utils.download_url_to_temp("https://example.test/vector.svg", max_bytes=10_000)


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


def test_download_url_to_temp_logs_cleanup_failure_without_masking_write_error(monkeypatch, tmp_path) -> None:
    target = tmp_path / "partial-secret.png"
    target.write_bytes(b"partial")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 8
    _mock_response(monkeypatch, FakeResponse(png))
    warnings: list[str] = []

    class BrokenTemp:
        name = str(target)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data: bytes) -> None:
            raise OSError("disk full")

    def fail_remove(path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(utils.tempfile, "NamedTemporaryFile", lambda *a, **k: BrokenTemp())
    monkeypatch.setattr(utils.os, "remove", fail_remove)
    monkeypatch.setattr(
        utils.LOGGER,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    with pytest.raises(OSError, match="disk full"):
        utils.download_url_to_temp("https://example.test/image.png")

    assert target.exists() is True
    assert warnings == ["failed to remove a temporary file: PermissionError"]
    assert str(target) not in warnings[0]


def test_successful_download_remains_owned_by_caller_when_closing_fails(
    monkeypatch,
    tmp_path,
) -> None:
    class BrokenCloseResponse(FakeResponse):
        close_attempted = False

        def close(self) -> None:
            self.close_attempted = True
            raise OSError("response close failed")

    class BrokenCloseConnection(FakeConnection):
        close_attempted = False

        def close(self) -> None:
            self.close_attempted = True
            raise OSError("connection close failed")

    response = BrokenCloseResponse(b"\x89PNG\r\n\x1a\n" + b"x" * 8)
    connection = BrokenCloseConnection()
    monkeypatch.setattr(utils, "_request_url_once", lambda *args, **kwargs: (connection, response))
    original_named_temporary_file = utils.tempfile.NamedTemporaryFile
    monkeypatch.setattr(
        utils.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: original_named_temporary_file(*args, dir=tmp_path, **kwargs),
    )
    warnings: list[str] = []
    monkeypatch.setattr(utils.LOGGER, "warning", lambda message, *args: warnings.append(message % args))

    path, display = utils.download_url_to_temp("https://example.test/image.png")
    try:
        assert Path(path).exists()
        assert display == "image.png"
    finally:
        Path(path).unlink(missing_ok=True)

    assert response.close_attempted is True
    assert connection.close_attempted is True
    assert list(tmp_path.iterdir()) == []
    assert warnings == [
        "failed to close URL response: OSError",
        "failed to close URL connection: OSError",
    ]


def test_download_validation_error_is_not_masked_by_response_close_failure(monkeypatch) -> None:
    response = FakeResponse(b"not an image", {"Content-Type": "text/plain"})
    connection = FakeConnection()

    def fail_close() -> None:
        raise OSError("response close failed")

    response.close = fail_close
    monkeypatch.setattr(utils, "_request_url_once", lambda *args, **kwargs: (connection, response))

    with pytest.raises(RuntimeError, match="URL did not return an image"):
        utils.download_url_to_temp("https://example.test/not-image")

    assert connection.closed is True


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


def test_download_url_to_temp_enforces_redirect_limit_and_closes_each_hop(monkeypatch) -> None:
    responses = [
        FakeResponse(b"", {"Location": "https://cdn.example.test/second"}, status=302),
        FakeResponse(b"", {"Location": "https://cdn.example.test/third"}, status=302),
    ]
    connections = [FakeConnection(), FakeConnection()]

    def fake_request(url: str, **kwargs):
        index = 2 - len(responses)
        return connections[index], responses.pop(0)

    original_responses = list(responses)
    monkeypatch.setattr(utils, "_request_url_once", fake_request)

    with pytest.raises(RuntimeError, match=r"exceeded redirect limit \(1\)"):
        utils.download_url_to_temp("https://example.test/start", max_redirects=1)

    assert all(response.closed for response in original_responses)
    assert all(connection.closed for connection in connections)


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


def test_request_url_once_shares_one_deadline_across_resolved_addresses(monkeypatch) -> None:
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, ("203.0.113.1", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, ("203.0.113.2", 80)),
    ]
    clock = [100.0]
    observed_timeouts: list[float] = []

    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(utils, "_resolve_url_addresses", lambda *args, **kwargs: addresses)

    class FailingConnection:
        def __init__(self, host, port, address, timeout) -> None:
            observed_timeouts.append(timeout)

        def request(self, method, path, headers) -> None:
            clock[0] += 0.6
            raise OSError("unreachable")

        def close(self) -> None:
            return None

    monkeypatch.setattr(utils, "_PinnedHTTPConnection", FailingConnection)

    with pytest.raises(TimeoutError, match="URL request timed out"):
        utils._request_url_once(
            "http://example.test/image.png",
            timeout=1.0,
            allow_private=False,
            context=ssl.create_default_context(),
        )

    assert observed_timeouts == pytest.approx([1.0, 0.4])


def test_request_url_once_bounds_dns_resolution_by_the_request_deadline(monkeypatch) -> None:
    resolver_started = threading.Event()
    release_resolver = threading.Event()
    resolver_finished = threading.Event()

    def slow_getaddrinfo(*args, **kwargs):
        resolver_started.set()
        try:
            assert release_resolver.wait(timeout=2)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]
        finally:
            resolver_finished.set()

    monkeypatch.setattr(utils.socket, "getaddrinfo", slow_getaddrinfo)
    context = ssl.create_default_context()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="host resolution timed out"):
            utils._request_url_once(
                "http://example.test/image.png",
                timeout=0.05,
                allow_private=False,
                context=context,
            )
        assert resolver_started.is_set()
        assert time.monotonic() - started < 1.0
    finally:
        release_resolver.set()
        assert resolver_finished.wait(timeout=1)


def test_url_validation_rejects_embedded_credentials() -> None:
    with pytest.raises(RuntimeError, match="embedded credentials"):
        utils._validated_url("https://user:secret@example.test/image.png")


@pytest.mark.parametrize(
    ("url", "headers", "expected_display"),
    [
        ("https://example.test/photo.avif", {"Content-Type": "image/avif"}, "photo.avif"),
        ("https://example.test/no-extension", {"Content-Type": "application/octet-stream"}, "no-extension"),
        ("https://example.test/wrong-mime", {"Content-Type": "image/jpeg"}, "wrong-mime"),
        ("https://example.test/missing-mime", {"X-Test": "present"}, "missing-mime"),
    ],
)
def test_download_url_to_temp_accepts_avif_by_content_regardless_of_mime_or_path(
    monkeypatch,
    url: str,
    headers: dict[str, str],
    expected_display: str,
) -> None:
    encoded = make_avif_bytes()
    response = FakeResponse(encoded, headers)
    connection = _mock_response(monkeypatch, response)

    path, display = utils.download_url_to_temp(url)
    try:
        downloaded = Path(path)
        assert downloaded.suffix == ".avif"
        assert downloaded.read_bytes() == encoded
        assert display == expected_display
    finally:
        Path(path).unlink(missing_ok=True)
    assert response.closed is True
    assert connection.closed is True


@pytest.mark.parametrize(
    ("url", "data", "content_type"),
    [
        ("https://example.test/fake", b"<!doctype html><html></html>", "image/avif"),
        ("https://example.test/fake.avif", b"not an image", "application/octet-stream"),
        (
            "https://example.test/truncated",
            b"\x00\x00\x00\x20ftypavif\x00\x00\x00\x00",
            "image/avif",
        ),
    ],
)
def test_download_url_to_temp_rejects_invalid_avif_hints(
    monkeypatch,
    url: str,
    data: bytes,
    content_type: str,
) -> None:
    response = FakeResponse(data, {"Content-Type": content_type})
    connection = _mock_response(monkeypatch, response)

    with pytest.raises(RuntimeError, match="valid AVIF image"):
        utils.download_url_to_temp(url)

    assert response.closed is True
    assert connection.closed is True


def test_download_url_to_temp_revalidates_redirect_and_keeps_avif_suffix(monkeypatch) -> None:
    encoded = make_avif_bytes()
    requested: list[str] = []
    responses = [
        FakeResponse(b"", {"Location": "https://cdn.example.test/final"}, status=302),
        FakeResponse(encoded, {"Content-Type": "application/octet-stream"}),
    ]

    def fake_request(url: str, **kwargs):
        requested.append(url)
        return FakeConnection(), responses.pop(0)

    monkeypatch.setattr(utils, "_request_url_once", fake_request)
    path, display = utils.download_url_to_temp("https://example.test/start.avif")
    try:
        assert Path(path).suffix == ".avif"
        assert Path(path).read_bytes() == encoded
        assert display == "final"
        assert requested == [
            "https://example.test/start.avif",
            "https://cdn.example.test/final",
        ]
    finally:
        Path(path).unlink(missing_ok=True)


def test_download_url_to_temp_cleans_partial_avif_after_streaming_limit(monkeypatch, tmp_path) -> None:
    encoded = make_avif_bytes(size=(32, 24))
    assert len(encoded) > 270
    _mock_response(monkeypatch, FakeResponse(encoded, {"Content-Type": "image/avif"}))
    original_named_temporary_file = utils.tempfile.NamedTemporaryFile

    def temporary_in_test_directory(*args, **kwargs):
        return original_named_temporary_file(*args, dir=tmp_path, **kwargs)

    monkeypatch.setattr(utils.tempfile, "NamedTemporaryFile", temporary_in_test_directory)

    with pytest.raises(RuntimeError, match="URL too large"):
        utils.download_url_to_temp("https://example.test/large.avif", max_bytes=270)

    assert list(tmp_path.iterdir()) == []


def test_download_url_to_temp_obeys_stricter_avif_source_limit(monkeypatch, tmp_path) -> None:
    encoded = make_avif_bytes(size=(32, 24))
    avif_limit = len(encoded) - 1
    monkeypatch.setenv("MODIMG_MAX_AVIF_BYTES", str(avif_limit))
    _mock_response(monkeypatch, FakeResponse(encoded, {"Content-Type": "image/avif"}))
    original_named_temporary_file = utils.tempfile.NamedTemporaryFile

    def temporary_in_test_directory(*args, **kwargs):
        return original_named_temporary_file(*args, dir=tmp_path, **kwargs)

    monkeypatch.setattr(utils.tempfile, "NamedTemporaryFile", temporary_in_test_directory)

    with pytest.raises(RuntimeError, match="URL AVIF too large"):
        utils.download_url_to_temp(
            "https://example.test/source-limit.avif",
            max_bytes=len(encoded) + 100,
        )

    assert list(tmp_path.iterdir()) == []
