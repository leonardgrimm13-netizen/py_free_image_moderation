"""Small utilities shared across the project."""
from __future__ import annotations

import io
import http.client
import ipaddress
import json
import math
import mimetypes
import os
import re
import socket
import ssl
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Tuple

from PIL import Image


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_SENSITIVE_ENV_NAMES = ("OPENAI_API_KEY", "SIGHTENGINE_SECRET", "SIGHTENGINE_USER")
_URL_IN_TEXT_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

def env_int(name: str, default: int) -> int:
    """Read an int from env, returning default on missing/invalid."""
    try:
        v = os.getenv(name)
        if v is None:
            return default
        v = str(v).strip()
        if v == "":
            return default
        if re.fullmatch(r"[+-]?\d+(?:\.0+)?", v):
            return int(float(v))
        return int(v)
    except Exception:
        return default


def env_int_any(names: tuple[str, ...], default: int) -> int:
    """Read the first defined env var in `names` as an int."""
    for n in names:
        if os.getenv(n) is not None:
            return env_int(n, default)
    return default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a float from env, returning default on missing/invalid values.

    Optional min/max bounds clamp the parsed value.
    """
    raw = os.getenv(name)
    try:
        value = float(str(raw).strip()) if raw is not None and str(raw).strip() != "" else float(default)
    except Exception:
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def env_label_set(name: str, default: str = "") -> set[str]:
    """Read a comma-separated env var as a normalized lowercase label set."""
    raw = os.getenv(name, default) or default
    return {x.strip().lower() for x in str(raw).split(",") if x.strip()}


def parse_label_float_map(raw: Any, *, min_value: float = 0.0, max_value: float = 1.0) -> dict[str, float]:
    """Parse comma-separated ``label:value`` entries into a normalized mapping.

    Invalid entries are ignored. Values are clamped to the supplied bounds so a
    typo in configuration cannot produce impossible confidence thresholds.
    """
    out: dict[str, float] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        label, value = item.split(":", 1)
        label = label.strip().lower()
        if not label:
            continue
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        out[label] = max(float(min_value), min(float(max_value), parsed))
    return out


def env_label_float_map(name: str, default: str = "", *, min_value: float = 0.0, max_value: float = 1.0) -> dict[str, float]:
    """Read a comma-separated env var of ``label:value`` confidence thresholds."""
    return parse_label_float_map(os.getenv(name, default), min_value=min_value, max_value=max_value)


def status_value(status: Any) -> str:
    """Return a lowercase engine status string for enums and legacy strings."""
    return str(status.value if hasattr(status, "value") else status).lower()


def safe_float01(v: Any, default: float = 0.0) -> float:
    """Convert to float in [0,1]. NaN/inf/invalid -> default."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return float(default)
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f
    except Exception:
        return float(default)

def is_url(s: str) -> bool:
    try:
        p = urllib.parse.urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def now_ms() -> int:
    return int(time.time() * 1000)

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()

def guess_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            return bool(_sniff_image(f.read(16))[0])
    except OSError:
        return False

def _sniff_image(data0: bytes) -> Tuple[str, str]:
    if data0.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data0.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data0[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"
    if len(data0) >= 12 and data0[:4] == b"RIFF" and data0[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if data0.startswith(b"BM"):
        return ".bmp", "image/bmp"
    if data0.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff", "image/tiff"
    return "", ""

def redact_url(url: str) -> str:
    """Return a report-safe URL without credentials, query values, or fragments."""
    try:
        parsed = urllib.parse.urlsplit(str(url))
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        query = "<redacted>" if parsed.query else ""
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))
    except (TypeError, ValueError):
        return "<redacted-url>"


def redact_sensitive_text(value: Any, *, extra_secrets: tuple[str, ...] = ()) -> str:
    """Remove configured API credentials and URL secrets from an error message."""
    text = str(value)
    secrets = [*(os.getenv(name) or "" for name in _SENSITIVE_ENV_NAMES), *extra_secrets]
    for secret in secrets:
        if len(secret) >= 4:
            text = text.replace(secret, "<redacted>")
    return _URL_IN_TEXT_RE.sub(lambda match: redact_url(match.group(0)), text)


def _remove_file_if_exists(path: str | os.PathLike[str]) -> None:
    """Remove a cleanup target while tolerating an already-removed file."""
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace a text file using a temporary file in the same directory."""
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            fd = -1
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            mode = target.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = None
        if mode is not None:
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    finally:
        if fd >= 0:
            os.close(fd)
        _remove_file_if_exists(tmp_name)


def _validated_url(url: str) -> tuple[urllib.parse.SplitResult, str, int]:
    if any(ord(char) < 32 for char in str(url)):
        raise RuntimeError("URL contains control characters")
    try:
        parsed = urllib.parse.urlsplit(str(url))
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("URL contains an invalid port") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise RuntimeError("URL scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("URLs containing embedded credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise RuntimeError("URL host is missing")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise RuntimeError("URL host is invalid") from exc
    resolved_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    if not 1 <= resolved_port <= 65535:
        raise RuntimeError("URL port is outside the valid range")
    return parsed, ascii_host, resolved_port


def _resolve_url_addresses(host: str, port: int, *, allow_private: bool) -> list[tuple[int, int, int, tuple[Any, ...]]]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RuntimeError(f"URL host could not be resolved: {host}") from exc

    addresses: list[tuple[int, int, int, tuple[Any, ...]]] = []
    seen: set[tuple[int, tuple[Any, ...]]] = set()
    for family, socktype, proto, _canonname, sockaddr in infos:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        raw_ip = str(sockaddr[0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if ip.is_multicast or ip.is_unspecified:
            raise RuntimeError(f"URL target address is not usable ({ip.compressed})")
        if not allow_private and not ip.is_global:
            raise RuntimeError(f"URL target is not publicly routable ({ip.compressed})")
        key = (family, tuple(sockaddr))
        if key not in seen:
            seen.add(key)
            addresses.append((family, socktype, proto, tuple(sockaddr)))
    if not addresses:
        raise RuntimeError("URL host has no usable IPv4 or IPv6 address")
    return addresses


def _connect_resolved(address: tuple[int, int, int, tuple[Any, ...]], timeout: float) -> socket.socket:
    family, socktype, proto, sockaddr = address
    sock = socket.socket(family, socktype, proto)
    try:
        sock.settimeout(timeout)
        sock.connect(sockaddr)
        return sock
    except BaseException:
        sock.close()
        raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, address: tuple[int, int, int, tuple[Any, ...]], timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._resolved_address = address

    def connect(self) -> None:
        self.sock = _connect_resolved(self._resolved_address, float(self.timeout))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        address: tuple[int, int, int, tuple[Any, ...]],
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._resolved_address = address

    def connect(self) -> None:
        raw_sock = _connect_resolved(self._resolved_address, float(self.timeout))
        try:
            self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)
        except BaseException:
            raw_sock.close()
            raise


def _request_url_once(
    url: str,
    *,
    timeout: float,
    allow_private: bool,
    context: ssl.SSLContext,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    parsed, host, port = _validated_url(url)
    addresses = _resolve_url_addresses(host, port, allow_private=allow_private)
    path = urllib.parse.quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    if parsed.query:
        path = f"{path}?{urllib.parse.quote(parsed.query, safe='=&;%:+,/?@!$()*-._~')}"
    headers = {"User-Agent": "image-moderator/1.0", "Accept": "image/*,*/*;q=0.8", "Connection": "close"}
    last_error: OSError | ssl.SSLError | http.client.HTTPException | None = None
    for address in addresses:
        connection: http.client.HTTPConnection
        if parsed.scheme.lower() == "https":
            connection = _PinnedHTTPSConnection(host, port, address, timeout, context)
        else:
            connection = _PinnedHTTPConnection(host, port, address, timeout)
        try:
            connection.request("GET", path, headers=headers)
            return connection, connection.getresponse()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
            connection.close()
    raise RuntimeError(f"Could not connect to URL host: {host}") from last_error


def _response_header(response: http.client.HTTPResponse, name: str) -> str:
    value = response.getheader(name)
    return str(value or "")


def download_url_to_temp(
    url: str,
    max_bytes: int = 25_000_000,
    timeout_sec: int | float = 20,
    *,
    max_redirects: int = 5,
    allow_private: bool | None = None,
) -> tuple[str, str]:
    """Safely stream an image URL to a temporary file.

    DNS results are validated and pinned to the socket connection, and every
    redirect is validated independently to prevent SSRF through redirect or DNS
    rebinding tricks.
    """
    byte_limit = max(1, int(max_bytes))
    timeout = max(0.1, float(timeout_sec))
    redirect_limit = max(0, int(max_redirects))
    private_allowed = env_bool("MODIMG_ALLOW_PRIVATE_URLS", False) if allow_private is None else bool(allow_private)
    context = ssl.create_default_context()
    deadline = time.monotonic() + timeout
    current_url = str(url)

    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    for redirect_count in range(redirect_limit + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"URL download timed out after {timeout:g}s")
        connection, response = _request_url_once(
            current_url,
            timeout=remaining,
            allow_private=private_allowed,
            context=context,
        )
        if response.status in {301, 302, 303, 307, 308}:
            redirect_status = response.status
            location = _response_header(response, "Location")
            response.close()
            connection.close()
            response = None
            connection = None
            if not location:
                raise RuntimeError(f"URL redirect returned HTTP {redirect_status} without Location")
            if redirect_count >= redirect_limit:
                raise RuntimeError(f"URL exceeded redirect limit ({redirect_limit})")
            current_url = urllib.parse.urljoin(current_url, location)
            continue
        break

    if connection is None or response is None:
        raise RuntimeError("URL request did not produce a response")

    tmp_path = ""
    try:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"URL request failed with HTTP status {response.status}")
        content_length = _response_header(response, "Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = -1
            if declared_size > byte_limit:
                raise RuntimeError(f"URL too large: {declared_size} bytes (limit {byte_limit})")

        content_type = _response_header(response, "Content-Type").split(";", 1)[0].strip().lower()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"URL download timed out after {timeout:g}s")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        first_chunk = response.read(min(65_536, byte_limit + 1))
        if len(first_chunk) > byte_limit:
            raise RuntimeError(f"URL too large: downloaded > {byte_limit} bytes")
        sniff_ext, _sniff_mime = _sniff_image(first_chunk)
        if not sniff_ext:
            if content_type and not content_type.startswith("image/"):
                raise RuntimeError(f"URL did not return an image (content-type={content_type})")
            raise RuntimeError("URL does not contain a supported image format (jpeg/png/webp/gif/bmp/tiff)")

        total = len(first_chunk)
        with tempfile.NamedTemporaryFile(delete=False, suffix=sniff_ext) as tmp:
            tmp_path = tmp.name
            tmp.write(first_chunk)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"URL download timed out after {timeout:g}s")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read(min(65_536, byte_limit - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > byte_limit:
                    raise RuntimeError(f"URL too large: downloaded > {byte_limit} bytes")
                tmp.write(chunk)

        display = os.path.basename(urllib.parse.urlsplit(current_url).path) or ("downloaded" + sniff_ext)
        return tmp_path, display
    except BaseException:
        if tmp_path:
            _remove_file_if_exists(tmp_path)
        raise
    finally:
        response.close()
        connection.close()

def safe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return str(obj)


_JSON_CONVERSION_FAILED = object()


def _best_effort_json_conversion(value: Any, method_name: str) -> Any:
    """Call optional NumPy-like converters without trusting third-party objects."""
    try:
        converter = getattr(value, method_name)
        return converter()
    except Exception:
        return _JSON_CONVERSION_FAILED


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of common runtime values.

    Engine details can contain enums, dataclasses, pathlib paths, numpy scalar
    values/arrays, or bytes from optional dependencies. Keep the report useful
    without allowing one non-serializable detail to crash the CLI.
    """
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "value"):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        converted = _best_effort_json_conversion(value, "item")
        if converted is not _JSON_CONVERSION_FAILED:
            return json_safe(converted)
    if hasattr(value, "tolist"):
        converted = _best_effort_json_conversion(value, "tolist")
        if converted is not _JSON_CONVERSION_FAILED:
            return json_safe(converted)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def json_dumps_safe(value: Any, **kwargs: Any) -> str:
    """Dump JSON after normalizing values that stdlib json cannot encode."""
    return json.dumps(json_safe(value), **kwargs)
