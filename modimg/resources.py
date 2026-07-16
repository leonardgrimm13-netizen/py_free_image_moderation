"""Resolve bundled files in source checkouts and installed distributions."""
from __future__ import annotations

import site
import sys
import sysconfig
from pathlib import Path


def source_root() -> Path:
    """Return the checkout root, or the site-packages directory when installed."""
    return Path(__file__).resolve().parent.parent


def _installation_roots() -> list[Path]:
    roots = [Path(sys.prefix)]
    for value_getter in (
        lambda: sysconfig.get_path("data"),
        site.getuserbase,
    ):
        try:
            value = value_getter()
        except (AttributeError, OSError, TypeError):
            value = None
        if value:
            roots.append(Path(value))
    return roots


def _candidate_paths(candidate: Path, roots: list[Path]) -> list[Path]:
    if candidate.is_absolute():
        return [candidate]

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root / candidate
        key = str(resolved.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def _safe_cwd() -> Path | None:
    """Return the current directory when it still exists and is accessible."""
    try:
        return Path.cwd()
    except OSError:
        return None


def resource_candidates(path: str | Path) -> list[Path]:
    """Return candidates for an explicitly configured relative resource."""
    candidate = Path(path).expanduser()
    roots: list[Path] = []
    cwd = _safe_cwd()
    if cwd is not None:
        roots.append(cwd)
    roots.append(source_root())
    roots.extend(_installation_roots())
    return _candidate_paths(candidate, roots)


def bundled_resource_candidates(path: str | Path) -> list[Path]:
    """Return candidates for a bundled default without allowing cwd shadowing."""
    candidate = Path(path).expanduser()
    roots: list[Path] = [source_root(), *_installation_roots()]
    return _candidate_paths(candidate, roots)


def resolve_resource_path(path: str | Path) -> Path:
    """Resolve the first existing candidate, or the source-root candidate."""
    candidates = resource_candidates(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def resolve_bundled_resource_path(path: str | Path) -> Path:
    """Resolve a packaged default, preferring trusted installation roots."""
    candidates = bundled_resource_candidates(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)
