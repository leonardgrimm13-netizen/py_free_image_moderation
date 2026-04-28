from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .pipeline import run_on_input
from .utils import is_image_file, is_url
from .config import load_dotenv_candidates


def _env_int(name: str, default: int) -> int:
    """Parse integer env vars defensively to avoid CLI crashes on bad values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default

def _select_scores(engine_name: str, scores: Dict[str, Any]) -> List[tuple[str, float]]:
    """
    Reduce noisy score output.

    Controlled via .env:
      - SCORE_VERBOSE=1                -> print all keys for all engines
      - SCORE_MAX_KEYS=8               -> max keys for non-Sightengine engines
      - SIGHTENGINE_SCORE_MODE=compact -> compact|full|keys
      - SIGHTENGINE_SCORE_KEYS=...     -> comma-separated keys when mode=keys
      - SIGHTENGINE_EXTRA_TOPK=0       -> add strongest remaining keys (compact mode)
    """
    # Global override: show everything
    if os.getenv("SCORE_VERBOSE", "0").strip() == "1":
        items: List[tuple[str, float]] = []
        for k, v in scores.items():
            try:
                items.append((k, float(v)))
            except Exception:
                continue
        return items

    name = (engine_name or "").lower()

    # Special handling for Sightengine (very verbose by default)
    if "sightengine" in name:
        mode = (os.getenv("SIGHTENGINE_SCORE_MODE", "compact") or "compact").strip().lower()

        if mode in ("full", "all", "verbose"):
            items: List[tuple[str, float]] = []
            for k, v in scores.items():
                try:
                    items.append((k, float(v)))
                except Exception:
                    continue
            return items

        if mode == "keys":
            keys_raw = os.getenv("SIGHTENGINE_SCORE_KEYS", "") or ""
            wanted = [k.strip() for k in keys_raw.split(",") if k.strip()]
            items: List[tuple[str, float]] = []
            for k in wanted:
                if k in scores:
                    try:
                        items.append((k, float(scores[k])))
                    except Exception:
                        pass
            return items

        # compact (default)
        preferred = [
            "nudity_safe", "nudity_raw", "nudity_partial",
            "weapon_firearm", "weapon_firearm_toy", "weapon_knife",
            "gore_prob", "violence_prob", "offensive_max",
        ]
        items: List[tuple[str, float]] = []
        for k in preferred:
            if k in scores:
                try:
                    items.append((k, float(scores[k])))
                except Exception:
                    pass

        # optionally include strongest remaining signals above a tiny threshold
        extra_topk = _env_int("SIGHTENGINE_EXTRA_TOPK", 0)
        if extra_topk > 0:
            rest: List[tuple[str, float]] = []
            for k, v in scores.items():
                if k in preferred:
                    continue
                try:
                    rest.append((k, float(v)))
                except Exception:
                    continue
            rest.sort(key=lambda kv: kv[1], reverse=True)
            for k, v in rest[:extra_topk]:
                if v >= 0.05:  # avoid spam from zeros
                    items.append((k, v))

        return items

    # Other engines: keep up to SCORE_MAX_KEYS (default 8), highest first.
    max_keys = _env_int("SCORE_MAX_KEYS", 8)
    rest: List[tuple[str, float]] = []
    for k, v in scores.items():
        try:
            rest.append((k, float(v)))
        except Exception:
            continue
    rest.sort(key=lambda kv: kv[1], reverse=True)
    return rest[:max_keys]


def _iter_paths(p: str, recursive: bool) -> List[str]:
    if is_url(p):
        return [p]
    if os.path.isdir(p):
        out = []
        if recursive:
            for root, _, files in os.walk(p):
                for f in files:
                    fp = os.path.join(root, f)
                    if is_image_file(fp):
                        out.append(fp)
        else:
            for f in os.listdir(p):
                fp = os.path.join(p, f)
                if os.path.isfile(fp) and is_image_file(fp):
                    out.append(fp)
        return sorted(out)
    return [p]

def _print_report(rep: Dict[str, Any]) -> None:
    v = rep["verdict"]
    results = rep["results"]
    name = rep["name"]

    print("="*70)
    print(name)
    print(f"FINAL: {'OK' if v.label=='OK' else 'NOT_OK'}  (verdict={v.label}) | nudity={v.nudity_risk:.2f} violence={v.violence_risk:.2f} hate={v.hate_risk:.2f}")
    for r in v.reasons:
        print(f" - {r}")
    if rep.get("auto_learn"):
        print(f" - {rep['auto_learn']}")

    for r in results:
        st = (r.status or "").lower()
        tag = {"ok":"ok", "skipped":"skipped", "error":"error"}.get(st, st)
        msg = ""
        if st == "ok" and r.scores:
            parts = []
            for k, vv in _select_scores(r.name, r.scores):
                parts.append(f"{k}={float(vv):.2f}")
            if parts:
                msg = ", ".join(parts)
        elif r.error:
            msg = r.error
        took = f"{int(r.took_ms or 0)}ms"
        print(f"   [{tag:<7}] {r.name:<22} ({took}) {msg}")

def main(argv: List[str] | None = None) -> int:
    # Ensure .env was loaded (already loaded on import), but keep debug message consistent
    load_dotenv_candidates()

    ap = argparse.ArgumentParser(description="Moderate an image/GIF or folder with multiple optional engines.")
    ap.add_argument("input", nargs="?", default="", help="Path/dir/URL to moderate")
    ap.add_argument("--no-apis", action="store_true", help="Disable API engines (OpenAI/Sightengine)")
    ap.add_argument(
        "--sample-frames",
        type=int,
        default=_env_int("SAMPLE_FRAMES", 12),
        help="Max frames to sample from animated images",
    )
    ap.add_argument("--recursive", action="store_true", help="When input is a directory, recurse")
    ap.add_argument("--json", dest="json_out", default="", help="Write report(s) to JSON file")
    args = ap.parse_args(argv)

    if not args.input:
        ap.error("input is required (path/dir/url)")

    reports: List[Dict[str, Any]] = []
    for p in _iter_paths(args.input, args.recursive):
        rep = run_on_input(p, no_apis=args.no_apis, sample_frames=args.sample_frames)
        _print_report(rep)
        reports.append({
            "name": rep["name"],
            "path": rep["path"],
            "verdict": rep["verdict"].__dict__,
            "results": [r.__dict__ for r in rep["results"]],
            "auto_learn": rep.get("auto_learn"),
        })

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(reports if len(reports)>1 else reports[0], f, ensure_ascii=False, indent=2)

    # exit code: 0 if all OK, 2 otherwise
    if all(r["verdict"]["label"] == "OK" for r in reports):
        return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
