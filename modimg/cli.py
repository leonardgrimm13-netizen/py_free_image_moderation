from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .config import get_config, load_dotenv_candidates
from .logging_utils import get_logger
from .pipeline import run_on_input
from .utils import is_image_file, is_url

LOGGER = get_logger("cli")


def _enum_value(v: Any) -> str:
    """Return Enum.value for serialization/output, fallback to str."""
    return str(v.value) if hasattr(v, "value") else str(v)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _select_scores(engine_name: str, scores: Dict[str, Any]) -> List[tuple[str, float]]:
    if os.getenv("SCORE_VERBOSE", "0").strip() == "1":
        return [(k, float(v)) for k, v in scores.items() if isinstance(v, (float, int))]

    if "sightengine" in (engine_name or "").lower():
        mode = (os.getenv("SIGHTENGINE_SCORE_MODE", "compact") or "compact").strip().lower()
        if mode in ("full", "all", "verbose"):
            return [(k, float(v)) for k, v in scores.items() if isinstance(v, (float, int))]
        if mode == "keys":
            keys_raw = os.getenv("SIGHTENGINE_SCORE_KEYS", "")
            out: List[tuple[str, float]] = []
            for k in [x.strip() for x in keys_raw.split(",") if x.strip()]:
                if k in scores and isinstance(scores[k], (float, int)):
                    out.append((k, float(scores[k])))
            return out

        preferred = [
            "nudity_safe", "nudity_raw", "nudity_partial", "weapon_firearm", "weapon_firearm_toy", "weapon_knife", "gore_prob", "violence_prob", "offensive_max",
        ]
        items = [(k, float(scores[k])) for k in preferred if k in scores and isinstance(scores[k], (float, int))]
        extra_topk = _env_int("SIGHTENGINE_EXTRA_TOPK", 0)
        if extra_topk > 0:
            rest = sorted(
                [(k, float(v)) for k, v in scores.items() if k not in preferred and isinstance(v, (float, int))],
                key=lambda kv: kv[1],
                reverse=True,
            )
            items.extend([(k, v) for k, v in rest[:extra_topk] if v >= 0.05])
        return items

    max_keys = _env_int("SCORE_MAX_KEYS", 8)
    rest = sorted([(k, float(v)) for k, v in scores.items() if isinstance(v, (float, int))], key=lambda kv: kv[1], reverse=True)
    return rest[:max_keys]


def _iter_paths(p: str, recursive: bool) -> List[str]:
    if is_url(p):
        return [p]
    path = Path(p)
    if path.is_dir():
        if recursive:
            return sorted(str(x) for x in path.rglob("*") if x.is_file() and is_image_file(str(x)))
        return sorted(str(x) for x in path.iterdir() if x.is_file() and is_image_file(str(x)))
    return [p]


def _print_report(rep: Dict[str, Any]) -> None:
    v = rep["verdict"]
    results = rep["results"]

    LOGGER.info("%s", "=" * 70)
    LOGGER.info("%s", rep["name"])
    LOGGER.info(
        "FINAL: %s  (verdict=%s) | nudity=%.2f violence=%.2f hate=%.2f",
        "OK" if _enum_value(v.label) == "OK" else "NOT_OK",
        _enum_value(v.label),
        v.nudity_risk,
        v.violence_risk,
        v.hate_risk,
    )
    for reason in v.reasons:
        LOGGER.info(" - %s", reason)
    if rep.get("auto_learn"):
        LOGGER.info(" - %s", rep["auto_learn"])

    for r in results:
        st = _enum_value(r.status).lower()
        msg = ""
        if st == "ok" and r.scores:
            msg = ", ".join(f"{k}={float(vv):.2f}" for k, vv in _select_scores(r.name, r.scores))
        elif r.error:
            msg = r.error
        LOGGER.info("   [%-7s] %-22s (%sms) %s", st, r.name, int(r.took_ms or 0), msg)


def main(argv: List[str] | None = None) -> int:
    load_dotenv_candidates()
    cfg = get_config(reload=True)
    ap = argparse.ArgumentParser(description="Moderate an image/GIF or folder with multiple optional engines.")
    ap.add_argument("input", nargs="?", default="", help="Path/dir/URL to moderate")
    ap.add_argument("--no-apis", action="store_true", help="Disable API engines (OpenAI/Sightengine)")
    ap.add_argument("--sample-frames", type=int, default=cfg.sample_frames, help="Max frames to sample from animated images")
    ap.add_argument("--recursive", action="store_true", help="When input is a directory, recurse")
    ap.add_argument("--json", dest="json_out", default="", help="Write report(s) to JSON file")
    args = ap.parse_args(argv)

    if not args.input:
        ap.error("input is required (path/dir/url)")

    reports: List[Dict[str, Any]] = []
    for p in _iter_paths(args.input, args.recursive):
        rep = run_on_input(p, no_apis=args.no_apis, sample_frames=args.sample_frames)
        _print_report(rep)
        reports.append(
            {
                "name": rep["name"],
                "path": rep["path"],
                "verdict": {
                    **rep["verdict"].__dict__,
                    "label": _enum_value(rep["verdict"].label),
                },
                "results": [
                    {
                        **r.__dict__,
                        "status": _enum_value(r.status),
                    }
                    for r in rep["results"]
                ],
                "auto_learn": rep.get("auto_learn"),
            }
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2), encoding="utf-8")

    return 0 if all(r["verdict"]["label"] == "OK" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
