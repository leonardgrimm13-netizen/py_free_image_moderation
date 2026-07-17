from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import get_config, load_dotenv_candidates
from .benchmark import collect_benchmark_item, format_benchmark_summary, summarize_benchmark
from .enums import EngineStatus, VerdictLabel
from .engines import OpenAIRunState, SightengineRunState
from .logging_utils import get_logger
from .pipeline import run_on_input
from .types import EngineResult, Verdict
from .utils import atomic_write_text, env_int, is_image_file, is_url, json_dumps_safe, json_safe, redact_sensitive_text, redact_url

LOGGER = get_logger("cli")


def _enum_value(v: Any) -> str:
    """Return Enum.value for serialization/output, fallback to str."""
    return str(v.value) if hasattr(v, "value") else str(v)


def _select_scores(engine_name: str, scores: Dict[str, Any]) -> List[tuple[str, float]]:
    if os.getenv("SCORE_VERBOSE", "0").strip() == "1":
        return [(k, float(v)) for k, v in scores.items() if isinstance(v, (float, int))]

    if (engine_name or "") == "YOLO forbidden symbols":
        preferred = [
            "forbidden_symbols_max_conf",
            "forbidden_symbols_detection_count",
            "forbidden_symbols_review_hit",
            "forbidden_symbols_block_hit",
        ]
        return [(k, float(scores[k])) for k in preferred if k in scores and isinstance(scores[k], (float, int))]

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
        extra_topk = env_int("SIGHTENGINE_EXTRA_TOPK", 0)
        if extra_topk > 0:
            rest = sorted(
                [(k, float(v)) for k, v in scores.items() if k not in preferred and isinstance(v, (float, int))],
                key=lambda kv: kv[1],
                reverse=True,
            )
            items.extend([(k, v) for k, v in rest[:extra_topk] if v >= 0.05])
        return items

    max_keys = env_int("SCORE_MAX_KEYS", 8)
    rest = sorted([(k, float(v)) for k, v in scores.items() if isinstance(v, (float, int))], key=lambda kv: kv[1], reverse=True)
    return rest[:max_keys]


def _iter_paths(p: str, recursive: bool) -> List[str]:
    if is_url(p):
        return [p]
    path = Path(p).expanduser()
    if path.is_dir():
        if recursive:
            return sorted(str(x) for x in path.rglob("*") if x.is_file() and is_image_file(str(x)))
        return sorted(str(x) for x in path.iterdir() if x.is_file() and is_image_file(str(x)))
    return [str(path)]


def _input_identity(value: str) -> str:
    if is_url(value):
        return f"url:{value}"
    try:
        normalized = os.path.normcase(str(Path(value).resolve(strict=False)))
    except OSError:
        normalized = os.path.normcase(os.path.abspath(value))
    return f"path:{normalized}"


def _expand_inputs(values: List[str], recursive: bool) -> List[str]:
    inputs: List[str] = []
    seen: set[str] = set()
    for value in values:
        for expanded in _iter_paths(value, recursive):
            identity = _input_identity(expanded)
            if identity in seen:
                continue
            seen.add(identity)
            inputs.append(expanded)
    return inputs


def _write_json_file(path: str, payload: Any) -> None:
    out = Path(path).expanduser()
    atomic_write_text(out, json_dumps_safe(payload, ensure_ascii=False, indent=2))


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


def _serialize_report(rep: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
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
    if rep.get("preprocessing") is not None:
        payload["preprocessing"] = rep["preprocessing"]
    return json_safe(payload)


def _error_report(inp: str, exc: Exception) -> Dict[str, Any]:
    safe_input = redact_url(inp) if is_url(inp) else inp
    error = redact_sensitive_text(f"{type(exc).__name__}: {exc}", extra_secrets=(inp,) if is_url(inp) else ())
    details: Dict[str, Any] = {}
    if get_config().debug:
        details["trace"] = redact_sensitive_text(
            traceback.format_exc()[-2000:],
            extra_secrets=(inp,) if is_url(inp) else (),
        )
    return {
        "name": safe_input,
        "path": safe_input,
        "verdict": Verdict(
            VerdictLabel.REVIEW,
            0.0,
            0.0,
            0.0,
            [f"processing_failure: {error}"],
        ),
        "results": [
            EngineResult(
                name="Processor",
                status=EngineStatus.ERROR,
                error=f"failed to process image: {error}",
                details=details,
            )
        ],
        "auto_learn": "",
    }


def _process_input(
    idx: int,
    inp: str,
    *,
    no_apis: bool,
    sample_frames: int,
    benchmark_enabled: bool,
    openai_run_state: OpenAIRunState,
    sightengine_run_state: SightengineRunState,
) -> Tuple[int, Dict[str, Any], Dict[str, Any] | None]:
    file_start = time.perf_counter() if benchmark_enabled else 0.0
    try:
        rep = run_on_input(
            inp,
            no_apis=no_apis,
            sample_frames=sample_frames,
            openai_run_state=openai_run_state,
            sightengine_run_state=sightengine_run_state,
        )
    except Exception as exc:
        rep = _error_report(inp, exc)
    benchmark_item = None
    if benchmark_enabled:
        file_total_ms = int((time.perf_counter() - file_start) * 1000)
        benchmark_item = collect_benchmark_item(rep, file_total_ms)
    return idx, rep, benchmark_item


def main(argv: List[str] | None = None) -> int:
    load_dotenv_candidates(include_cwd=True)
    cfg = get_config(reload=True)
    ap = argparse.ArgumentParser(description="Moderate images, GIFs, SVGs, and AVIF files with multiple optional engines.")
    ap.add_argument("input", nargs="*", help="Path(s), directory/directories, or URL(s) to moderate")
    ap.add_argument("--no-apis", action="store_true", help="Disable API engines (OpenAI/Sightengine)")
    ap.add_argument("--sample-frames", type=int, default=cfg.sample_frames, help="Max frames to sample from animated images")
    ap.add_argument("--recursive", action="store_true", help="When input is a directory, recurse")
    ap.add_argument("--json", dest="json_out", default="", help="Write report(s) to JSON file")
    ap.add_argument("--benchmark", action="store_true", help="Print runtime benchmark summary")
    ap.add_argument("--benchmark-json", default="", help="Write runtime benchmark summary to JSON file")
    ap.add_argument(
        "--file-workers",
        type=int,
        default=cfg.file_workers,
        help="Number of input files to process concurrently (env: MODIMG_FILE_WORKERS; default: 1)",
    )
    args = ap.parse_args(argv)
    benchmark_enabled = bool(args.benchmark or args.benchmark_json)
    file_workers = min(
        max(1, int(args.file_workers or 1)),
        max(1, env_int("MODIMG_MAX_FILE_WORKERS", 32)),
    )

    raw_inputs = [str(value) for value in args.input if str(value).strip()]
    if not raw_inputs:
        ap.error("input is required (path/dir/url)")

    inputs = _expand_inputs(raw_inputs, args.recursive)
    if not inputs:
        message = "No supported image, GIF, SVG, or AVIF files found."
        LOGGER.error("%s", message)
        if args.json_out:
            _write_json_file(args.json_out, {"error": message, "results": []})
        return 2

    reports: List[Dict[str, Any]] = []
    benchmark_items: List[Dict[str, Any]] = []
    openai_run_state = OpenAIRunState()
    sightengine_run_state = SightengineRunState()
    bench_start = time.perf_counter() if benchmark_enabled else None
    processing_wall_ms: int | None = None
    if file_workers <= 1 or len(inputs) <= 1:
        for idx, p in enumerate(inputs):
            _, rep, benchmark_item = _process_input(
                idx,
                p,
                no_apis=args.no_apis,
                sample_frames=args.sample_frames,
                benchmark_enabled=benchmark_enabled,
                openai_run_state=openai_run_state,
                sightengine_run_state=sightengine_run_state,
            )
            if benchmark_item is not None:
                benchmark_items.append(benchmark_item)
            _print_report(rep)
            reports.append(_serialize_report(rep))
    else:
        completed_map: Dict[int, Tuple[int, Dict[str, Any], Dict[str, Any] | None]] = {}
        with ThreadPoolExecutor(max_workers=min(file_workers, len(inputs))) as executor:
            futures = {
                executor.submit(
                    _process_input,
                    idx,
                    p,
                    no_apis=args.no_apis,
                    sample_frames=args.sample_frames,
                    benchmark_enabled=benchmark_enabled,
                    openai_run_state=openai_run_state,
                    sightengine_run_state=sightengine_run_state,
                ): idx
                for idx, p in enumerate(inputs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    completed_map[idx] = fut.result()
                except Exception as exc:
                    rep = _error_report(inputs[idx], exc)
                    benchmark_item = collect_benchmark_item(rep, 0) if benchmark_enabled else None
                    completed_map[idx] = (idx, rep, benchmark_item)
        completed = [completed_map[idx] for idx in range(len(inputs))]

        for _, rep, benchmark_item in completed:
            if benchmark_item is not None:
                benchmark_items.append(benchmark_item)
            _print_report(rep)
            reports.append(_serialize_report(rep))

    if benchmark_enabled:
        processing_wall_ms = int((time.perf_counter() - bench_start) * 1000) if bench_start is not None else None

    benchmark_summary = None
    if benchmark_enabled:
        benchmark_summary = summarize_benchmark(benchmark_items, total_wall_ms=processing_wall_ms)

    if args.json_out:
        _write_json_file(args.json_out, reports if len(reports) > 1 else reports[0])
    if benchmark_summary is not None and args.benchmark:
        LOGGER.info("%s", format_benchmark_summary(benchmark_summary))
    if benchmark_summary is not None and args.benchmark_json:
        _write_json_file(args.benchmark_json, benchmark_summary)

    return 0 if all(r["verdict"]["label"] == "OK" for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
