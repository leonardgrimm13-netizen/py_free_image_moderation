from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from statistics import median
from typing import Any


def enum_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def safe_int_ms(value: Any) -> int:
    if value is None:
        return 0
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return 0
    return out if out >= 0 else 0


def percentile(values: list[int | float], q: float) -> int:
    if not values:
        return 0
    cleaned = sorted(safe_int_ms(v) for v in values)
    qq = max(0.0, min(100.0, float(q)))
    rank = ceil((qq / 100.0) * len(cleaned)) - 1
    rank = max(0, min(len(cleaned) - 1, rank))
    return int(cleaned[rank])


def verdict_label_from_report(rep: dict) -> str:
    verdict = rep.get("verdict")
    label = None
    if verdict is None:
        return "UNKNOWN"
    if isinstance(verdict, dict):
        label = verdict.get("label")
    else:
        label = get_attr_or_key(verdict, "label", None)
        if label is None and isinstance(verdict, str):
            label = verdict
    if label is None:
        return "UNKNOWN"
    out = enum_value(label)
    return out if out else "UNKNOWN"


def collect_benchmark_item(rep: dict, total_ms: int) -> dict:
    engines = []
    results = rep.get("results") or []
    for result in results:
        name = str(get_attr_or_key(result, "name", "engine") or "engine")
        status = enum_value(get_attr_or_key(result, "status", "")).lower()
        took_ms = safe_int_ms(get_attr_or_key(result, "took_ms", 0))
        engines.append({"name": name, "status": status, "took_ms": took_ms})

    engine_total_ms = sum(e["took_ms"] for e in engines)
    total_ms_i = safe_int_ms(total_ms)
    unattributed_ms = max(0, total_ms_i - engine_total_ms)

    slowest_engine = None
    slowest_engine_ms = 0
    if engines:
        slowest = max(engines, key=lambda e: e["took_ms"])
        slowest_engine = slowest["name"]
        slowest_engine_ms = slowest["took_ms"]

    return {
        "name": str(rep.get("name", "")),
        "path": str(rep.get("path", "")),
        "verdict_label": verdict_label_from_report(rep),
        "total_ms": total_ms_i,
        "engine_total_ms": engine_total_ms,
        "unattributed_ms": unattributed_ms,
        "engine_count": len(engines),
        "slowest_engine": slowest_engine,
        "slowest_engine_ms": slowest_engine_ms,
        "engines": engines,
    }


def summarize_benchmark(items: list[dict], total_wall_ms: int | None = None) -> dict:
    file_times = [safe_int_ms(item.get("total_ms", 0)) for item in items]
    total_files = len(items)
    total_ms = sum(file_times)
    wall_ms = safe_int_ms(total_wall_ms) if total_wall_ms is not None else total_ms

    verdict_counts = Counter(str(item.get("verdict_label", "UNKNOWN")) for item in items)
    for k in ("OK", "REVIEW", "BLOCK"):
        verdict_counts.setdefault(k, 0)

    engine_rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "runs": 0,
            "ok": 0,
            "skipped": 0,
            "error": 0,
            "other_status": 0,
            "total_ms": 0,
            "times": [],
        }
    )

    for item in items:
        for engine in item.get("engines", []):
            name = str(engine.get("name", "engine"))
            status = str(engine.get("status", "")).lower()
            took_ms = safe_int_ms(engine.get("took_ms", 0))

            st = engine_rollup[name]
            st["runs"] += 1
            st["total_ms"] += took_ms
            st["times"].append(took_ms)
            if status == "ok":
                st["ok"] += 1
            elif status in ("skipped", "skip", "disabled"):
                st["skipped"] += 1
            elif status == "error":
                st["error"] += 1
            else:
                st["other_status"] += 1

    engine_stats: dict[str, dict[str, Any]] = {}
    for name, st in engine_rollup.items():
        times = [safe_int_ms(v) for v in st["times"]]
        runs = int(st["runs"])
        total_engine_ms = safe_int_ms(st["total_ms"])
        engine_stats[name] = {
            "runs": runs,
            "ok": int(st["ok"]),
            "skipped": int(st["skipped"]),
            "error": int(st["error"]),
            "other_status": int(st["other_status"]),
            "total_ms": total_engine_ms,
            "avg_ms": round(total_engine_ms / runs, 2) if runs else 0.0,
            "median_ms": int(median(times)) if times else 0,
            "min_ms": min(times) if times else 0,
            "max_ms": max(times) if times else 0,
            "p95_ms": percentile(times, 95),
        }

    slowest_files = sorted(items, key=lambda x: safe_int_ms(x.get("total_ms", 0)), reverse=True)[:10]
    slowest_files = [
        {
            "name": str(it.get("name", "")),
            "path": str(it.get("path", "")),
            "verdict_label": str(it.get("verdict_label", "UNKNOWN")),
            "total_ms": safe_int_ms(it.get("total_ms", 0)),
            "engine_total_ms": safe_int_ms(it.get("engine_total_ms", 0)),
            "unattributed_ms": safe_int_ms(it.get("unattributed_ms", 0)),
            "engine_count": safe_int_ms(it.get("engine_count", 0)),
            "slowest_engine": it.get("slowest_engine"),
            "slowest_engine_ms": safe_int_ms(it.get("slowest_engine_ms", 0)),
        }
        for it in slowest_files
    ]

    slowest_engines = [{"name": name, **stats} for name, stats in engine_stats.items()]
    slowest_engines.sort(key=lambda x: (-x["total_ms"], -x["avg_ms"], x["name"]))

    return {
        "version": 1,
        "total_files": total_files,
        "total_ms": total_ms,
        "total_wall_ms": wall_ms,
        "avg_file_ms": round(total_ms / total_files, 2) if total_files else 0.0,
        "median_file_ms": int(median(file_times)) if file_times else 0,
        "min_file_ms": min(file_times) if file_times else 0,
        "max_file_ms": max(file_times) if file_times else 0,
        "p95_file_ms": percentile(file_times, 95),
        "verdict_counts": dict(verdict_counts),
        "engine_stats": engine_stats,
        "slowest_files": slowest_files,
        "slowest_engines": slowest_engines,
    }


def format_benchmark_summary(summary: dict) -> str:
    verdicts = summary.get("verdict_counts", {})
    lines = [
        "=" * 70,
        "BENCHMARK",
        "=" * 70,
        (
            f"Files: {safe_int_ms(summary.get('total_files', 0))} | "
            f"Total: {safe_int_ms(summary.get('total_ms', 0))}ms | "
            f"Wall: {safe_int_ms(summary.get('total_wall_ms', 0))}ms | "
            f"Avg/file: {float(summary.get('avg_file_ms', 0.0)):.2f}ms | "
            f"Min: {safe_int_ms(summary.get('min_file_ms', 0))}ms | "
            f"Median: {safe_int_ms(summary.get('median_file_ms', 0))}ms | "
            f"P95: {safe_int_ms(summary.get('p95_file_ms', 0))}ms | "
            f"Max: {safe_int_ms(summary.get('max_file_ms', 0))}ms"
        ),
        f"Verdicts: OK={safe_int_ms(verdicts.get('OK', 0))} REVIEW={safe_int_ms(verdicts.get('REVIEW', 0))} BLOCK={safe_int_ms(verdicts.get('BLOCK', 0))}",
        "",
        "Per engine:",
    ]

    slowest_engines = summary.get("slowest_engines", []) or []
    if slowest_engines:
        for engine in slowest_engines:
            lines.append(
                " - {name} runs={runs} ok={ok} skipped={skipped} error={error} other={other} total={total}ms avg={avg:.2f}ms p95={p95}ms max={maxv}ms".format(
                    name=str(engine.get("name", "engine")),
                    runs=safe_int_ms(engine.get("runs", 0)),
                    ok=safe_int_ms(engine.get("ok", 0)),
                    skipped=safe_int_ms(engine.get("skipped", 0)),
                    error=safe_int_ms(engine.get("error", 0)),
                    other=safe_int_ms(engine.get("other_status", 0)),
                    total=safe_int_ms(engine.get("total_ms", 0)),
                    avg=float(engine.get("avg_ms", 0.0)),
                    p95=safe_int_ms(engine.get("p95_ms", 0)),
                    maxv=safe_int_ms(engine.get("max_ms", 0)),
                )
            )
    else:
        lines.append(" - none")

    lines.append("")
    lines.append("Slowest files:")
    slowest_files = summary.get("slowest_files", []) or []
    if slowest_files:
        for idx, item in enumerate(slowest_files, start=1):
            lines.append(
                f" {idx}. {item.get('name', '')} {safe_int_ms(item.get('total_ms', 0))}ms "
                f"verdict={item.get('verdict_label', 'UNKNOWN')} "
                f"slowest={item.get('slowest_engine')} {safe_int_ms(item.get('slowest_engine_ms', 0))}ms "
                f"unattributed={safe_int_ms(item.get('unattributed_ms', 0))}ms"
            )
    else:
        lines.append(" - none")

    return "\n".join(lines)
