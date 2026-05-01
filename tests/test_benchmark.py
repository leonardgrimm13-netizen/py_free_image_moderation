from __future__ import annotations

from modimg.benchmark import (
    collect_benchmark_item,
    enum_value,
    format_benchmark_summary,
    percentile,
    summarize_benchmark,
)
from modimg.enums import EngineStatus, VerdictLabel
from modimg.types import EngineResult, Verdict


def test_enum_value_handles_none_enum_and_string() -> None:
    assert enum_value(None) == ""
    assert enum_value("abc") == "abc"
    assert enum_value(EngineStatus.OK) == "ok"
    assert enum_value(VerdictLabel.REVIEW) == "REVIEW"


def test_percentile_empty_single_and_multiple_values() -> None:
    assert percentile([], 95) == 0
    assert percentile([123], 95) == 123
    assert percentile([1, 2, 3, 4, 100], 95) >= 4
    assert percentile([1, 2, 3], -10) == 1
    assert percentile([1, 2, 3], 200) == 3


def test_collect_benchmark_item_basic() -> None:
    rep = {
        "name": "sample.png",
        "path": "sample.png",
        "verdict": Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, []),
        "results": [
            EngineResult(name="EngineA", status=EngineStatus.OK, took_ms=10),
            EngineResult(name="EngineB", status=EngineStatus.SKIPPED, took_ms=5),
            EngineResult(name="EngineC", status="error", took_ms=None),
        ],
    }
    item = collect_benchmark_item(rep, total_ms=30)
    assert item["name"] == "sample.png"
    assert item["path"] == "sample.png"
    assert item["verdict_label"] == "OK"
    assert item["total_ms"] == 30
    assert item["engine_total_ms"] == 15
    assert item["unattributed_ms"] == 15
    assert item["engine_count"] == 3
    assert item["slowest_engine"] == "EngineA"
    assert item["slowest_engine_ms"] == 10
    assert item["engines"][0]["status"] == "ok"
    assert item["engines"][1]["status"] == "skipped"
    assert item["engines"][2]["took_ms"] == 0


def test_collect_benchmark_item_accepts_dict_results() -> None:
    rep = {
        "name": "dict.png",
        "path": "dict.png",
        "verdict": {"label": "BLOCK"},
        "results": [{"name": "DictEngine", "status": "ok", "took_ms": "12"}],
    }
    item = collect_benchmark_item(rep, total_ms=20)
    assert item["verdict_label"] == "BLOCK"
    assert item["engine_total_ms"] == 12
    assert item["slowest_engine"] == "DictEngine"


def test_summarize_benchmark_basic() -> None:
    items = [
        {
            "name": "a.png",
            "path": "a.png",
            "verdict_label": "OK",
            "total_ms": 10,
            "engine_total_ms": 9,
            "unattributed_ms": 1,
            "engine_count": 2,
            "slowest_engine": "EngineA",
            "slowest_engine_ms": 5,
            "engines": [
                {"name": "EngineA", "status": "ok", "took_ms": 5},
                {"name": "EngineB", "status": "skipped", "took_ms": 4},
            ],
        },
        {
            "name": "b.png",
            "path": "b.png",
            "verdict_label": "REVIEW",
            "total_ms": 30,
            "engine_total_ms": 20,
            "unattributed_ms": 10,
            "engine_count": 2,
            "slowest_engine": "EngineA",
            "slowest_engine_ms": 12,
            "engines": [
                {"name": "EngineA", "status": "ok", "took_ms": 12},
                {"name": "EngineB", "status": "disabled", "took_ms": 8},
            ],
        },
        {
            "name": "c.png",
            "path": "c.png",
            "verdict_label": "BLOCK",
            "total_ms": 20,
            "engine_total_ms": 17,
            "unattributed_ms": 3,
            "engine_count": 2,
            "slowest_engine": "EngineA",
            "slowest_engine_ms": 10,
            "engines": [
                {"name": "EngineA", "status": "error", "took_ms": 10},
                {"name": "EngineB", "status": "skip", "took_ms": 7},
            ],
        },
    ]
    summary = summarize_benchmark(items, total_wall_ms=70)
    assert summary["version"] == 1
    assert summary["total_files"] == 3
    assert summary["total_ms"] == 60
    assert summary["total_wall_ms"] == 70
    assert summary["avg_file_ms"] == 20.0
    assert summary["min_file_ms"] == 10
    assert summary["max_file_ms"] == 30
    assert summary["median_file_ms"] == 20
    assert isinstance(summary["p95_file_ms"], int)
    assert summary["verdict_counts"]["OK"] == 1
    assert summary["verdict_counts"]["REVIEW"] == 1
    assert summary["verdict_counts"]["BLOCK"] == 1
    assert summary["engine_stats"]["EngineA"]["runs"] == 3
    assert summary["engine_stats"]["EngineA"]["ok"] == 2
    assert summary["engine_stats"]["EngineA"]["error"] == 1
    assert summary["engine_stats"]["EngineB"]["skipped"] == 3
    assert "avg_ms" in summary["engine_stats"]["EngineA"]
    assert summary["slowest_files"][0]["total_ms"] >= summary["slowest_files"][1]["total_ms"]
    assert summary["slowest_engines"][0]["total_ms"] >= summary["slowest_engines"][1]["total_ms"]


def test_summarize_benchmark_empty() -> None:
    summary = summarize_benchmark([], total_wall_ms=0)
    assert summary["version"] == 1
    assert summary["total_files"] == 0
    assert summary["total_ms"] == 0
    assert summary["total_wall_ms"] == 0
    assert summary["avg_file_ms"] == 0
    assert summary["min_file_ms"] == 0
    assert summary["max_file_ms"] == 0
    assert summary["median_file_ms"] == 0
    assert summary["p95_file_ms"] == 0
    assert summary["verdict_counts"]["OK"] == 0
    assert summary["verdict_counts"]["REVIEW"] == 0
    assert summary["verdict_counts"]["BLOCK"] == 0
    assert summary["engine_stats"] == {}
    assert summary["slowest_files"] == []
    assert summary["slowest_engines"] == []


def test_format_benchmark_summary_contains_expected_text() -> None:
    summary = summarize_benchmark(
        [
            {
                "name": "a.png",
                "path": "a.png",
                "verdict_label": "OK",
                "total_ms": 12,
                "engine_total_ms": 10,
                "unattributed_ms": 2,
                "engine_count": 1,
                "slowest_engine": "EngineA",
                "slowest_engine_ms": 10,
                "engines": [{"name": "EngineA", "status": "ok", "took_ms": 10}],
            }
        ],
        total_wall_ms=13,
    )
    text = format_benchmark_summary(summary)
    assert "BENCHMARK" in text
    assert "Files" in text
    assert "Verdicts" in text
    assert "Per engine" in text
    assert "Slowest files" in text
    assert "EngineA" in text
