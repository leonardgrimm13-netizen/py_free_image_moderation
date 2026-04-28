from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from .types import EngineResult, Verdict, Frame
from .utils import is_url, download_url_to_temp, now_ms
from .frames import load_frames
from .verdict import compute_verdict
from .phash import (
    append_phash_to_allowlist,
    append_phash_to_blocklist,
    frame_phash_hex_int,
    get_allowlist_path,
    get_blocklist_path,
)
from .engines import (
    PHashAllowlistEngine, PHashBlocklistEngine, OCREngine,
    NudeNetEngine, OpenNSFW2Engine, YOLOWorldWeaponsEngine,
    OpenAIModerationEngine, SightengineEngine,
)

def build_pre_engines(*, no_apis: bool = False) -> List[Any]:
    """Engines that should run first and may short-circuit the entire pipeline."""
    # Safety: blocklist should take precedence over allowlist.
    return [PHashBlocklistEngine(), PHashAllowlistEngine()]

def build_main_engines(*, no_apis: bool = False) -> List[Any]:
    """All other engines (potentially slow/expensive)."""
    engines: List[Any] = []
    engines.append(OCREngine())
    engines.append(NudeNetEngine())
    engines.append(OpenNSFW2Engine())
    engines.append(YOLOWorldWeaponsEngine())
    if not no_apis:
        engines.append(OpenAIModerationEngine())
        engines.append(SightengineEngine())
    return engines

def run_engines(path: str, frames: List[Frame], engines: List[Any]) -> List[EngineResult]:
    results: List[EngineResult] = []
    for eng in engines:
        t0 = now_ms()
        try:
            ok, why = eng.available()
            if not ok:
                results.append(EngineResult(name=eng.name, status="skipped", error=why, took_ms=now_ms()-t0))
                continue
            res = eng.run(path, frames)
            # if engine didn't set took_ms
            if res.took_ms is None:
                res.took_ms = now_ms()-t0
            results.append(res)
        except Exception as e:
            results.append(EngineResult(name=getattr(eng, "name", "engine"), status="error", error=f"{type(e).__name__}: {e}", details={"trace": traceback.format_exc()[-2000:]}, took_ms=now_ms()-t0))
    return results

def _short_circuit_from_phash(results: List[EngineResult]) -> Optional[Verdict]:
    # Prefer BLOCK over OK if both somehow match.
    block = None
    allow = None
    for r in results:
        if r.status != "ok":
            continue
        if r.name == "pHash blocklist" and r.scores.get("phash_block_match") == 1.0:
            block = Verdict("BLOCK", 1.0, 1.0, 1.0, [f"Blocklist match (distance={r.details.get('distance')})"])
        if r.name == "pHash allowlist" and r.scores.get("phash_allow_match") == 1.0:
            allow = Verdict("OK", 0.0, 0.0, 0.0, [f"Allowlist match (distance={r.details.get('distance')})"])
    return block or allow

def maybe_auto_learn(verdict: Verdict, frames: List[Frame]) -> Optional[str]:
    """Auto-append pHash to allow/block lists if enabled. Returns message or None."""
    try:
        if not frames:
            return None
        # Master switch (preferred): PHASH_AUTO_LEARN_ENABLE
        # Back-compat: older builds used PHASH_AUTO_APPEND / PHASH_AUTO_ALLOW_APPEND.
        auto_learn = os.getenv("PHASH_AUTO_LEARN_ENABLE", "0").strip() == "1"
        if not auto_learn:
            # If the master switch is off, honor legacy flags only.
            legacy_any = os.getenv("PHASH_AUTO_APPEND", "0").strip() == "1" or os.getenv("PHASH_AUTO_ALLOW_APPEND", "0").strip() == "1"
            if not legacy_any:
                return None
        learn_first_last = os.getenv("PHASH_GIF_LEARN_FIRST_LAST", "0").strip() == "1"
        # Determine hashes to append
        frs = [frames[0], frames[-1]] if learn_first_last and len(frames) > 1 else [frames[0]]
        hashes = []
        for fr in frs:
            hx, _ = frame_phash_hex_int(fr)
            hashes.append(hx)

        # Defaults when PHASH_AUTO_LEARN_ENABLE=1:
        # - OK -> allowlist append ON
        # - BLOCK -> blocklist append ON
        # - REVIEW -> blocklist append OFF (to avoid poisoning lists)
        allow_append = os.getenv("PHASH_AUTO_ALLOW_APPEND", "" ).strip()
        block_append = os.getenv("PHASH_AUTO_BLOCK_APPEND", "" ).strip()
        if auto_learn:
            if allow_append == "":
                allow_append = "1"
            if block_append == "":
                block_append = "1"  # only used for BLOCK (see below)

        if verdict.label == "OK" and allow_append == "1":
            label = os.getenv("PHASH_AUTO_ALLOW_LABEL", os.getenv("PHASH_AUTO_LABEL", "ok")).strip() or "ok"
            apath = get_allowlist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_allowlist(hx, apath, label) or added_any
            if added_any:
                return f"Auto-added pHash to allowlist ({apath})"
        # Blocklist learning is intentionally stricter: only learn from BLOCK by default.
        if verdict.label == "BLOCK" and block_append == "1":
            label = os.getenv("PHASH_AUTO_BLOCK_LABEL", os.getenv("PHASH_AUTO_LABEL", "not_ok")).strip() or "not_ok"
            bpath = get_blocklist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_blocklist(hx, bpath, label) or added_any
            if added_any:
                return f"Auto-added pHash to blocklist ({bpath})"
    except Exception:
        return None
    return None

def run_on_input(inp: str, *, no_apis: bool = False, sample_frames: int = 12) -> Dict[str, Any]:
    tmp_path = None
    display_name = inp

    try:
        if is_url(inp):
            tmp_path, display_name = download_url_to_temp(inp)
            path = tmp_path
        else:
            path = inp

        frames = load_frames(path, sample_frames=sample_frames)
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        # Can't parse the image -> require review rather than crashing.
        v = Verdict(
            label="REVIEW",
            nudity_risk=0.0,
            violence_risk=0.0,
            hate_risk=0.0,
            reasons=[f"loader_failure: {type(e).__name__}: {e}"],
        )
        return {
            "name": display_name,
            "path": inp,
            "verdict": v,
            "results": [EngineResult(name="Loader", status="error", error=f"failed to load image: {type(e).__name__}: {e}")],
            "auto_learn": "",
        }

    # 1) Run pre-checks (fast short-circuit).
    pre_engines = build_pre_engines(no_apis=no_apis)
    pre_results = run_engines(path, frames, pre_engines)
    sc = _short_circuit_from_phash(pre_results)

    if sc is not None and os.getenv("SHORT_CIRCUIT_PHASH", "1").strip() != "0":
        # pHash allow/block decides — skip everything else.
        results = pre_results
        v = sc
    else:
        # 2) Run the rest.
        main_engines = build_main_engines(no_apis=no_apis)
        main_results = run_engines(path, frames, main_engines)
        results = pre_results + main_results
        v = compute_verdict(results)

    learn_msg = maybe_auto_learn(v, frames)

    if tmp_path:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return {
        "name": display_name,
        "path": inp,
        "verdict": v,
        "results": results,
        "auto_learn": learn_msg,
    }
