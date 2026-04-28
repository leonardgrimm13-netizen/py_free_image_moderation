from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .enums import EngineStatus, VerdictLabel
from .logging_utils import get_logger
from .types import Engine, EngineResult, Verdict, Frame
from .utils import is_url, download_url_to_temp
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
    PHashAllowlistEngine,
    PHashBlocklistEngine,
    OCREngine,
    NudeNetEngine,
    OpenNSFW2Engine,
    YOLOWorldWeaponsEngine,
    OpenAIModerationEngine,
    SightengineEngine,
)

LOGGER = get_logger("pipeline")


def build_pre_engines(*, no_apis: bool = False) -> List[Engine]:
    return [PHashBlocklistEngine(), PHashAllowlistEngine()]


def build_main_engines(*, no_apis: bool = False) -> List[Engine]:
    engines: List[Engine] = [OCREngine(), NudeNetEngine(), OpenNSFW2Engine(), YOLOWorldWeaponsEngine()]
    cfg = get_config()
    if (not no_apis) and (not cfg.openai_disable):
        engines.append(OpenAIModerationEngine())
    if (not no_apis) and (not cfg.sightengine_disable):
        engines.append(SightengineEngine())
    return engines


def _run_single_engine(path: str, frames: List[Frame], engine: Engine) -> EngineResult:
    try:
        return engine.execute(path, frames)
    except Exception as exc:  # last-resort protection
        details: Dict[str, Any] = {}
        if get_config().debug:
            details["trace"] = traceback.format_exc()[-2000:]
        return EngineResult(
            name=getattr(engine, "name", "engine"),
            status=EngineStatus.ERROR,
            error=f"{type(exc).__name__}: {exc}",
            details=details,
        )


def run_engines(path: str, frames: List[Frame], engines: List[Engine]) -> List[EngineResult]:
    cfg = get_config()
    if not cfg.parallel_engines or len(engines) <= 1:
        return [_run_single_engine(path, frames, eng) for eng in engines]

    results: List[EngineResult] = []
    with ThreadPoolExecutor(max_workers=min(cfg.parallel_workers, len(engines))) as executor:
        futures = {executor.submit(_run_single_engine, path, frames, eng): idx for idx, eng in enumerate(engines)}
        ordered: Dict[int, EngineResult] = {}
        for fut in as_completed(futures):
            ordered[futures[fut]] = fut.result()
        for idx in range(len(engines)):
            results.append(ordered[idx])
    return results


def _short_circuit_from_phash(results: List[EngineResult]) -> Optional[Verdict]:
    block: Optional[Verdict] = None
    allow: Optional[Verdict] = None
    for r in results:
        if r.status != EngineStatus.OK:
            continue
        if r.name == "pHash blocklist" and r.scores.get("phash_block_match") == 1.0:
            block = Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, [f"Blocklist match (distance={r.details.get('distance')})"])
        if r.name == "pHash allowlist" and r.scores.get("phash_allow_match") == 1.0:
            allow = Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, [f"Allowlist match (distance={r.details.get('distance')})"])
    return block or allow


def maybe_auto_learn(verdict: Verdict, frames: List[Frame]) -> Optional[str]:
    try:
        if not frames:
            return None
        auto_learn = os.getenv("PHASH_AUTO_LEARN_ENABLE", "0").strip() == "1"
        if not auto_learn:
            legacy_any = os.getenv("PHASH_AUTO_APPEND", "0").strip() == "1" or os.getenv("PHASH_AUTO_ALLOW_APPEND", "0").strip() == "1"
            if not legacy_any:
                return None
        learn_first_last = os.getenv("PHASH_GIF_LEARN_FIRST_LAST", "0").strip() == "1"
        frs = [frames[0], frames[-1]] if learn_first_last and len(frames) > 1 else [frames[0]]
        hashes = [frame_phash_hex_int(fr)[0] for fr in frs]

        allow_append = os.getenv("PHASH_AUTO_ALLOW_APPEND", "").strip()
        block_append = os.getenv("PHASH_AUTO_BLOCK_APPEND", "").strip()
        if auto_learn:
            if allow_append == "":
                allow_append = "1"
            if block_append == "":
                block_append = "1"

        if verdict.label == VerdictLabel.OK and allow_append == "1":
            label = os.getenv("PHASH_AUTO_ALLOW_LABEL", os.getenv("PHASH_AUTO_LABEL", "ok")).strip() or "ok"
            apath = get_allowlist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_allowlist(hx, apath, label) or added_any
            if added_any:
                return f"Auto-added pHash to allowlist ({apath})"

        if verdict.label == VerdictLabel.BLOCK and block_append == "1":
            label = os.getenv("PHASH_AUTO_BLOCK_LABEL", os.getenv("PHASH_AUTO_LABEL", "not_ok")).strip() or "not_ok"
            bpath = get_blocklist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_blocklist(hx, bpath, label) or added_any
            if added_any:
                return f"Auto-added pHash to blocklist ({bpath})"
    except (ValueError, OSError):
        return None
    return None


def run_on_input(inp: str, *, no_apis: bool = False, sample_frames: int = 12) -> Dict[str, Any]:
    tmp_path: Optional[str] = None
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
            Path(tmp_path).unlink(missing_ok=True)
        v = Verdict(VerdictLabel.REVIEW, 0.0, 0.0, 0.0, [f"loader_failure: {type(e).__name__}: {e}"])
        return {
            "name": display_name,
            "path": inp,
            "verdict": v,
            "results": [EngineResult(name="Loader", status=EngineStatus.ERROR, error=f"failed to load image: {type(e).__name__}: {e}")],
            "auto_learn": "",
        }

    pre_results = run_engines(path, frames, build_pre_engines(no_apis=no_apis))
    sc = _short_circuit_from_phash(pre_results)

    if sc is not None and get_config().short_circuit_phash:
        results = pre_results
        v = sc
    else:
        main_results = run_engines(path, frames, build_main_engines(no_apis=no_apis))
        results = pre_results + main_results
        v = compute_verdict(results)

    learn_msg = maybe_auto_learn(v, frames)

    if tmp_path:
        Path(tmp_path).unlink(missing_ok=True)

    return {"name": display_name, "path": inp, "verdict": v, "results": results, "auto_learn": learn_msg}
