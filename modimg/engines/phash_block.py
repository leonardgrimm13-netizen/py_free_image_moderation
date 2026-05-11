from __future__ import annotations

import os
from typing import List, Tuple, Optional

from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame
from ..utils import env_int_any, now_ms
from .. import phash as ph


class PHashBlocklistEngine(Engine):
    """Offline blocklist. If matched -> pipeline may short-circuit to BLOCK."""

    name = "pHash blocklist"

    def __init__(self, blocklist_path: Optional[str] = None, max_distance: Optional[int] = None) -> None:
        super().__init__()
        self.blocklist_path = blocklist_path or ph.get_blocklist_path()
        self.max_distance = int(
            max_distance
            if max_distance is not None
            else env_int_any(("PHASH_BLOCK_MAX_DISTANCE", "PHASH_MAXDIST", "PHASH_BLOCK_MAXDIST"), 6)
        )

    def available(self) -> Tuple[bool, str]:
        if os.getenv("PHASH_BLOCK_DISABLE", "0") == "1":
            return False, "disabled via PHASH_BLOCK_DISABLE=1"
        p = ph.resolve_list_path(self.blocklist_path)
        if not p:
            return False, "blocklist path not set"
        if not os.path.exists(p):
            return False, f"blocklist not found ({p})"
        return True, ""

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 2) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms() - start)

        fr_first = frames[0]
        fr_last = frames[-1]
        first_hex, first_int = ph.frame_phash_hex_int(fr_first)
        last_hex, last_int = ph.frame_phash_hex_int(fr_last)

        best: Optional[tuple[int, str, str, str]] = None  # (dist, hex, label, which)

        if self.max_distance <= 0:
            mp = ph.load_phash_exact_map(self.blocklist_path, default_label="block")
            if not mp:
                return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="blocklist empty", took_ms=now_ms() - start)
            found = mp.get(len(first_hex), {}).get(first_int)
            if found is not None:
                best = (0, found[0], found[1], "first")
            found2 = mp.get(len(last_hex), {}).get(last_int)
            if found2 is not None and best is None:
                best = (0, found2[0], found2[1], "last")
        else:
            entries = ph.load_phash_list(self.blocklist_path, default_label="block")
            if not entries:
                return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="blocklist empty", took_ms=now_ms() - start)
            bm = ph.best_match_distance(first_int, len(first_hex), entries, self.max_distance)
            if bm is not None:
                best = (bm[0], bm[1], bm[2], "first")
            bm2 = ph.best_match_distance(last_int, len(last_hex), entries, self.max_distance)
            if bm2 is not None and (best is None or bm2[0] < best[0]):
                best = (bm2[0], bm2[1], bm2[2], "last")

        if best is None:
            return EngineResult(
                name=self.name,
                status=EngineStatus.OK,
                scores={"phash_block_match": 0.0},
                details={"first": first_hex, "last": last_hex},
                took_ms=now_ms() - start,
            )

        dist, hx, label, which = best
        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={"phash_block_match": 1.0},
            details={
                "match_hex": hx,
                "match_label": label,
                "distance": int(dist),
                "matched_on": which,
                "first": first_hex,
                "last": last_hex,
            },
            took_ms=now_ms() - start,
        )
