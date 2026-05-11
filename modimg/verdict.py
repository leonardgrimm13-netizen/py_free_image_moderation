from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .config import get_config
from .enums import EngineStatus, VerdictLabel
from .types import EngineResult, Verdict
from .utils import env_bool, env_float, safe_float01, status_value


@dataclass
class _VerdictState:
    reasons: List[str]
    nudity: float = 0.0
    violence: float = 0.0
    hate: float = 0.0

    def bump(self, attr: str, value: float, reason: str, threshold: float) -> None:
        if value >= threshold:
            self.reasons.append(reason)
        setattr(self, attr, max(getattr(self, attr), value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _no_checks_policy() -> str:
    policy = (os.getenv("NO_CHECKS_POLICY", "review") or "review").strip().lower()
    if policy in ("ok", "allow", "open"):
        return "ok"
    if policy in ("block", "deny", "fail", "fail_closed"):
        return "block"
    return "review"


def _core_engine_names() -> set[str]:
    aliases = {
        "phash_allowlist": "pHash allowlist",
        "phash_blocklist": "pHash blocklist",
        "phash_allow": "pHash allowlist",
        "phash_block": "pHash blocklist",
        "ocr": "OCR text",
        "openai": "OpenAI Moderation",
        "sightengine": "Sightengine",
        "forbidden_symbols_yolo": "YOLO forbidden symbols",
        "forbidden_symbols": "YOLO forbidden symbols",
        "yolo_forbidden_symbols": "YOLO forbidden symbols",
    }
    core_env = (os.getenv("CORE_ENGINES", "") or "").strip()
    if core_env:
        return {aliases.get(c.strip().lower(), c.strip()) for c in core_env.split(",") if c.strip()}
    return {"pHash allowlist", "pHash blocklist", "OCR text", "OpenAI Moderation", "Sightengine", "YOLO forbidden symbols"}


def _apply_error_policy(results: List[EngineResult], state: _VerdictState) -> Verdict | None:
    core_set = _core_engine_names()
    err_all = [r for r in results if status_value(r.status) == EngineStatus.ERROR.value]
    err_core = [r for r in err_all if (not core_set) or (r.name in core_set)]
    if err_core:
        names = ", ".join([r.name for r in err_core[:6]])
        state.reasons.append(f"Some checks failed: {names}")
        policy = (os.getenv("ENGINE_ERROR_POLICY", "review") or "review").strip().lower()
        if policy in ("lenient", "loose"):
            policy = "ignore"
        elif policy in ("strict", "hard"):
            policy = "review"
        if policy in ("block", "not_ok", "fail", "fail_closed", "deny"):
            return Verdict(VerdictLabel.BLOCK, max(state.nudity, 0.5), max(state.violence, 0.5), max(state.hate, 0.5), state.reasons)
        if policy not in ("ignore", "open", "allow"):
            state.nudity = max(state.nudity, 0.40)
            state.violence = max(state.violence, 0.40)
            state.hate = max(state.hate, 0.40)
    if err_all and not err_core:
        names = ", ".join([r.name for r in err_all[:6]])
        state.reasons.append(f"Non-core checks failed (ignored): {names}")
    return None


def _apply_no_checks(results: List[EngineResult], state: _VerdictState) -> Verdict | None:
    if any(status_value(r.status) == EngineStatus.OK.value for r in results):
        return None
    reason = "No checks ran (all engines skipped/disabled)."
    if reason not in state.reasons:
        state.reasons.append(reason)
    policy = _no_checks_policy()
    label = VerdictLabel.OK if policy == "ok" else VerdictLabel.BLOCK if policy == "block" else VerdictLabel.REVIEW
    return Verdict(label, state.nudity, state.violence, state.hate, state.reasons)


def _apply_phash(r: EngineResult, state: _VerdictState) -> Verdict | None:
    s = r.scores or {}
    if r.name == "pHash allowlist" and safe_float01(s.get("phash_allow_match", 0.0) or 0.0) >= 1.0:
        lbl = (r.details or {}).get("match_label") or (r.details or {}).get("matched_label")
        state.reasons.append("pHash allowlist match" + (f" ({lbl})" if lbl else ""))
        return Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, state.reasons)
    if r.name == "pHash blocklist" and safe_float01(s.get("phash_block_match", 0.0)) >= 1.0:
        lbl = (r.details or {}).get("match_label") or (r.details or {}).get("matched_label")
        state.reasons.append("pHash blocklist match" + (f" ({lbl})" if lbl else ""))
        return Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, state.reasons)
    return None


def _apply_ocr(r: EngineResult, state: _VerdictState) -> Verdict | None:
    if r.name == "OCR text" and safe_float01((r.scores or {}).get("ocr_match", 0.0) or 0.0) >= 1.0:
        state.reasons.append("OCR text blocked")
        return Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, state.reasons)
    return None


def _apply_nsfw(r: EngineResult, state: _VerdictState) -> None:
    s = r.scores or {}
    if r.name == "OpenNSFW2":
        n = safe_float01(s.get("nsfw_probability", 0.0))
        state.bump("nudity", n, f"OpenNSFW2 NSFW={n:.2f}", 0.50)
    if r.name == "NudeNet":
        exposed = safe_float01(s.get("nudity_exposed", 0.0))
        covered = safe_float01(s.get("nudity_covered", 0.0))
        state.bump("nudity", exposed, f"NudeNet exposed={exposed:.2f}", 0.40)
        state.bump("nudity", covered * 0.5, f"NudeNet covered={covered:.2f}", 0.60)
    if r.name.startswith("NSFWJS"):
        n = safe_float01(s.get("nsfw_combined", 0.0))
        state.bump("nudity", n, f"NSFWJS nsfw={n:.2f}", 0.50)


def _apply_forbidden_symbols(r: EngineResult, state: _VerdictState) -> None:
    if r.name != "YOLO forbidden symbols":
        return
    max_conf = safe_float01((r.scores or {}).get("forbidden_symbols_max_conf", 0.0))
    block_conf = env_float("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", 0.90)
    review_conf = env_float("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", 0.30)
    top_label = str((r.details or {}).get("top_label") or "").strip()
    label_part = f": {top_label}" if top_label else ""
    if max_conf >= block_conf:
        state.reasons.append(f"YOLO forbidden symbol detected{label_part} confidence={max_conf:.2f}")
        state.hate = max(state.hate, 1.0)
    elif max_conf >= review_conf:
        state.reasons.append(f"YOLO possible forbidden symbol{label_part} confidence={max_conf:.2f}")
        state.hate = max(state.hate, env_float("FINAL_REVIEW_THRESHOLD", 0.40))


def _apply_yolo_weapons(r: EngineResult, state: _VerdictState) -> None:
    if r.name != "YOLO-World weapons":
        return
    s = r.scores or {}
    realistic = safe_float01(s.get("yolo_firearm_realistic", 0.0))
    y_firearm_thresh = env_float("YOLO_FIREARM_THRESH", 0.35)
    if realistic >= y_firearm_thresh:
        state.reasons.append(f"YOLO firearm realistic={realistic:.2f}")
        state.violence = max(state.violence, 1.0)
    toy = safe_float01(s.get("yolo_firearm_toy", 0.0))
    any_firearm = safe_float01(s.get("yolo_firearm", 0.0))
    toy_thresh = env_float("YOLO_FIREARM_TOY_THRESH", 0.25)
    if (not env_bool("ALLOW_TOY_GUN", False)) and (toy >= toy_thresh or any_firearm >= y_firearm_thresh):
        state.reasons.append(f"YOLO firearm-like (toy/uncertain)={max(toy, any_firearm):.2f}")
        state.violence = max(state.violence, 1.0)
    danger = safe_float01(s.get("yolo_knife_dangerous", 0.0))
    if danger >= env_float("YOLO_DANGEROUS_KNIFE_THRESH", 0.35):
        state.reasons.append(f"YOLO dangerous knife={danger:.2f}")
        state.violence = max(state.violence, 1.0)
    knife = safe_float01(s.get("yolo_knife", 0.0))
    if env_bool("YOLO_KNIFE_BLOCK_ALL", False) and knife >= env_float("YOLO_KNIFE_THRESH", 0.65):
        state.reasons.append(f"YOLO knife={knife:.2f}")
        state.violence = max(state.violence, 1.0)


def _apply_sightengine(r: EngineResult, state: _VerdictState) -> None:
    if r.name != "Sightengine":
        return
    s = r.scores or {}
    raw = safe_float01(s.get("nudity_raw", 0.0))
    partial = safe_float01(s.get("nudity_partial", 0.0))
    safe = safe_float01(s.get("nudity_safe", 0.0))
    if safe > 0.0:
        partial = min(partial, max(0.0, 1.0 - safe))
    state.bump("nudity", raw, f"Sightengine raw nudity={raw:.2f}", 0.30)
    partial_risk = partial * 0.6
    state.bump("nudity", partial_risk, f"Sightengine partial nudity={partial:.2f}", 0.42)
    firearm = safe_float01(s.get("weapon_firearm", 0.0))
    firearm_toy = safe_float01(s.get("weapon_firearm_toy", 0.0))
    firearm_gesture = safe_float01(s.get("weapon_firearm_gesture", 0.0))
    firearm_animated = _safe_float(s.get("weapon_firearm_type_animated", 0.0))
    realistic_firearm = firearm * (1.0 - max(firearm_toy, firearm_gesture, firearm_animated))
    se_firearm_thresh = env_float("SE_FIREARM_THRESH", 0.35)
    if env_bool("SE_BLOCK_ANY_FIREARM", False) and firearm >= se_firearm_thresh:
        state.reasons.append(f"Sightengine firearm(any)={firearm:.2f} (toy={firearm_toy:.2f}, gesture={firearm_gesture:.2f}, animated={firearm_animated:.2f})")
        state.violence = max(state.violence, 1.0)
    if realistic_firearm >= se_firearm_thresh:
        state.reasons.append(f"Sightengine firearm: realistic={realistic_firearm:.2f} (firearm={firearm:.2f}, toy={firearm_toy:.2f}, gesture={firearm_gesture:.2f}, animated={firearm_animated:.2f})")
        state.violence = max(state.violence, 1.0)
    vio_prob = safe_float01(s.get("violence_prob", 0.0))
    vio_phys = safe_float01(s.get("violence_physical_violence", 0.0))
    vio_firearm_threat = safe_float01(s.get("violence_firearm_threat", 0.0))
    if max(vio_prob, vio_phys, vio_firearm_threat) >= env_float("SE_VIOLENCE_THRESH", 0.30):
        state.reasons.append(f"Sightengine violence: prob={vio_prob:.2f} physical={vio_phys:.2f} firearm_threat={vio_firearm_threat:.2f}")
        state.violence = max(state.violence, 1.0)
    gore_prob = safe_float01(s.get("gore_prob", 0.0))
    gore_max = max(gore_prob, _safe_float(s.get("gore_very_bloody", 0.0)), _safe_float(s.get("gore_slightly_bloody", 0.0)), _safe_float(s.get("gore_serious_injury", 0.0)), _safe_float(s.get("gore_superficial_injury", 0.0)), _safe_float(s.get("gore_corpse", 0.0)), _safe_float(s.get("gore_body_organ", 0.0)))
    if gore_max >= env_float("SE_GORE_THRESH", 0.20):
        state.reasons.append(f"Sightengine gore/blood: score={gore_max:.2f} (prob={gore_prob:.2f})")
        state.violence = max(state.violence, 1.0)
    offensive_max = safe_float01(s.get("offensive_max", 0.0))
    if offensive_max >= env_float("SE_OFFENSIVE_THRESH", 0.50):
        state.reasons.append(f"Sightengine offensive symbols: score={offensive_max:.2f}")
        state.hate = max(state.hate, 1.0)
    knife = safe_float01(s.get("weapon_knife", 0.0))
    knife_ctx = max(vio_prob, vio_phys, vio_firearm_threat, gore_max)
    if knife >= env_float("SE_KNIFE_THRESH", 0.65) and (env_bool("SE_KNIFE_BLOCK_ALL", False) or knife_ctx >= env_float("SE_KNIFE_CONTEXT_THRESH", 0.25)):
        state.reasons.append(f"Sightengine knife: score={knife:.2f} ctx={knife_ctx:.2f}")
        state.violence = max(state.violence, 1.0)


def _apply_openai(r: EngineResult, state: _VerdictState) -> Verdict | None:
    if r.name != "OpenAI Moderation":
        return None
    s = r.scores or {}
    minors = safe_float01(s.get("sexual/minors", 0.0))
    sexual = safe_float01(s.get("sexual", 0.0))
    violence = max(safe_float01(s.get("violence", 0.0)), safe_float01(s.get("violence/graphic", 0.0)))
    hate = max(safe_float01(s.get("hate", 0.0)), safe_float01(s.get("hate/threatening", 0.0)))
    if minors > 0.01:
        state.reasons.append("OpenAI: sexual/minors detected")
        return Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, state.reasons)
    state.bump("nudity", sexual, f"OpenAI sexual={sexual:.2f}", 0.50)
    state.bump("violence", violence, f"OpenAI violence={violence:.2f}", 0.50)
    state.bump("hate", hate, f"OpenAI hate={hate:.2f}", 0.50)
    return None


def _final_label(state: _VerdictState) -> VerdictLabel:
    block_t = get_config().final_block_threshold
    review_t = env_float("FINAL_REVIEW_THRESHOLD", 0.40)
    if state.nudity >= block_t or state.violence >= block_t or state.hate >= block_t:
        return VerdictLabel.BLOCK
    if state.nudity >= review_t or state.violence >= review_t or state.hate >= review_t:
        return VerdictLabel.REVIEW
    return VerdictLabel.OK


def compute_verdict(results: List[EngineResult]) -> Verdict:
    """Compute final OK / REVIEW / BLOCK verdict from engine outputs."""
    state = _VerdictState(reasons=[])
    early = _apply_error_policy(results, state) or _apply_no_checks(results, state)
    if early is not None:
        return early
    for r in results:
        if status_value(r.status) != EngineStatus.OK.value:
            continue
        for fn in (_apply_phash, _apply_ocr, _apply_openai):
            early = fn(r, state)
            if early is not None:
                return early
        _apply_nsfw(r, state)
        _apply_forbidden_symbols(r, state)
        _apply_yolo_weapons(r, state)
        _apply_sightengine(r, state)
    label = _final_label(state)
    if label != VerdictLabel.OK and not state.reasons:
        state.reasons.append("Borderline content detected by one or more engines.")
    return Verdict(label, state.nudity, state.violence, state.hate, state.reasons)

# -----------------------------
# Runner
# -----------------------------

def pick_file_dialog() -> Optional[str]:
    """Open a file picker if Tkinter is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilename(
                title="Select an image/GIF",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.gif *.bmp *.tif *.tiff"), ("All files", "*.*")],
            )
            return path or None
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        return None

def pick_folder_dialog() -> Optional[str]:
    """Open a folder picker if Tkinter is available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        root.destroy()
        if folder:
            return str(folder)
        return None
    except Exception:
        return None
