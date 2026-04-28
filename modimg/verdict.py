from __future__ import annotations

import os
from typing import List, Optional

from .types import EngineResult, Verdict
from .utils import safe_float01


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def compute_verdict(results: List[EngineResult]) -> Verdict:
    """
    Conservative heuristic:
      - BLOCK if strong nudity/porn or graphic violence/hate
      - REVIEW for borderline (racy, mild violence)
      - OK otherwise
    """
    reasons: List[str] = []
    nudity = 0.0
    violence = 0.0
    hate = 0.0

    sf = safe_float01

    # If any *core* engine crashed, be conservative.
    # ENGINE_ERROR_POLICY: ignore | review | block
    # Allow users to specify core engines either by display name or by a short alias.
    _ALIASES = {
        "phash_allowlist": "pHash allowlist",
        "phash_blocklist": "pHash blocklist",
        "phash_allow": "pHash allowlist",
        "phash_block": "pHash blocklist",
        "ocr": "OCR text",
        "openai": "OpenAI Moderation",
        "sightengine": "Sightengine",
    }

    core_env = (os.getenv("CORE_ENGINES", "") or "").strip()
    if core_env:
        raw = [c.strip() for c in core_env.split(",") if c.strip()]
        core_set = set(_ALIASES.get(c.lower(), c) for c in raw)
    else:
        # Sensible defaults: treat only the *policy/guardrail* engines as core.
        # Offline heuristics may be optional on some machines and should not
        # force REVIEW just because a dependency is missing.
        core_set = {
            "pHash allowlist",
            "pHash blocklist",
            "OCR text",
            "OpenAI Moderation",
            "Sightengine",
        }

    err_all = [r for r in results if (r.status or "").lower() == "error"]
    err_core = [r for r in err_all if (not core_set) or (r.name in core_set)]

    if err_core:
        names = ", ".join([r.name for r in err_core[:6]])
        reasons.append(f"Some checks failed: {names}")
        policy = (os.getenv("ENGINE_ERROR_POLICY", "review") or "review").strip().lower()
        # Back-compat naming
        if policy in ("lenient", "loose"):
            policy = "ignore"
        elif policy in ("strict", "hard"):
            policy = "review"
        if policy in ("block", "not_ok", "fail", "fail_closed", "deny"):
            return Verdict("BLOCK", max(nudity, 0.5), max(violence, 0.5), max(hate, 0.5), reasons)
        if policy in ("ignore", "open", "allow"):
            # proceed without forcing REVIEW/BLOCK
            pass
        else:
            # default: REVIEW
            nudity = max(nudity, 0.40)
            violence = max(violence, 0.40)
            hate = max(hate, 0.40)

    if err_all and not err_core:
        names = ", ".join([r.name for r in err_all[:6]])
        reasons.append(f"Non-core checks failed (ignored): {names}")

    # If nothing produced scores (all checks were skipped/disabled), treat as OK by default.
    # This avoids false NOT_OK verdicts when the user hasn't installed optional dependencies
    # or has no API keys configured.
    if not any((r.status or "").lower() == "ok" for r in results):
        if not reasons:
            reasons.append("No checks ran (all engines skipped/disabled).")
        return Verdict("OK", nudity, violence, hate, reasons)


    # Helpers
    def bump(current: float, value: float, reason: str, thresh: float) -> float:
        nonlocal reasons
        if value >= thresh:
            reasons.append(reason)
        return max(current, value)

    # Aggregate
    for r in results:
        if r.status != "ok":
            continue
        s = r.scores or {}
        if r.name == "pHash allowlist":
            try:
                match = sf((r.scores or {}).get("phash_allow_match", 0.0) or 0.0)
            except Exception:
                match = 0.0
            if match >= 1.0:
                lbl = None
                try:
                    lbl = (r.details or {}).get("match_label") or (r.details or {}).get("matched_label")
                except Exception:
                    lbl = None
                reasons.append("pHash allowlist match" + (f" ({lbl})" if lbl else ""))
                return Verdict("OK", 0.0, 0.0, 0.0, reasons)

        if r.name == "pHash blocklist":
            match = sf(s.get("phash_block_match", 0.0))
            if match >= 1.0:
                lbl = None
                try:
                    lbl = (r.details or {}).get("match_label") or (r.details or {}).get("matched_label")
                except Exception:
                    lbl = None
                reasons.append("pHash blocklist match" + (f" ({lbl})" if lbl else ""))
                return Verdict("BLOCK", 1.0, 1.0, 1.0, reasons)

        n = 0.0
        v = 0.0
        h = 0.0

        if r.name == "OCR text":
            try:
                flagged = sf((r.scores or {}).get("ocr_match", 0.0) or 0.0)
            except Exception:
                flagged = 0.0
            if flagged >= 1.0:
                reasons.append("OCR text blocked")
                return Verdict("BLOCK", 1.0, 1.0, 1.0, reasons)

        if r.name == "OpenNSFW2":
            n = sf(s.get("nsfw_probability", 0.0))
            nudity = bump(nudity, n, f"OpenNSFW2 NSFW={n:.2f}", 0.50)

        if r.name == "NudeNet":
            exposed = sf(s.get("nudity_exposed", 0.0))
            covered = sf(s.get("nudity_covered", 0.0))
            # exposed strong, covered mild
            nudity = bump(nudity, exposed, f"NudeNet exposed={exposed:.2f}", 0.40)
            nudity = bump(nudity, covered * 0.5, f"NudeNet covered={covered:.2f}", 0.60)

        if r.name.startswith("NSFWJS"):
            n = sf(s.get("nsfw_combined", 0.0))
            nudity = bump(nudity, n, f"NSFWJS nsfw={n:.2f}", 0.50)


        if r.name == "YOLO-World weapons":
            realistic = sf(s.get("yolo_firearm_realistic", 0.0))
            y_firearm_thresh = _env_float("YOLO_FIREARM_THRESH", 0.35)
            if realistic >= y_firearm_thresh:
                reasons.append(f"YOLO firearm realistic={realistic:.2f}")
                violence = max(violence, 1.0)

            # Cutouts/renders often get classified as 'toy'. By default we still treat firearm-like as NOT_OK unless ALLOW_TOY_GUN=1
            toy = sf(s.get("yolo_firearm_toy", 0.0))
            any_firearm = sf(s.get("yolo_firearm", 0.0))
            toy_thresh = _env_float("YOLO_FIREARM_TOY_THRESH", 0.25)
            allow_toy = os.getenv("ALLOW_TOY_GUN", "0").strip().lower() in ("1", "true", "yes", "on")
            if (not allow_toy) and (toy >= toy_thresh or any_firearm >= y_firearm_thresh):
                reasons.append(f"YOLO firearm-like (toy/uncertain)={max(toy, any_firearm):.2f}")
                violence = max(violence, 1.0)

            danger = sf(s.get("yolo_knife_dangerous", 0.0))
            y_dknife_thresh = _env_float("YOLO_DANGEROUS_KNIFE_THRESH", 0.35)
            if danger >= y_dknife_thresh:
                reasons.append(f"YOLO dangerous knife={danger:.2f}")
                violence = max(violence, 1.0)

            knife = sf(s.get("yolo_knife", 0.0))
            y_knife_thresh = _env_float("YOLO_KNIFE_THRESH", 0.65)
            knife_block_all = os.getenv("YOLO_KNIFE_BLOCK_ALL", "0").strip().lower() in ("1", "true", "yes", "on")
            if knife_block_all and knife >= y_knife_thresh:
                reasons.append(f"YOLO knife={knife:.2f}")
                violence = max(violence, 1.0)

        if r.name == "Sightengine":
            raw = sf(s.get("nudity_raw", 0.0))
            partial = sf(s.get("nudity_partial", 0.0))
            safe = sf(s.get("nudity_safe", 0.0))
            # If the API reports a high 'safe/none' probability, cap partial nudity accordingly.
            if safe > 0.0:
                partial = min(partial, max(0.0, 1.0 - safe))
            nudity = bump(nudity, raw, f"Sightengine raw nudity={raw:.2f}", 0.30)
            partial_risk = partial * 0.6
            # Threshold must be consistent with the weighted risk value (0.70*0.6 = 0.42)
            nudity = bump(nudity, partial_risk, f"Sightengine partial nudity={partial:.2f}", 0.42)

            # --- Extra Sightengine policies (weapons/violence/gore/offensive) ---
            firearm = sf(s.get("weapon_firearm", 0.0))
            firearm_toy = sf(s.get("weapon_firearm_toy", 0.0))
            firearm_gesture = sf(s.get("weapon_firearm_gesture", 0.0))
            firearm_animated = _safe_float(s.get("weapon_firearm_type_animated", 0.0))
            realistic_firearm = firearm * (1.0 - max(firearm_toy, firearm_gesture, firearm_animated))
            se_firearm_thresh = _env_float("SE_FIREARM_THRESH", 0.35)
            block_any_firearm = os.getenv("SE_BLOCK_ANY_FIREARM", "0").strip().lower() in ("1","true","yes","on")
            if block_any_firearm and firearm >= se_firearm_thresh:
                reasons.append(f"Sightengine firearm(any)={firearm:.2f} (toy={firearm_toy:.2f}, gesture={firearm_gesture:.2f}, animated={firearm_animated:.2f})")
                violence = max(violence, 1.0)
            if realistic_firearm >= se_firearm_thresh:
                reasons.append(
                    f"Sightengine firearm: realistic={realistic_firearm:.2f} (firearm={firearm:.2f}, toy={firearm_toy:.2f}, gesture={firearm_gesture:.2f}, animated={firearm_animated:.2f})"
                )
                violence = max(violence, 1.0)

            vio_prob = sf(s.get("violence_prob", 0.0))
            vio_phys = sf(s.get("violence_physical_violence", 0.0))
            vio_firearm_threat = sf(s.get("violence_firearm_threat", 0.0))
            se_violence_thresh = _env_float("SE_VIOLENCE_THRESH", 0.30)
            if max(vio_prob, vio_phys, vio_firearm_threat) >= se_violence_thresh:
                reasons.append(
                    f"Sightengine violence: prob={vio_prob:.2f} physical={vio_phys:.2f} firearm_threat={vio_firearm_threat:.2f}"
                )
                violence = max(violence, 1.0)

            gore_prob = sf(s.get("gore_prob", 0.0))
            gore_max = max(
                gore_prob,
                _safe_float(s.get("gore_very_bloody", 0.0)),
                _safe_float(s.get("gore_slightly_bloody", 0.0)),
                _safe_float(s.get("gore_serious_injury", 0.0)),
                _safe_float(s.get("gore_superficial_injury", 0.0)),
                _safe_float(s.get("gore_corpse", 0.0)),
                _safe_float(s.get("gore_body_organ", 0.0)),
            )
            se_gore_thresh = _env_float("SE_GORE_THRESH", 0.20)
            if gore_max >= se_gore_thresh:
                reasons.append(f"Sightengine gore/blood: score={gore_max:.2f} (prob={gore_prob:.2f})")
                violence = max(violence, 1.0)

            offensive_max = sf(s.get("offensive_max", 0.0))
            se_offensive_thresh = _env_float("SE_OFFENSIVE_THRESH", 0.50)
            if offensive_max >= se_offensive_thresh:
                reasons.append(f"Sightengine offensive symbols: score={offensive_max:.2f}")
                hate = max(hate, 1.0)

            knife = sf(s.get("weapon_knife", 0.0))
            se_knife_thresh = _env_float("SE_KNIFE_THRESH", 0.65)
            knife_block_all = os.getenv("SE_KNIFE_BLOCK_ALL", "0").strip().lower() in ("1", "true", "yes", "on")
            knife_ctx = max(vio_prob, vio_phys, vio_firearm_threat, gore_max)
            knife_ctx_thresh = _env_float("SE_KNIFE_CONTEXT_THRESH", 0.25)
            if knife >= se_knife_thresh and (knife_block_all or knife_ctx >= knife_ctx_thresh):
                reasons.append(f"Sightengine knife: score={knife:.2f} ctx={knife_ctx:.2f}")
                violence = max(violence, 1.0)


        if r.name == "OpenAI Moderation":
            # Sexual & violence/hate categories
            minors = sf(s.get("sexual/minors", 0.0))
            sexual = sf(s.get("sexual", 0.0))
            v = max(sf(s.get("violence", 0.0)), sf(s.get("violence/graphic", 0.0)))
            h = max(sf(s.get("hate", 0.0)), sf(s.get("hate/threatening", 0.0)))

            if minors > 0.01:
                reasons.append("OpenAI: sexual/minors detected")
                return Verdict("BLOCK", 1.0, 1.0, 1.0, reasons)

            nudity = bump(nudity, sexual, f"OpenAI sexual={sexual:.2f}", 0.50)
            violence = bump(violence, v, f"OpenAI violence={v:.2f}", 0.50)
            hate = bump(hate, h, f"OpenAI hate={h:.2f}", 0.50)

    # Final decision thresholds (configurable via .env)
    block_t = _env_float("FINAL_BLOCK_THRESHOLD", 0.85)
    review_t = _env_float("FINAL_REVIEW_THRESHOLD", 0.40)
    label = "OK"
    if nudity >= block_t or violence >= block_t or hate >= block_t:
        label = "BLOCK"
    elif nudity >= review_t or violence >= review_t or hate >= review_t:
        label = "REVIEW"

    # Ensure at least one reason for REVIEW/BLOCK
    if label != "OK" and not reasons:
        reasons.append("Borderline content detected by one or more engines.")

    return Verdict(label, nudity, violence, hate, reasons)

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

