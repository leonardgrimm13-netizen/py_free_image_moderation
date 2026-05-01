from pathlib import Path
import re


REQUIRED_KEYS = {
    "MODIMG_LOG_LEVEL",
    "MODIMG_DEBUG",
    "DEBUG",
    "DOTENV_OVERRIDE",
    "TF_CPP_MIN_LOG_LEVEL",
    "MODIMG_PARALLEL_ENGINES",
    "MODIMG_PARALLEL_WORKERS",
    "SAMPLE_FRAMES",
    "SCORE_VERBOSE",
    "SCORE_MAX_KEYS",
    "SIGHTENGINE_SCORE_MODE",
    "SIGHTENGINE_SCORE_KEYS",
    "SIGHTENGINE_EXTRA_TOPK",
    "API_POLICY",
    "PHASH_ALLOWLIST",
    "PHASH_BLOCKLIST",
    "PHASH_ALLOW_DISABLE",
    "PHASH_BLOCK_DISABLE",
    "PHASH_ALLOW_MAX_DISTANCE",
    "PHASH_BLOCK_MAX_DISTANCE",
    "SHORT_CIRCUIT_PHASH",
    "PHASH_AUTO_LEARN_ENABLE",
    "PHASH_AUTO_APPEND",
    "PHASH_AUTO_ALLOW_APPEND",
    "PHASH_AUTO_BLOCK_APPEND",
    "PHASH_AUTO_LABEL",
    "PHASH_AUTO_ALLOW_LABEL",
    "PHASH_AUTO_BLOCK_LABEL",
    "PHASH_GIF_LEARN_FIRST_LAST",
    "OCR_ENABLE",
    "OCR_LANG",
    "OCR_MAX_FRAMES",
    "OCR_MIN_LEN",
    "TESSERACT_CMD",
    "OPENNSFW2_DISABLE",
    "NUDENET_DISABLE",
    "YOLO_BACKEND",
    "YOLO_WORLD_MODEL",
    "YOLO_WEAPON_MODEL",
    "YOLO_WEAPONS_WEIGHTS",
    "YOLO_CONF",
    "YOLO_IOU",
    "YOLO_IMGSZ",
    "YOLO_MAX_DET",
    "YOLO_MAX_FRAMES",
    "YOLO_DEVICE",
    "YOLO_FIREARM_THRESH",
    "YOLO_FIREARM_TOY_THRESH",
    "ALLOW_TOY_GUN",
    "YOLO_DANGEROUS_KNIFE_THRESH",
    "YOLO_KNIFE_THRESH",
    "YOLO_KNIFE_BLOCK_ALL",
    "OPENAI_DISABLE",
    "OPENAI_API_KEY",
    "OPENAI_MODERATION_MODEL",
    "OPENAI_REQUEST_TIMEOUT_SEC",
    "OPENAI_MIN_INTERVAL_SEC",
    "OPENAI_MAX_RETRIES",
    "OPENAI_BACKOFF_BASE_SEC",
    "OPENAI_BACKOFF_MAX_SEC",
    "OPENAI_MAX_TOTAL_SLEEP_SEC",
    "OPENAI_429_POLICY",
    "OPENAI_MAX_429_RETRIES",
    "OPENAI_CACHE_ENABLE",
    "OPENAI_CACHE_PATH",
    "OPENAI_CACHE_MAX_ITEMS",
    "SIGHTENGINE_DISABLE",
    "SIGHTENGINE_USER",
    "SIGHTENGINE_SECRET",
    "SIGHTENGINE_MODELS",
    "SE_FIREARM_THRESH",
    "SE_BLOCK_ANY_FIREARM",
    "SE_VIOLENCE_THRESH",
    "SE_GORE_THRESH",
    "SE_OFFENSIVE_THRESH",
    "SE_KNIFE_THRESH",
    "SE_KNIFE_BLOCK_ALL",
    "SE_KNIFE_CONTEXT_THRESH",
    "CORE_ENGINES",
    "ENGINE_ERROR_POLICY",
    "NO_CHECKS_POLICY",
    "FINAL_BLOCK_THRESHOLD",
    "FINAL_REVIEW_THRESHOLD",
}


def _active_env_keys(text: str) -> set[str]:
    keys = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", stripped)
        if m:
            keys.add(m.group(1))
    return keys


def test_env_example_has_all_required_active_keys() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    active_keys = _active_env_keys(text)
    missing = sorted(REQUIRED_KEYS - active_keys)
    assert not missing, f"Missing active keys in .env.example: {missing}"


def test_env_example_contains_legacy_phash_alias_comments() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    for alias in ("PHASH_ALLOW_MAXDIST", "PHASH_BLOCK_MAXDIST", "PHASH_MAXDIST"):
        assert alias in text
