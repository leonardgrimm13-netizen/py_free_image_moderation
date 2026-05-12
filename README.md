# py_free_image_moderation
A flexible Python project for **image and GIF moderation** with multiple engines (local + API), pHash lists, and clear CLI output.

**Languages:** **English** | [German](README.de.md)

## Contents
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Verification](#verification)
- [Important configuration (.env)](#important-configuration-env)
- [Result logic (OK / REVIEW / BLOCK)](#result-logic-ok--review--block)
- [Tips for running](#tips-for-running)

---

## ✨ Features
- **Multi-stage moderation** for single images, GIFs, directories, and URLs
- **pHash allowlist/blocklist** for very fast short-circuit decisions; pHash auto-learning is off by default to avoid learning false positives
- **OCR text check** (e.g., against text blocklists)
- Combinable engines:
  - `OpenNSFW2`
  - `NudeNet`
  - `YOLO` (weapon detection)
  - `YOLO forbidden symbols` (local forbidden/harmful-symbol detection using `models/forbidden_symbols_yolo.pt`)
  - `OpenAI Moderation` (optional via API key)
  - `Sightengine` (optional via API credentials)
- **GIF handling** with configurable frame sampling
- **JSON export** for further processing in pipelines
- **Conservative verdict logic** with clear, traceable reasons

---

## 📁 Project structure
```text
py_free_image_moderation/
├── moderate_image.py         # Entry point (CLI wrapper)
├── requirements.txt
├── requirements_api.txt
├── models/
│   └── forbidden_symbols_yolo.pt  # bundled local YOLO model for forbidden-symbol detection
├── data/
│   ├── phash_allowlist.txt
│   ├── phash_blocklist.txt
│   └── ocr_text_blocklist.txt
└── modimg/
    ├── cli.py               # Args, output, JSON export
    ├── pipeline.py          # Flow & engine orchestration
    ├── verdict.py           # Final decision logic
    ├── frames.py            # Image/GIF frame loading
    ├── phash.py             # pHash utilities
    ├── config.py            # .env loading
    └── engines/             # Individual moderation engines
```

---

## ⚙️ Installation
> Recommended: Python **3.11+** in a virtual environment.

### 1) Repository and venv
```bash
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

python3 -m venv .venv
source .venv/bin/activate
```

Windows activation, if needed: `.venv\Scripts\activate`.

### 2) Install options

#### A) Offline/Local
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Includes the local runtime + engine dependencies (without API clients):
- `Pillow`
- `numpy`
- `ImageHash`
- `opennsfw2`
- `nudenet`
- `ultralytics`
- `pytesseract`

This enables the local pipeline including pHash, local YOLO forbidden-symbol detection, and `--no-apis`.

#### B) With APIs
```bash
python3 -m pip install -r requirements_api.txt
```

Includes everything from `requirements.txt` plus API clients:
- `openai` (OpenAI moderation)
- `requests` (HTTP client used by Sightengine)
- `sightengine` (Sightengine API package, kept for compatibility)

Editable installs use the same split via extras:
```bash
python3 -m pip install -e ".[dev]"      # tests/linting only
python3 -m pip install -e ".[local]"    # local vision engines
python3 -m pip install -e ".[api]"      # API engines
python3 -m pip install -e ".[all]"      # local vision + API engines
```

### 3) Dev/Test dependencies
```bash
python3 -m pip install -r requirements_dev.txt
```

Includes e.g. `pytest` for local test runs.

### 4) Bundled local YOLO model
This repository includes `models/forbidden_symbols_yolo.pt` directly as a normal repository file.

The model is loaded locally by the `YOLO forbidden symbols` engine. It never calls Roboflow or any external API at runtime. If the file is missing, set `FORBIDDEN_SYMBOLS_YOLO_MODEL` to an absolute path or run from the project root. If Git-LFS left a pointer file instead of the real weights, run `git lfs pull`.

The separate `YOLO-World weapons` engine is optional. By default it looks for `.cache/ultralytics/weights/yolov8s-oiv7.pt` and reports `SKIPPED` when no model exists. To use your own weapon weights, set either `YOLO_WEAPON_MODEL=/absolute/or/project/relative/model.pt` or `YOLO_WORLD_MODEL=/absolute/or/project/relative/model.pt`.

### 5) Optional system dependency for OCR
For OCR you typically need a local Tesseract install:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (Homebrew): `brew install tesseract`

---

## 🚀 Quickstart

### Check a single image
```bash
python3 moderate_image.py /path/to/image.jpg
```

### Check a GIF (frame sampling)
```bash
python3 moderate_image.py /path/to/file.gif --sample-frames 12
```

### Check a URL
```bash
python3 moderate_image.py "https://example.com/image.jpg"
```

### Check a directory
```bash
python3 moderate_image.py ./images --recursive
```

### Without external APIs (base install is enough)
```bash
python3 moderate_image.py ./images --recursive --no-apis
```

Local YOLO forbidden-symbol output is shown with compact numeric scores, for example:
```text
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.72, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.93, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=1.00
```

### Write a JSON report
```bash
python3 moderate_image.py ./images --recursive --json moderation_report.json
```

### Benchmark mode
Benchmark mode measures runtime per file and per engine without changing moderation decisions.

```bash
python3 moderate_image.py ./images --recursive --no-apis --benchmark
python3 moderate_image.py ./images --recursive --no-apis --benchmark-json benchmark.json
python3 moderate_image.py ./images --recursive --no-apis --json moderation_report.json --benchmark-json benchmark.json
```

Benchmark JSON field `total_wall_ms` includes only wall-clock time spent processing inputs (not time spent writing JSON output files).

**Exit codes:**
- `0` = all results are `OK`
- `2` = at least one result is not `OK`

---

## ✅ Verification
```bash
python3 -m compileall -q .
pytest -q
python3 moderate_image.py --help
python3 moderate_image.py path/to/test.png --no-apis
```

Expected behavior (short):
- `python3 -m compileall -q .` → exit code `0` if code is syntactically valid.
- `pytest -q` → exit code `0` if tests pass, otherwise non-zero.
- `python3 moderate_image.py --help` → exit code `0` and shows CLI help.
- `python3 moderate_image.py path/to/test.png --no-apis` → exit code `0` if the input is `OK`, or `2` if it returns `REVIEW`/`BLOCK`.

Optional engines may be missing; they must show up as `skipped`/`disabled` in output instead of aborting execution.

---

## 🔧 Important configuration (.env)
The project automatically loads `.env` from the project root. Example:

The loader checks `.env`, then `.env.txt`, and finally `.env.example` as fallback defaults. For best results, copy `.env.example` to `.env` and edit `.env` for your environment.

```env
# API engines
OPENAI_API_KEY=...
SIGHTENGINE_USER=...
SIGHTENGINE_SECRET=...

# Global
SAMPLE_FRAMES=12
SHORT_CIRCUIT_PHASH=1
ENGINE_ERROR_POLICY=review

# OCR
OCR_ENABLE=1
OCR_LANG=eng

# pHash auto-learn
PHASH_AUTO_LEARN_ENABLE=0
PHASH_AUTO_ALLOW_APPEND=0
PHASH_AUTO_BLOCK_APPEND=0

# Optional YOLO-World weapons model
# If unset and .cache/ultralytics/weights/yolov8s-oiv7.pt is missing, the engine is skipped.
YOLO_WEAPON_MODEL=
YOLO_WORLD_MODEL=
YOLO_CONF=0.25
YOLO_IMGSZ=640
YOLO_MAX_FRAMES=2
YOLO_DEVICE=

# Local YOLO forbidden/harmful-symbol model
FORBIDDEN_SYMBOLS_YOLO_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_MODEL=models/forbidden_symbols_yolo.pt
FORBIDDEN_SYMBOLS_YOLO_CONF=0.20
FORBIDDEN_SYMBOLS_YOLO_IOU=0.45
FORBIDDEN_SYMBOLS_YOLO_IMGSZ=960
FORBIDDEN_SYMBOLS_YOLO_MAX_DET=20
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=2
FORBIDDEN_SYMBOLS_YOLO_DEVICE=auto
FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30
FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90
FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF=isis:0.75,swastika:0.50
FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF=isis:0.92,swastika:0.90
FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES=1
FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=
```

Useful toggles:
- Main performance knobs: `SAMPLE_FRAMES`, `API_POLICY`, `YOLO_IMGSZ`, `YOLO_MAX_FRAMES`, `YOLO_MAX_DET`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES`, `OCR_MAX_FRAMES`, `PHASH_ALLOW_MAX_DISTANCE`, `PHASH_BLOCK_MAX_DISTANCE`
- `API_POLICY=always|on_review|never` controls when API engines run
- `OPENAI_DISABLE=1` / omit `SIGHTENGINE_*` if you don’t use API engines
- `PHASH_ALLOW_DISABLE=1` or `PHASH_BLOCK_DISABLE=1` to disable them selectively
- `SCORE_VERBOSE=1` for more verbose engine scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` for centralized logging
- `MODIMG_PARALLEL_ENGINES=1` to run independent engines concurrently (optional/experimental; disabled by default)
- `NO_CHECKS_POLICY=review` controls the fallback when no engine ran: `ok` = allow, `review` = safer default, `block` = strictest mode
- `YOLO_WEAPON_MODEL` or `YOLO_WORLD_MODEL` points to custom YOLO weapon weights; without weights the weapon engine is skipped, not failed.


### Local YOLO forbidden-symbol configuration
- `FORBIDDEN_SYMBOLS_YOLO_ENABLE=1` enables the bundled local model by default.
- `FORBIDDEN_SYMBOLS_YOLO_CONF=0.20` controls the raw YOLO detection confidence.
- `FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30` controls when detections should push the verdict to `REVIEW`.
- `FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90` controls when detections should push the verdict to `BLOCK`.
- Label-specific thresholds can override the global verdict thresholds for named labels only, e.g. `FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF=isis:0.75,swastika:0.50` and `FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF=isis:0.92,swastika:0.90`. Labels not listed there keep using the global review/block thresholds.
- Label-specific thresholds are safer than `FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=isis` for noisy classes: low-confidence ISIS false positives can be suppressed while genuine high-confidence ISIS detections still reach `REVIEW`/`BLOCK`.
- Recommended defaults: `conf=0.20`, `review=0.30`, `block=0.90`, `imgsz=960`.
- `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0` disables frame inference for this engine and returns an OK result with zero detections.
- For faster CPU-only scans, try `SAMPLE_FRAMES=3`, `OCR_MAX_FRAMES=1`, `YOLO_IMGSZ=416`, `YOLO_MAX_FRAMES=1`, `YOLO_DEVICE=cpu`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ=640`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=1`, and `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`.
- Unreliable labels can be ignored at runtime, e.g. `FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=communism,antifa`.

---

## 🧠 Result logic (OK / REVIEW / BLOCK)
- **Staged pipeline:** `pHash` → local engines → optional API engines → final verdict
- **pHash short-circuit** can decide early:
  - allowlist hit → `OK`
  - blocklist hit → `BLOCK`
- If pHash does not short-circuit, the remaining local engines run, including `YOLO forbidden symbols`.
- The forbidden-symbol YOLO engine contributes to hate/policy risk: at or above the block threshold it should result in `BLOCK`; at or above the review threshold it should result in `REVIEW`.
- Detection labels and boxes are written to JSON under engine `details.detections`.
- `verdict.py` condenses signals (nudity, violence, hate) into the final decision
- Error behavior can be controlled via `ENGINE_ERROR_POLICY` (`ignore`, `review`, `block`)

---

## 🛠️ Tips for running
- Start with `--no-apis` to verify the local pipeline and performance first.
- Use `--json` if results should be processed in CI/CD or backend services.
- Maintain `data/phash_allowlist.txt` and `data/phash_blocklist.txt` regularly for stable decisions on recurring content.
- For GIFs, increase `--sample-frames` if problematic content appears only in a few frames.
