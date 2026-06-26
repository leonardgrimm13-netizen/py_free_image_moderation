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
  - `YOLO weapon detection` (local YOLO-detection for weapon detection using `models/weapon_detection_yolo.pt`)
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
├── requirements.txt        # core runtime
├── requirements_local.txt  # local vision/OCR engines
├── requirements_api.txt    # API engines
├── requirements_all.txt    # local + API runtime
├── requirements_dev.txt    # tests/lint/build tools
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
> Recommended and supported for this project: Python **3.11 or 3.12** in a virtual environment.
>
> `pyproject.toml` declares `>=3.11,<3.13`. Python 3.13+ is not claimed as supported unless you test it yourself.

### 1) Repository and venv
Linux/macOS:
```bash
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

python3.12 -m venv .venv  # or: python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:
```powershell
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

py -3.12 -m venv .venv  # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install options

#### A) Base/Core
```bash
python -m pip install -r requirements.txt
```

Includes only the lightweight core dependencies:
- `Pillow`
- `numpy`
- `ImageHash`

This is enough for image loading, GIF frame sampling, pHash allow/block lists, JSON output, and graceful skipping of optional engines.

#### B) Local/Vision
```bash
python -m pip install -r requirements_local.txt
```

Includes the base runtime plus local vision/OCR engine dependencies:
- `opennsfw2[tf-keras]`
- `nudenet`
- `ultralytics`
- `pytesseract`

This enables the local pipeline including OpenNSFW2, NudeNet, YOLO weapons, local YOLO forbidden-symbol detection, OCR Python bindings, and `--no-apis`.

OpenNSFW2 is installed with the `tf-keras` extra on purpose. OpenNSFW2 needs a backend, and this project uses the TensorFlow/tf-keras path as the stable default for Python 3.11/3.12. The plain `opennsfw2` package is not enough for reliable local inference.

#### C) API engines
```bash
python -m pip install -r requirements_api.txt
```

Includes the base runtime plus API clients:
- `openai` (OpenAI moderation)
- `requests` (HTTP client used by the Sightengine engine)

The code calls Sightengine through direct HTTP requests; it does not import a separate `sightengine` SDK.

#### D) All runtime engines
```bash
python -m pip install -r requirements_all.txt
```

Editable installs use the same split via extras:
```bash
python -m pip install -e ".[dev]"      # tests/linting only
python -m pip install -e ".[local]"    # local vision engines
python -m pip install -e ".[api]"      # API engines
python -m pip install -e ".[all]"      # local vision + API engines
```

### 3) Dev/Test dependencies
```bash
python -m pip install -r requirements_dev.txt
```

Includes the base runtime plus `pytest`, `pytest-cov`, `ruff`, and `build`.

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
python moderate_image.py /path/to/image.jpg
```

### Check a GIF (frame sampling)
```bash
python moderate_image.py /path/to/file.gif --sample-frames 12
```

### Check a URL
```bash
python moderate_image.py "https://example.com/image.jpg"
```

### Check a directory
```bash
python moderate_image.py ./images --recursive
```

### Check a directory with file parallelism
```bash
python moderate_image.py ./images --recursive --file-workers 2
```

### Without external APIs
```bash
python moderate_image.py ./images --recursive --no-apis
```

With only the base install, optional local vision engines report `skipped` instead of crashing. Install `requirements_local.txt` or `.[local]` when you want local inference.

Local YOLO forbidden-symbol output is shown with compact numeric scores, for example:
```text
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.72, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.93, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=1.00
```

### Write a JSON report
```bash
python moderate_image.py ./images --recursive --json moderation_report.json
```

### Benchmark mode
Benchmark mode measures runtime per file and per engine without changing moderation decisions.

```bash
python moderate_image.py ./images --recursive --no-apis --benchmark
python moderate_image.py ./images --recursive --no-apis --benchmark-json benchmark.json
python moderate_image.py ./images --recursive --no-apis --json moderation_report.json --benchmark-json benchmark.json
```

Benchmark JSON field `total_wall_ms` includes only wall-clock time spent processing inputs (not time spent writing JSON output files).

**Exit codes:**
- `0` = all results are `OK`
- `2` = at least one result is not `OK`

---

## ✅ Verification
Core install:
```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python moderate_image.py --help
python moderate_image.py path/to/test.png --no-apis
python -m pip check
```

Dev/test install:
```bash
python -m pip install -r requirements_dev.txt
pytest -q
```

Local/Vision install smoke test:
```bash
python -m pip install -r requirements_local.txt
python -c "import opennsfw2, nudenet, ultralytics, pytesseract"
python -m pip check
```

Expected behavior (short):
- `python -m compileall -q .` → exit code `0` if code is syntactically valid.
- `pytest -q` → exit code `0` if tests pass, otherwise non-zero.
- `python moderate_image.py --help` → exit code `0` and shows CLI help.
- `python moderate_image.py path/to/test.png --no-apis` → exit code `0` if the input is `OK`, or `2` if it returns `REVIEW`/`BLOCK`.

Optional engines may be missing; they must show up as `skipped`/`disabled` in output instead of aborting execution.

---

## 🔧 Important configuration (.env)
The project automatically loads `.env` from the project root. Example:

The loader checks `.env`, then `.env.txt`. It does not load `.env.example` automatically, because that file is documentation and may contain placeholder credentials or heavy optional-engine settings. For best results, copy `.env.example` to `.env` and edit `.env` for your environment.

```env
# API engines
OPENAI_API_KEY=...
SIGHTENGINE_USER=...
SIGHTENGINE_SECRET=...

# Global
SAMPLE_FRAMES=12
SHORT_CIRCUIT_PHASH=1
ENGINE_ERROR_POLICY=review
API_POLICY=always
MODIMG_PARALLEL_ENGINES=0
MODIMG_PARALLEL_WORKERS=4
MODIMG_FILE_WORKERS=1

# OCR
OCR_ENABLE=1
OCR_LANG=eng

# pHash auto-learn
PHASH_AUTO_LEARN_ENABLE=0
PHASH_AUTO_ALLOW_APPEND=0
PHASH_AUTO_BLOCK_APPEND=0

# Optional YOLO-World weapons model
# If unset and models/yolov8s-oiv7.pt is missing, the engine is skipped.
YOLO_WEAPON_MODEL=
YOLO_WORLD_MODEL=
YOLO_CONF=0.25
YOLO_IMGSZ=640
YOLO_MAX_FRAMES=2
YOLO_DEVICE=
YOLO_BATCH_ENABLE=1

# OpenNSFW2 speed/stability
OPENNSFW2_IN_PROCESS=1

# Local YOLO forbidden/harmful-symbol model
FORBIDDEN_SYMBOLS_YOLO_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_MODEL=models/forbidden_symbols_yolo.pt
FORBIDDEN_SYMBOLS_YOLO_CONF=0.20
FORBIDDEN_SYMBOLS_YOLO_IOU=0.45
FORBIDDEN_SYMBOLS_YOLO_IMGSZ=960
FORBIDDEN_SYMBOLS_YOLO_MAX_DET=20
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=2
FORBIDDEN_SYMBOLS_YOLO_DEVICE=auto
FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK=1
FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30
FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90
FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF=
FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF=
FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES=1
FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=
```

Useful toggles:
- Main performance knobs: `SAMPLE_FRAMES`, `API_POLICY`, `MODIMG_FILE_WORKERS`, `MODIMG_PARALLEL_ENGINES`, `MODIMG_PARALLEL_WORKERS`, `OPENNSFW2_IN_PROCESS`, `YOLO_BATCH_ENABLE`, `YOLO_IMGSZ`, `YOLO_MAX_FRAMES`, `YOLO_MAX_DET`, `FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES`, `OCR_MAX_FRAMES`, `PHASH_ALLOW_MAX_DISTANCE`, `PHASH_BLOCK_MAX_DISTANCE`
- `API_POLICY=always|on_review|never` controls when API engines run
- `OPENAI_DISABLE=1` / omit `SIGHTENGINE_*` if you don’t use API engines
- `PHASH_ALLOW_DISABLE=1` or `PHASH_BLOCK_DISABLE=1` to disable them selectively
- `SCORE_VERBOSE=1` for more verbose engine scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` for centralized logging
- `MODIMG_PARALLEL_ENGINES=1` to run independent engines concurrently (optional/experimental; disabled by default)
- `MODIMG_FILE_WORKERS=2` or `--file-workers 2` processes multiple input files concurrently; default is `1`.
- `OPENNSFW2_IN_PROCESS=1` is the faster default. Set `OPENNSFW2_IN_PROCESS=0` for the isolated subprocess path, or `auto` to try in-process with one subprocess fallback on normal errors.
- `NO_CHECKS_POLICY=review` controls the fallback when no engine ran: `ok` = allow, `review` = safer default, `block` = strictest mode
- `YOLO_WEAPON_MODEL` or `YOLO_WORLD_MODEL` points to custom YOLO weapon weights; without weights the weapon engine is skipped, not failed.


### Fast preset
Recommended first performance test: `MODIMG_FILE_WORKERS=2` and `MODIMG_PARALLEL_WORKERS=4`.

```env
MODIMG_PARALLEL_ENGINES=1
MODIMG_PARALLEL_WORKERS=4
MODIMG_FILE_WORKERS=2
OPENNSFW2_IN_PROCESS=1
YOLO_BATCH_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE=1
YOLO_MAX_FRAMES=1
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=1
SAMPLE_FRAMES=4
API_POLICY=on_review
```

Speed tradeoffs:
- Checking fewer frames can reduce accuracy on GIFs/animated images if policy-relevant content appears only in skipped frames.
- Too many file workers can overload GPU/VRAM, especially when both YOLO engines run on GPU.
- Start with file workers `2` and engine workers `4`, then benchmark before increasing either value.

### Local YOLO forbidden-symbol configuration
- `FORBIDDEN_SYMBOLS_YOLO_ENABLE=1` enables the bundled local model by default.
- `FORBIDDEN_SYMBOLS_YOLO_CONF=0.20` controls the raw YOLO detection confidence.
- `FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30` controls when detections should push the verdict to `REVIEW`.
- `FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90` controls when detections should push the verdict to `BLOCK`.
- `FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF` and `FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF` can override thresholds per label, e.g. `swastika:0.50,isis:0.75`.
- Recommended defaults: `conf=0.20`, `review=0.30`, `block=0.90`, `imgsz=960`.
- `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0` disables frame inference for this engine and returns an OK result with zero detections.
- For faster CPU-only scans, try the fast preset plus `OCR_MAX_FRAMES=1`, `YOLO_IMGSZ=416`, `YOLO_DEVICE=cpu`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ=640`, and `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`.
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

## Troubleshooting
- `pip install -r requirements.txt` installs nothing or behaves oddly: check that every dependency is on its own line. A damaged requirements file with all dependencies on one line or with dependency lines accidentally commented out is invalid.
- Unsupported Python version: use Python 3.11 or 3.12. Python 3.13+ may work for some packages, but this project does not promise it and the ML stack may reject it.
- OpenNSFW2 backend missing: install `requirements_local.txt` or `.[local]`. The project intentionally uses `opennsfw2[tf-keras]`; plain `opennsfw2` can import without having a usable inference backend.
- OpenNSFW2 native crash: the faster default is in-process prediction. Set `OPENNSFW2_IN_PROCESS=0` to use the isolated Python subprocess so TensorFlow/native crashes become a controlled engine error instead of killing the CLI. `OPENNSFW2_IN_PROCESS=auto` tries in-process first and falls back once on normal Python exceptions.
- TensorFlow/Keras compatibility: keep the local install in a fresh Python 3.11/3.12 venv. If you previously installed Keras/TensorFlow packages manually, recreate the venv and reinstall `requirements_local.txt`.
- Tesseract missing: install the system binary and set `TESSERACT_CMD` if it is not on `PATH`. The OCR engine returns `skipped`/controlled `error` instead of crashing.
- CUDA/GPU not available: the CLI defaults `CUDA_VISIBLE_DEVICES=-1` for process safety. For CPU-only runs you can also set `YOLO_DEVICE=cpu` and `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`; for GPU runs, set `CUDA_VISIBLE_DEVICES` explicitly before starting the CLI.
- YOLO model missing: set `FORBIDDEN_SYMBOLS_YOLO_MODEL` or `YOLO_WEAPON_MODEL` to an absolute path. Missing optional weights are reported as `skipped`.
- Git-LFS pointer instead of real `.pt`: run `git lfs pull`; pointer files are detected and skipped with a clear message.
- API keys missing: `OPENAI_API_KEY`, `SIGHTENGINE_USER`, and `SIGHTENGINE_SECRET` can be omitted. API engines skip cleanly unless enabled credentials are present.
- Windows paths: quote paths with spaces, for example `python moderate_image.py "C:\Users\me\Pictures\Test Image.PNG" --no-apis`. Backslashes and uppercase extensions are supported.
