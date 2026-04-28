# py_free_image_moderation
A flexible Python project for **image and GIF moderation** with multiple engines (local + API), pHash lists, and clear CLI output.

**Languages:** **English** | [Deutsch](README.de.md)

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
- **pHash allowlist/blocklist** for very fast short-circuit decisions
- **OCR text check** (e.g., against text blocklists)
- Combinable engines:
  - `OpenNSFW2`
  - `NudeNet`
  - `YOLO` (weapon detection)
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

python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 2) Install options

#### A) Offline/Local
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Includes the local runtime + engine dependencies (without API clients):
- `Pillow`
- `numpy`
- `ImageHash`
- `opennsfw2`
- `nudenet`
- `ultralytics`
- `pytesseract`

This enables the local pipeline including pHash and `--no-apis`.

#### B) With APIs
```bash
pip install -r requirements_api.txt
```

Includes everything from `requirements.txt` plus API clients:
- `openai` (OpenAI moderation)
- `sightengine` (Sightengine API)

### 3) Dev/Test dependencies
```bash
pip install -r requirements-dev.txt
```

Includes e.g. `pytest` for local test runs.

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

### Without external APIs (base install is enough)
```bash
python moderate_image.py ./images --recursive --no-apis
```

### Write a JSON report
```bash
python moderate_image.py ./images --recursive --json moderation_report.json
```

**Exit codes:**
- `0` = all results are `OK`
- `2` = at least one result is not `OK`

---

## ✅ Verification
```bash
python -m compileall -q .
pytest -q
python moderate_image.py --help
python moderate_image.py --no-apis
```

Expected behavior (short):
- `python -m compileall -q .` → exit code `0` if code is syntactically valid.
- `pytest -q` → exit code `0` if tests pass, otherwise non-zero.
- `python moderate_image.py --help` → exit code `0` and shows CLI help.
- `python moderate_image.py --no-apis` → exit code `0` (only `OK`) or `2` (at least one `REVIEW/BLOCK`).

Optional engines may be missing; they must show up as `skipped`/`disabled` in output instead of aborting execution.

---

## 🔧 Important configuration (.env)
The project automatically loads `.env` from the project root. Example:

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
PHASH_AUTO_ALLOW_APPEND=1
PHASH_AUTO_BLOCK_APPEND=1
```

Useful toggles:
- `OPENAI_DISABLE=1` / omit `SIGHTENGINE_*` if you don’t use API engines
- `PHASH_ALLOW_DISABLE=1` or `PHASH_BLOCK_DISABLE=1` to disable them selectively
- `SCORE_VERBOSE=1` for more verbose engine scores

---

## 🧠 Result logic (OK / REVIEW / BLOCK)
- **pHash short-circuit** can decide early:
  - allowlist hit → `OK`
  - blocklist hit → `BLOCK`
- Then the remaining engines are aggregated
- `verdict.py` condenses signals (nudity, violence, hate) into the final decision
- Error behavior can be controlled via `ENGINE_ERROR_POLICY` (`ignore`, `review`, `block`)

---

## 🛠️ Tips for running
- Start with `--no-apis` to verify the local pipeline and performance first.
- Use `--json` if results should be processed in CI/CD or backend services.
- Maintain `data/phash_allowlist.txt` and `data/phash_blocklist.txt` regularly for stable decisions on recurring content.
- For GIFs, increase `--sample-frames` if problematic content appears only in a few frames.
