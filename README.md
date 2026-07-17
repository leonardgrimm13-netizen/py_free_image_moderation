# py_free_image_moderation
A flexible Python project for **image, GIF, SVG, and AVIF moderation** with multiple engines (local + API), pHash lists, and clear CLI output.

**Languages:** **English** | [German](README.de.md)

## Contents
- [Features](#features)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Native AVIF processing](#native-avif-processing)
- [Secure SVG preprocessing](#secure-svg-preprocessing)
- [Verification](#verification)
- [Important configuration (.env)](#important-configuration-env)
- [Result logic (OK / REVIEW / BLOCK)](#result-logic-ok--review--block)
- [Tips for running](#tips-for-running)

---

<a name="features"></a>
## ✨ Features
- **Multi-stage moderation** for raster images, GIFs, SVGs, AVIF files, directories, and URLs
- **Native AVIF decoding** with Pillow, content-based `avif`/`avis` detection, bounded sequence sampling, and lazy compatibility files only for path-based engines
- **Secure SVG preprocessing** with `defusedxml` validation and bounded `resvg_py` SVG-to-PNG rasterization before any engine runs
- **pHash allowlist/blocklist** for very fast short-circuit decisions; pHash auto-learning is off by default to avoid learning false positives
- **OCR text check** (e.g., against text blocklists)
- Combinable engines:
  - `OpenNSFW2`
  - `NudeNet`
  - `YOLO weapon detection` (local YOLO detection using `models/yolov8s-oiv7.pt`)
  - `YOLO forbidden symbols` (local forbidden/harmful-symbol detection using `models/forbidden_symbols_yolo.pt`)
  - `OpenAI Moderation` (optional via API key)
  - `Sightengine` (optional via API credentials)
- **GIF handling** with configurable frame sampling
- **JSON export** for further processing in pipelines
- **Conservative verdict logic** with clear, traceable reasons

---

<a name="project-structure"></a>
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
│   ├── forbidden_symbols_yolo.pt  # bundled local YOLO model for forbidden-symbol detection
│   └── yolov8s-oiv7.pt            # bundled local YOLO model used for weapon detection
├── data/
│   ├── phash_allowlist.txt
│   ├── phash_blocklist.txt
│   └── ocr_text_blocklist.txt
└── modimg/
    ├── cli.py               # Args, output, JSON export
    ├── pipeline.py          # Flow & engine orchestration
    ├── preprocessing.py     # Input normalization before decoding/inference
    ├── svg.py               # Secure validation and bounded PNG rasterization
    ├── verdict.py           # Final decision logic
    ├── frames.py            # Raster/GIF/WebP/AVIF frame loading
    ├── phash.py             # pHash utilities
    ├── config.py            # .env loading
    └── engines/             # Individual moderation engines
```

---

<a name="installation"></a>
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
- `Pillow>=11.3.0` (including native AVIF decoding)
- `numpy`
- `ImageHash`
- `resvg_py>=0.3,<1.0` (SVG-to-PNG renderer)
- `defusedxml>=0.7.1,<1.0` (secure XML pre-validation)

This is enough for raster and AVIF loading, GIF/WebP/AVIF frame sampling, secure SVG preprocessing, pHash allow/block lists, JSON output, and graceful skipping of optional engines. AVIF decoding requires Pillow 11.3.0 or newer with its AVIF codec enabled; SVG support is also part of the core installation, not an optional extra.

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

### 4) Bundled local YOLO models
This repository includes `models/forbidden_symbols_yolo.pt` and `models/yolov8s-oiv7.pt` directly as normal repository files.

The model is loaded locally by the `YOLO forbidden symbols` engine. It never calls Roboflow or any external API at runtime. If the file is missing, set `FORBIDDEN_SYMBOLS_YOLO_MODEL` to an absolute path or run from the project root. If Git-LFS left a pointer file instead of the real weights, run `git lfs pull`.

The separate `YOLO-World weapons` engine uses the bundled `models/yolov8s-oiv7.pt` by default. Both source checkouts and installed wheels resolve bundled data automatically. To use your own weapon weights, set either `YOLO_WEAPON_MODEL=/absolute/or/project/relative/model.pt` or `YOLO_WORLD_MODEL=/absolute/or/project/relative/model.pt`.

### 5) Optional system dependency for OCR
For OCR you typically need a local Tesseract install:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (Homebrew): `brew install tesseract`

---

<a name="quickstart"></a>
## 🚀 Quickstart

### Check a single image
```bash
python moderate_image.py /path/to/image.jpg
```

### Check a GIF (frame sampling)
```bash
python moderate_image.py /path/to/file.gif --sample-frames 12
```

### Check a local SVG
```bash
python moderate_image.py /path/to/artwork.svg --no-apis
python -m modimg.cli /path/to/artwork.svg --no-apis
moderate-image /path/to/artwork.svg --no-apis
```

Uppercase `.SVG` files and valid SVG content without a file extension are also discovered by content. An `.svg` suffix alone never makes malformed or non-SVG XML valid.

### Check a local AVIF
```bash
python moderate_image.py /path/to/photo.avif --no-apis
python -m modimg.cli /path/to/photo.AVIF --no-apis
moderate-image /path/to/extensionless-avif --no-apis
```

`.avif` matching is case-insensitive, so `.AVIF` and mixed-case variants work. A valid extensionless AVIF, or an AVIF stored under an unrelated suffix, is recognized from its ISO-BMFF `ftyp` contents rather than trusted from the name alone.

### Check an image, SVG, or AVIF URL
```bash
python moderate_image.py "https://example.com/image.jpg"
python moderate_image.py "https://example.com/artwork.svg" --no-apis
python moderate_image.py "https://example.com/photo.avif" --no-apis
python moderate_image.py "https://example.com/image?id=42" --no-apis
```

### Check a directory
```bash
python moderate_image.py ./images --recursive
```

### Check multiple explicit inputs
```bash
python moderate_image.py first.jpg second.png animation.gif photo.avif artwork.svg --no-apis
```

Repeated or overlapping local paths are processed once, preserving the order of their first occurrence.

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

<a name="native-avif-processing"></a>
## 🌄 Native AVIF processing

AVIF is recognized from a bounded parse of the leading ISO-BMFF `ftyp` box. The declared box size, box boundaries, major brand, and compatible brands are validated; only the explicit AVIF brands `avif` and `avis` are accepted. Merely finding the text `avif`, seeing an `.avif` suffix, or receiving `Content-Type: image/avif` is not sufficient. Generic HEIF/HEIC files without an AVIF brand are not treated as AVIF.

Local `.avif`, uppercase `.AVIF`, mixed-case suffixes, extensionless files, and AVIF content under an unrelated suffix are supported. Public HTTP(S) URLs are likewise accepted by content even when their path has no extension and their `Content-Type` is absent, generic, or incorrect. URL input continues to use the same DNS-pinned, redirect-validated, SSRF-protected, size-limited downloader. Invalid data advertised as AVIF is rejected, while a valid download is stored temporarily with an `.avif` suffix and removed centrally after processing.

Pillow 11.3.0 or newer decodes AVIF directly into the pipeline's canonical RGB frames. AVIF is not routinely converted to PNG or JPEG. Static images and real multi-frame `avis` sequences have been tested locally; sequences use the same frame-count, sampled-frame, dimension, per-image pixel, and aggregate decoded-pixel limits as GIF and animated WebP input. Selected frames are decoded once and reused by pHash, OCR, both YOLO engines, automatic pHash learning, and all later pipeline stages.

OpenAI and Sightengine continue to request JPEG bytes from each frame's existing thread-safe cache, so both can share one encoding per frame. NudeNet and OpenNSFW2 are path-based integrations that cannot be assumed to decode AVIF in every backend or subprocess mode. Only when either active engine actually asks for a compatible path does the frame lazily create a fully written temporary JPEG from its already decoded RGB pixels. The same cached bytes and per-frame path are reused across engines and concurrent calls; no AVIF re-decode or duplicate fallback encoding is needed. If neither path engine runs, no compatibility file is created.

Reports retain the original local path or redacted original URL. They can record `source_format: avif`, native decoding, and whether a JPEG engine fallback was actually created, but never expose an internal path. Frame-owned compatibility files and downloaded AVIF files are removed after success, loader or engine errors, exceptions, and early returns.

Pillow builds can theoretically be compiled without the AVIF codec. When AVIF content is detected in such an environment, loading fails in a controlled way with a message that Pillow 11.3.0 or newer with AVIF support is required, instead of leaking a low-level decoder or plugin exception.

---

<a name="secure-svg-preprocessing"></a>
## 🛡️ Secure SVG preprocessing

SVG is active XML content, so an SVG file is never passed directly to Pillow or a moderation engine. Local files and public HTTP(S) URLs use this normalization flow:

```text
original local path or URL
    → content detection and defusedxml validation
    → bounded resvg_py rendering (resources_dir=None)
    → PNG signature, Pillow verify/decode, and dimension checks
    → opaque-background RGB PNG
    → normal frame loading and every moderation engine
    → centralized temporary-file cleanup
```

The renderer is called through its Python API, without a shell command. It receives the validated SVG text and no user-controlled resource directory. The resulting bytes must be a decodable PNG with the expected bounded dimensions; Pillow then composites transparency onto the configured opaque background and writes a verified RGB PNG. This normalized PNG path is supplied to every engine, including engines that read a file path directly. Raster images and GIFs continue through their existing path without SVG rasterization.

Detection is content-based. `.svg` and `.SVG` are directory candidates, but the suffix, URL path, and `Content-Type` are never trusted without successfully parsing the complete document and confirming an `svg` root element. Valid extensionless SVG is accepted. UTF-8 (with or without BOM) and BOM-marked UTF-16 LE/BE are supported; contradictory, malformed, or unsupported encodings are rejected. SVG URLs retain the existing HTTP(S)-only downloader, DNS pinning, redirect validation, SSRF protection, timeout, and download limit. An HTML response advertised as `image/svg+xml` is still rejected.

Gzip-compressed `.svgz` input is intentionally not supported.

Security policy for untrusted SVG:

- `DOCTYPE`, entity declarations/expansion, external XML entities, malformed XML, `<script>`, `<foreignObject>`, `<iframe>`, `<object>`, `<embed>`, and event-handler attributes are rejected.
- External URLs, local/relative/absolute/Windows/UNC paths, XML stylesheets, CSS `@import`, external fonts (`@font-face`), and non-fragment CSS `url(...)` references are rejected before rendering.
- Internal fragments such as `href="#symbol"` and `url(#gradient)` remain allowed for symbols, gradients, masks, and clip paths.
- Optional embedded raster data images are limited to valid base64 PNG, JPEG, WebP, or GIF content on `<image>`/`<feImage>` elements. MIME/content matching, per-image and aggregate byte limits, animation limits, and decoded-pixel limits apply. Nested SVG and arbitrary data URIs are rejected; set `MODIMG_SVG_ALLOW_DATA_IMAGES=0` to disable raster data images entirely.

Output dimensions come from valid absolute `width`/`height`, otherwise a valid `viewBox`, otherwise the configured defaults. Physical units use a fixed 96 DPI; relative/percentage dimensions do not provide a trusted standalone size. Oversized but valid artwork is scaled down with its aspect ratio preserved. SVG-specific dimension/pixel limits and the global image/decode limits are enforced before and after rendering, and the stricter applicable value wins. URL SVG source size is also bounded by the stricter of `MODIMG_MAX_SVG_BYTES` and `MODIMG_MAX_DOWNLOAD_BYTES`.

The default background is opaque white (`#ffffff`). `MODIMG_SVG_BACKGROUND` accepts an opaque Pillow-supported CSS color; invalid or transparent values produce a controlled configuration error instead of silently turning transparent regions black.

Reports keep the original local path or redacted original URL. They never expose the downloaded SVG or normalized PNG temporary path. Successful SVG normalization adds JSON-compatible metadata such as:

```json
{
  "preprocessing": {
    "source_format": "svg",
    "normalized_format": "png",
    "renderer": "resvg_py",
    "render_width": 1024,
    "render_height": 768,
    "background": "#ffffff"
  }
}
```

Downloaded sources and generated PNGs are tracked centrally and removed in reverse order after success or failure. Cleanup failures are safely logged without replacing the moderation result.

---

<a name="verification"></a>
## ✅ Verification
Core install:
```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python moderate_image.py --help
python moderate_image.py path/to/test.png --no-apis
python moderate_image.py path/to/test.svg --no-apis
python moderate_image.py path/to/test.avif --no-apis
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
- `python moderate_image.py path/to/test.png --no-apis` or the SVG/AVIF equivalent → exit code `0` if the input is `OK`, or `2` if it returns `REVIEW`/`BLOCK`.

Optional engines may be missing; they must show up as `skipped`/`disabled` in output instead of aborting execution.

---

<a name="important-configuration-env"></a>
## 🔧 Important configuration (.env)
Library imports automatically load `.env` from the source/package root. The installed CLI additionally checks the current working directory first, which allows a project-local `.env` without making ordinary imports trust arbitrary working directories. Example:

The loader checks `.env`, then `.env.txt`. It does not load `.env.example` automatically, because that file is documentation and may contain placeholder credentials or heavy optional-engine settings. For best results, copy `.env.example` to `.env` and edit `.env` for your environment.

Without `OPENAI_CACHE_PATH`, the OpenAI cache uses `XDG_CACHE_HOME` or `~/.cache` on Linux, `LOCALAPPDATA` on Windows, and `~/Library/Caches` on macOS, below `py-free-image-moderation`. An explicit absolute path is used unchanged. For backward compatibility, an explicit relative `OPENAI_CACHE_PATH` remains relative to the source/package root.

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
MODIMG_MAX_FILE_WORKERS=32

# Untrusted input limits
MODIMG_MAX_DOWNLOAD_BYTES=25000000
MODIMG_MAX_AVIF_BYTES=100000000
MODIMG_URL_TIMEOUT_SEC=20
MODIMG_MAX_URL_REDIRECTS=5
MODIMG_ALLOW_PRIVATE_URLS=0
MODIMG_MAX_IMAGE_DIMENSION=32768
MODIMG_MAX_IMAGE_PIXELS=64000000
MODIMG_MAX_ANIMATION_FRAMES=5000
MODIMG_MAX_DECODED_PIXELS=256000000

# SVG preprocessing
MODIMG_MAX_SVG_BYTES=10000000
MODIMG_SVG_DEFAULT_WIDTH=1024
MODIMG_SVG_DEFAULT_HEIGHT=1024
MODIMG_SVG_MAX_RENDER_DIMENSION=4096
MODIMG_SVG_MAX_RENDER_PIXELS=16000000
MODIMG_SVG_BACKGROUND=#ffffff
MODIMG_SVG_ALLOW_DATA_IMAGES=1
MODIMG_SVG_MAX_EMBEDDED_IMAGE_BYTES=5000000
MODIMG_SVG_MAX_TOTAL_EMBEDDED_BYTES=10000000

# OCR
OCR_ENABLE=1
OCR_LANG=eng
OCR_TIMEOUT_SEC=30

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
OPENNSFW2_IN_PROCESS=0

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

`MODIMG_MAX_AVIF_BYTES` limits an encoded AVIF source to 100 MB by default before Pillow reads it into memory. AVIF URLs obey both this source limit and `MODIMG_MAX_DOWNLOAD_BYTES`; the stricter configured value wins.

SVG defaults and limits:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `MODIMG_MAX_SVG_BYTES` | `10000000` | Maximum uncompressed SVG source size; URL input also obeys the download limit. |
| `MODIMG_SVG_DEFAULT_WIDTH` | `1024` | Fallback width when no usable absolute size or `viewBox` exists. |
| `MODIMG_SVG_DEFAULT_HEIGHT` | `1024` | Fallback height when no usable absolute size or `viewBox` exists. |
| `MODIMG_SVG_MAX_RENDER_DIMENSION` | `4096` | Maximum SVG raster width or height before stricter global limits. |
| `MODIMG_SVG_MAX_RENDER_PIXELS` | `16000000` | Maximum SVG raster pixel count before stricter global limits. |
| `MODIMG_SVG_BACKGROUND` | `#ffffff` | Required opaque CSS background used for rendering and alpha compositing. |
| `MODIMG_SVG_ALLOW_DATA_IMAGES` | `1` | Allows only the bounded, validated raster data images described above. |
| `MODIMG_SVG_MAX_EMBEDDED_IMAGE_BYTES` | `5000000` | Maximum decoded bytes for each embedded raster image. |
| `MODIMG_SVG_MAX_TOTAL_EMBEDDED_BYTES` | `10000000` | Maximum decoded bytes across all embedded raster images in one SVG. |

`MODIMG_MAX_IMAGE_DIMENSION`, `MODIMG_MAX_IMAGE_PIXELS`, and `MODIMG_MAX_DECODED_PIXELS` remain hard global ceilings; increasing an SVG-specific limit cannot bypass them. Invalid numeric values use the existing safe environment-parser defaults, while an invalid or non-opaque background is reported as a controlled SVG configuration error when SVG preprocessing runs.

Useful toggles:
- Main performance knobs: `SAMPLE_FRAMES`, `API_POLICY`, `MODIMG_FILE_WORKERS`, `MODIMG_PARALLEL_ENGINES`, `MODIMG_PARALLEL_WORKERS`, `OPENNSFW2_IN_PROCESS`, `YOLO_BATCH_ENABLE`, `YOLO_IMGSZ`, `YOLO_MAX_FRAMES`, `YOLO_MAX_DET`, `FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES`, `OCR_MAX_FRAMES`, `PHASH_ALLOW_MAX_DISTANCE`, `PHASH_BLOCK_MAX_DISTANCE`
- `API_POLICY=always|on_review|never` controls when API engines run
- `OPENAI_DISABLE=1` / omit `SIGHTENGINE_*` if you don’t use API engines
- `PHASH_ALLOW_DISABLE=1` or `PHASH_BLOCK_DISABLE=1` to disable them selectively
- `SCORE_VERBOSE=1` for more verbose engine scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` for centralized logging
- `MODIMG_PARALLEL_ENGINES=1` to run independent engines concurrently (optional/experimental; disabled by default)
- `MODIMG_FILE_WORKERS=2` or `--file-workers 2` processes multiple input files concurrently; default is `1`.
- `OPENNSFW2_IN_PROCESS=0` is the robust default and runs prediction in an isolated subprocess. Set `OPENNSFW2_IN_PROCESS=1` only for faster in-process prediction; native TensorFlow crashes can terminate the whole CLI process. `auto` tries in-process first and falls back once on normal Python errors.
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
- Checking fewer frames can reduce accuracy on GIFs, animated WebP, and AVIF sequences if policy-relevant content appears only in skipped frames.
- Too many file workers can overload GPU/VRAM, especially when both YOLO engines run on GPU.
- Start with file workers `2` and engine workers `4`, then benchmark before increasing either value.

### Local YOLO forbidden-symbol configuration
- `FORBIDDEN_SYMBOLS_YOLO_ENABLE=1` enables the bundled local model by default.
- `FORBIDDEN_SYMBOLS_YOLO_CONF=0.20` controls the raw YOLO detection confidence.
- `FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30` controls when detections should push the verdict to `REVIEW`.
- `FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90` controls when detections should push the verdict to `BLOCK`.
- `FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF` and `FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF` can override thresholds per label, e.g. `swastika:0.50,isis:0.75`.
- Recommended defaults: `conf=0.20`, `review=0.30`, `block=0.90`, `imgsz=960`.
- `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0` disables frame inference for this engine and returns `skipped`; it does not count as a successful moderation check.
- For faster CPU-only scans, try the fast preset plus `OCR_MAX_FRAMES=1`, `YOLO_IMGSZ=416`, `YOLO_DEVICE=cpu`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ=640`, and `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`.
- Unreliable labels can be ignored at runtime, e.g. `FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=communism,antifa`.

---

<a name="result-logic-ok--review--block"></a>
## 🧠 Result logic (OK / REVIEW / BLOCK)
- **Staged pipeline:** `pHash` → local engines → optional API engines → final verdict
- **pHash short-circuit** can decide early:
  - allowlist hit → `OK`
  - blocklist hit → `BLOCK`
- If pHash does not short-circuit, the remaining local engines run, including `YOLO forbidden symbols`.
- With `SHORT_CIRCUIT_PHASH=0`, an allowlist hit is only recorded as context and cannot override a later blocklist, OCR, local-engine, or API block.
- The forbidden-symbol YOLO engine contributes to hate/policy risk: at or above the block threshold it should result in `BLOCK`; at or above the review threshold it should result in `REVIEW`.
- Detection labels and boxes are written to JSON under engine `details.detections`.
- `verdict.py` condenses signals (nudity, violence, hate) into the final decision
- Error behavior can be controlled via `ENGINE_ERROR_POLICY` (`ignore`, `review`, `block`)

### Security limits for untrusted input
- URL input is restricted to HTTP(S), rejects embedded credentials, validates every DNS answer, pins the validated address to the connection, and revalidates every redirect. Loopback, private, link-local, multicast, and otherwise non-public targets are rejected by default.
- `MODIMG_ALLOW_PRIVATE_URLS=1` disables the private-network SSRF guard. Use it only for trusted URLs in a controlled network.
- URL downloads are streamed with a total timeout, redirect limit, byte limit, format signature check, and guaranteed cleanup of partial temporary files. Query values and URL credentials are removed from reports and errors.
- Pillow decode limits cap dimensions, pixels, animation frame counts, sampled frames, and aggregate decoded pixels, including native AVIF sequences. A selected animation frame that cannot be decoded fails the loader instead of being silently skipped.
- pHash lists and the OpenAI cache use bounded reads and atomic replacement. In-process locks serialize cache/list writes and cached model inference.
- Bundled model defaults are resolved only from source-package and installation roots; the current directory is never an automatic fallback. Explicitly configured relative resource paths resolve from the current directory first, followed by source-package and installation fallbacks. Treat every custom `.pt` file as trusted executable model data.
- OCR blocklists treat ordinary lines as literal text. Prefix a deliberate regular expression with `re:`.

---

<a name="tips-for-running"></a>
## 🛠️ Tips for running
- Start with `--no-apis` to verify the local pipeline and performance first.
- Use `--json` if results should be processed in CI/CD or backend services.
- Maintain `data/phash_allowlist.txt` and `data/phash_blocklist.txt` regularly for stable decisions on recurring content.
- For GIFs, animated WebP, and AVIF sequences, increase `--sample-frames` if problematic content appears only in a few frames.

## Troubleshooting
- `pip install -r requirements.txt` installs nothing or behaves oddly: check that every dependency is on its own line. A damaged requirements file with all dependencies on one line or with dependency lines accidentally commented out is invalid.
- Unsupported Python version: use Python 3.11 or 3.12. Python 3.13+ may work for some packages, but this project does not promise it and the ML stack may reject it.
- AVIF codec unavailable: install `Pillow>=11.3.0` using an official wheel or another build with AVIF support. Recognized AVIF input produces a controlled loader error when the active Pillow build lacks a working codec.
- OpenNSFW2 backend missing: install `requirements_local.txt` or `.[local]`. The project intentionally uses `opennsfw2[tf-keras]`; plain `opennsfw2` can import without having a usable inference backend.
- OpenNSFW2 native crash: the default is the isolated Python subprocess (`OPENNSFW2_IN_PROCESS=0`) so TensorFlow/native crashes become a controlled engine error instead of killing the CLI. `OPENNSFW2_IN_PROCESS=1` is faster but less robust because native TensorFlow crashes can terminate the whole process. `OPENNSFW2_IN_PROCESS=auto` tries in-process first and falls back once on normal Python exceptions, but cannot recover from native process crashes.
- TensorFlow/Keras compatibility: keep the local install in a fresh Python 3.11/3.12 venv. If you previously installed Keras/TensorFlow packages manually, recreate the venv and reinstall `requirements_local.txt`.
- Tesseract missing: install the system binary and set `TESSERACT_CMD` if it is not on `PATH`. The OCR engine returns `skipped`/controlled `error` instead of crashing.
- CUDA/GPU not available: the CLI defaults `CUDA_VISIBLE_DEVICES=-1` for process safety. For CPU-only runs you can also set `YOLO_DEVICE=cpu` and `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`; for GPU runs, set `CUDA_VISIBLE_DEVICES` explicitly before starting the CLI.
- YOLO model missing: set `FORBIDDEN_SYMBOLS_YOLO_MODEL` or `YOLO_WEAPON_MODEL` to an absolute path. Missing optional weights are reported as `skipped`.
- Git-LFS pointer instead of real `.pt`: run `git lfs pull`; pointer files are detected and skipped with a clear message.
- API keys missing: `OPENAI_API_KEY`, `SIGHTENGINE_USER`, and `SIGHTENGINE_SECRET` can be omitted. API engines skip cleanly unless enabled credentials are present.
- Windows paths: quote paths with spaces, for example `python moderate_image.py "C:\Users\me\Pictures\Test Image.PNG" --no-apis`. Backslashes and uppercase extensions are supported.
