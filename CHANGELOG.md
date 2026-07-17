# Changelog

## 0.2.0 - 2026-07-17
- Added production AVIF support for local files, directory scans, public HTTP(S) URLs, uppercase/mixed-case suffixes, extensionless input, and AVIF content stored under unrelated suffixes. Detection validates the bounded leading ISO-BMFF `ftyp` box and explicit `avif`/`avis` major or compatible brands instead of trusting filenames, MIME types, or incidental text; generic HEIF/HEIC brands remain unsupported.
- Raised the core requirement to `Pillow>=11.3.0` and added controlled loader errors for recognized AVIF when the active Pillow build has no working AVIF codec.
- Decodes AVIF natively to reusable RGB frames and applies a bounded encoded-source size plus the existing dimension, pixel, animation-frame, sampled-frame, and aggregate decoded-pixel limits. Static AVIF and real multi-frame `avis` sequences are covered by tests while preserving existing GIF and WebP sampling behavior.
- Added a thread-safe lazy JPEG compatibility path for AVIF only when active path-based NudeNet or OpenNSFW2 code needs it. The file reuses each frame's existing cached JPEG bytes, is shared across parallel consumers, and is not created for pHash, OCR, YOLO, OpenAI, Sightengine, or disabled path engines.
- Preserve the original local path or redacted URL in AVIF reports, expose only JSON-safe native-decode/fallback metadata, and centrally remove downloaded AVIF and frame-owned JPEG temporary files on success, errors, exceptions, and early returns without publishing their paths.
- Added core support for local, directory, and public HTTP(S) SVG inputs, including uppercase extensions and content-validated extensionless SVG files.
- Added `modimg.preprocessing` and `modimg.svg`: untrusted SVG is securely parsed with `defusedxml`, rendered with the Python `resvg_py` API, verified with Pillow, flattened onto a configurable opaque background, and normalized to an RGB PNG before frame loading or engine execution.
- Preserved the original local path or redacted URL in reports while passing the same normalized PNG path to every engine; SVG preprocessing metadata is retained in JSON without exposing temporary paths.
- Rejected DTDs/entities, scripts, `foreignObject`, active/embed elements, event handlers, external resources, local paths, external stylesheets/fonts, and non-fragment CSS references. Internal fragments remain supported, and optional raster data images are MIME-checked and bounded.
- Added independent SVG source, render-dimension, render-pixel, embedded-data, and background settings. SVG limits cannot exceed the existing global image/download/decode ceilings.
- Added centralized cleanup for downloaded SVG sources and generated PNGs on successful, loader-error, renderer-error, and engine-error paths.
- Require SVG renderer output to match the validated target dimensions exactly, and redact every pipeline-owned temporary path before engine errors are logged or returned.
- Enforce the decoded-pixel ceiling consistently for static raster formats, bound DNS/address retries and Tesseract subprocesses by configured timeouts, and close Sightengine responses on every return path.
- Added `resvg_py>=0.3,<1.0` and `defusedxml>=0.7.1,<1.0` to the core runtime and documented the SVG flow, security policy, examples, and environment variables in both READMEs.

## 0.1.2 - 2026-04-28
- Fixed package metadata typos (`name`, maintainers) in `pyproject.toml`.
- Updated README verification example for `--no-apis` to include an explicit input path and aligned expected behavior text.
- Documented `NO_CHECKS_POLICY` options (`ok`, `review`, `block`) under useful toggles.
- Removed unused `OpenAIModerationEngine._script_dir()` helper.

## 0.1.1 - 2026-04-28
- Added typed `Config` object with reload support in `modimg.config`.
- Added shared enums for engine and verdict states in `modimg.enums`.
- Introduced centralized logging setup and migrated CLI output plumbing to logger-backed output.
- Added optional concurrent engine execution (`MODIMG_PARALLEL_ENGINES`, `MODIMG_PARALLEL_WORKERS`) while preserving deterministic result ordering.
- Added regression tests for env flag parsing and concurrent ordering.
- Exported type information marker (`modimg/py.typed`) and optional dependency groups in `pyproject.toml`.
