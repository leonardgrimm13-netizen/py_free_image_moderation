# Changelog

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
