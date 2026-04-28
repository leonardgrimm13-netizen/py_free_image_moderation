# Changelog

## 0.1.1 - 2026-04-28
- Added typed `Config` object with reload support in `modimg.config`.
- Added shared enums for engine and verdict states in `modimg.enums`.
- Introduced centralized logging setup and migrated CLI output plumbing to logger-backed output.
- Added optional concurrent engine execution (`MODIMG_PARALLEL_ENGINES`, `MODIMG_PARALLEL_WORKERS`) while preserving deterministic result ordering.
- Added regression tests for env flag parsing and concurrent ordering.
- Exported type information marker (`modimg/py.typed`) and optional dependency groups in `pyproject.toml`.
