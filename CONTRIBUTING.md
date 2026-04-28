# Contributing

## Development workflow
1. Create a virtual environment and install dev dependencies.
2. Run `pytest -q` before committing.
3. Keep public CLI/API behavior backward compatible unless explicitly documented.

## Configuration conventions
- Use centralized configuration helpers in `modimg.config` (`get_config()`), not ad-hoc `os.getenv` reads.
- Prefer `_ENABLE` flags for positive toggles and `_DISABLE` for kill switches, but parse all booleans through shared helpers.

## Logging
- Use `modimg.logging_utils.get_logger` instead of `print` for diagnostics.
- Never log secrets (`OPENAI_API_KEY`, `SIGHTENGINE_SECRET`, etc.).

## Tests
- Add regression tests for every bug fix.
- Use `monkeypatch` for env-var dependent behavior.
