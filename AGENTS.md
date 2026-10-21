# AGENTS.md

viy is a library for detecting AI-generated images:
- it converts supported datasets into one canonical layout;
- it builds torch training and validation datasets with traced augmentations;
- it benchmarks detectors on every evaluation split.

## Before changing code

- Read `docs/agents.md`: module map, invariants, conventions, testing rules.
- Read the user guide page for the area you touch: `docs/datasets.md`, `docs/training.md`, `docs/augmentations.md` or `docs/benchmark.md`.
- If `PROJECT-CONTEXT.md` exists, read it for the research background.

## Must hold

- `None` in metadata means unknown, and `()` means known clean.
- A split exists if and only if its CSV exists.
- Augmentation randomness comes only from the generator argument.
- Proxy operations set `proxy_of`, end with `_proxy` and warn.
- Configs are `StrictPydanticBaseModel` subclasses, and objects built from them expose `from_pydantic_model`.
- No `try`/`except` around imports, no fallbacks, no re-raising with changed text.
- Every override has `@override`, every generic has type arguments, and every `ty: ignore` names its rule.

## Commands

```bash
uv sync
uv run pytest
uv run ruff format && uv run ruff check viy && uv run ty check
```

Tests use synthetic fixtures only. Do not download real datasets without being asked.
