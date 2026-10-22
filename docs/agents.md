# Notes for coding agents and contributors

This page explains how the library is put together and which rules every change must keep. The user guide lives in the other pages of `docs/`. If the repository contains `PROJECT-CONTEXT.md`, read it for background on the research problem the library serves.

## Map

```
viy/
  common.py          ImageArray, ScaleValue, label constants
  registry.py        Registry[T]: name -> object, descriptive lookup errors
  layout/            the canonical on-disk format
    records.py       SampleRecord, split CSV read/write
    store.py         DatasetLayout: paths, splits, dataset.json, DirLock
    linking.py       symlink / hardlink / copy materialization
    checks.py        sanity checks run on every split load
  datasets/          the supported pool
    base.py          DatasetBuilder: prepare-on-demand, Hub fetch, link, write, load, check
    hub.py           HubArtifact, snapshot download, zip extraction
    ntire.py, mllm.py   concrete builders: only collect() is dataset-specific
    configs.py       StorageConfig, DatasetSourceConfig
  augs/
    base.py          ImageOp, RangeImageOp, AppliedOp, OPS registry, proxy warnings
    ops/             concrete operations, one module per family
    steps.py         OpStep, SequenceStep, ChainStep, AlbumentationsStep
    configs.py       pydantic step configs, discriminated by `kind`
    presets.py       NTIRE mixes expressed as step configs
    pipeline.py      build_step, AugmentationPipeline
  data/
    preprocessing.py Preprocessor: custom settings or a transformers image processor
    dataset.py       SplitDataset (one split), DetectionSample
    collate.py       DetectionCollator, DetectionBatch
    detection.py     DatasetSpec, DetectionDataset (several splits, one stage), check_disjoint
    sampling.py      subsample
    configs.py       Stage, DatasetSpecConfig, DetectionDatasetConfig
  bench/
    model.py         DetectorModel, ModelMeta
    task.py          TaskMetadata, DetectionTask, builtin_tasks, get_tasks, register_task
    slices.py        slices from split metadata
    metrics.py       BinaryMetrics
    results.py       TaskResult, load_results, results_table
    runner.py        Benchmark, BenchmarkConfig
tests/               pytest on synthetic data; conftest.py builds fake raw datasets
docs/                user guide
```

Data flow for training:

```
raw files --DatasetBuilder.collect--> RawSample(source, record)
          --link_files--> <root>/<ds>/images/..., --write_records--> splits/<split>.csv
          --DatasetBuilder.load--> list[SampleRecord] + check_split
          --SplitDataset--> decode -> AugmentationPipeline(image, rng) -> Preprocessor -> DetectionSample
          --DetectionDataset--> concatenation of the SplitDatasets of one stage
```

The benchmark takes the same path up to `load`. After that, `model.preprocess` runs in workers, `model.predict` scores the batches, and `DetectionTask.evaluate` turns slices and metrics into a `TaskResult`.

## Invariants

Keep these true; the tests rely on them.

- **`None` means unknown.** In `SampleRecord`, `None` is unknown and `()` is a known clean image. Never fill unknown metadata with guesses.
- **A split exists if and only if `splits/<split>.csv` exists.** A prepared split is never rebuilt implicitly. Custom splits are plain CSV files over already linked images.
- **Preparation runs under the dataset `DirLock`** and re-checks for the split after taking the lock.
- **Augmentation randomness comes only from the `numpy.random.Generator` argument.** No `random`, no `np.random.*` module functions, no torch RNG inside operations. The same seed must give the same image and the same trace.
- **Operations do not modify their input**, and they return `uint8` RGB arrays. The size may change.
- **Operation names and levels follow the NTIRE metadata vocabulary** (`distortions`, `distortion_scales`), so traces and benchmark slices match.
- **A proxy declares `proxy_of`, ends its name with `_proxy`, and warns when created.** Registration rejects any mismatch.
- **Registration happens on class creation** through `__init_subclass__` for `DatasetBuilder` and `ImageOp`. A new module must be imported from its package `__init__.py`.
- **Configs subclass `kostyl.StrictPydanticBaseModel`.** Objects built from a config expose `from_pydantic_model(config)`.
- **Every entry point accepts `storage: StorageConfig | None`.** `None` means the default root, `$VIY_ROOT` or `~/.cache/viy`; no API requires a path.
- **Training and validation datasets are built separately**, one `DetectionDataset` per stage. `check_disjoint` compares resolved file paths to catch overlap.

## Code conventions

- Python 3.13 or newer: PEP 695 generics and `type` aliases, `typing.override`.
- Ruff runs with `select = ["ALL"]` and the ignores in `pyproject.toml`. Ty must be clean, including:
  - `missing-override-decorator`: every override needs `@override`;
  - `missing-type-argument`: write `dict[str, Any]` and `npt.NDArray[np.uint8]`, never bare generics;
  - `blanket-ignore-comment`: every ignore names its rule, as in `# ty: ignore[rule]`.
- Google-style docstrings with argument types in parentheses. Public classes, methods and `__init__` are documented. Functions that return `None` end with an explicit `return`.
- One import per line, with sections ordered as in the isort settings. Imports are at module top level and are never wrapped in `try`/`except`.
- No fallbacks and no re-raising with different text. Let errors surface, and write the context into the original message instead.
- Logging: `logger = setup_logger(fmt="detailed")` from `kostyl.utils` at module level, f-string messages, `warning_once` for repeated warnings.
- Comments only when they say something the code cannot. No parenthetical asides in code, comments or messages.
- Match the surrounding code: dataclasses with `frozen=True, slots=True` for records, `MappingProxyType` for constant mappings, `Final` for module constants.

## Commands

```bash
uv sync                        # environment
uv run pytest                  # tests, seconds; no network, no real data
uv run ruff format             # formatting
uv run ruff check viy          # lint; tests/ is excluded by configuration
uv run ty check                # types, including tests/
```

Run all three before handing work back.

## Testing

- **Never download real datasets in tests.** `tests/conftest.py` builds tiny raw layouts that mirror the real formats: NTIRE label CSVs with Python-literal lists, private-test metadata, and the MLLM folder tree. Extend these fixtures for new datasets.
- **Hub access is tested by monkeypatching `viy.datasets.hub.snapshot_download`.**
- **Parametrize over `OPS.names()`.** Every registered operation then gets the shape, dtype, immutability and reproducibility checks automatically.
- **Check new dataset converters against the real metadata when possible.** Copy only the label and metadata files, replace the images with empty placeholder files, and run `prepare` and `load` at full scale.

## Known gaps

- Learned codecs, adversarial attacks and watermarks are only proxies. The plan is genuine versions generated offline as new splits.
- The internal layout of the NTIRE training shard archives on the Hub has not been verified on a real download.
- Robustness curves on clean images with controlled per-level distortions are not implemented. The benchmark only slices distortions that a dataset already records.
- There is no generator registry with family, decoder and release date, and no near-duplicate detection across datasets.
