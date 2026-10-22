# viy documentation

| Page                                 | Read it when you want to                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| [Getting started](getting-started.md) | get a training loader and a benchmark result in a few minutes                   |
| [Datasets](datasets.md)               | understand where data lives, what each dataset provides, or make your own split |
| [Training data](training.md)          | configure training and validation datasets                                      |
| [Augmentations](augmentations.md)     | choose or write augmentations and read their traces                             |
| [Benchmark](benchmark.md)             | evaluate a model and compare results                                            |
| [Extending](extending.md)             | add a dataset, an operation or a benchmark task                                 |
| [Agent notes](agents.md)              | change the library code, as a person or as a coding agent                       |

The modules map onto these pages:

```
viy.layout     canonical on-disk format, split CSV, sanity checks          -> datasets.md
viy.datasets   supported datasets, Hub download, preparation              -> datasets.md
viy.augs       operations, steps, NTIRE presets                            -> augmentations.md
viy.data       Preprocessor, DetectionDataset, check_disjoint             -> training.md
viy.bench      DetectorModel, tasks, metrics, Benchmark runner             -> benchmark.md
```
