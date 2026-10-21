import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final
from typing import override

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL
from viy.layout import SampleRecord

from .base import DatasetBuilder
from .base import RawSample
from .base import SplitRole
from .base import SplitSpec
from .hub import HubArtifact


REPO: Final = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
REAL_FOLDER: Final = "Real"
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})


class MLLMGenerated(DatasetBuilder):
    """
    MLLM-Generated Image Detection benchmark, arXiv 2608.01258.

    Raw layout: ``images/<generator>/<domain> Images/<file>``, where the
    generator folder ``Real`` holds real photos. The Hub publishes a single
    split with every image, so the dataset has no validation or test split.
    Real images are mostly small JPEGs and fakes are large PNGs, which is a
    format shortcut reported by the split sanity checks.
    """

    name = "mllm_generated"
    description = (
        "GPT Image 2, Nano Banana 2, FLUX.2 and Qwen Image 2.1 against real photos."
    )
    split_specs = (
        SplitSpec(
            name="train",
            role=SplitRole.TRAIN,
            artifacts=(HubArtifact(REPO, "images/**"),),
            description="Every image of the dataset; the Hub publishes no other split.",
        ),
    )
    default_train_splits = ("train",)

    @override
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        images_dir = raw_dir / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"MLLM-Generated raw directory not found: {images_dir}"
            )
        for generator_dir in sorted(
            path for path in images_dir.iterdir() if path.is_dir()
        ):
            is_real = generator_dir.name == REAL_FOLDER
            for domain_dir in sorted(
                path for path in generator_dir.iterdir() if path.is_dir()
            ):
                domain = _slug(domain_dir.name.removesuffix(" Images"))
                for image in sorted(domain_dir.iterdir()):
                    if image.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    yield RawSample(
                        source=image,
                        record=SampleRecord(
                            image=f"images/{_slug(generator_dir.name)}/{domain}/{image.name}",
                            label=REAL_LABEL if is_real else FAKE_LABEL,
                            generator=None if is_real else generator_dir.name,
                            extra={"domain": domain},
                        ),
                    )
        return


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
