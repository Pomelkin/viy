from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Final
from typing import override

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL
from viy.layout import SampleRecord

from .base import SplitRole
from .base import SplitSpec
from .hub import HubArtifact
from .parquet import ParquetShardBuilder


REPO: Final = "ComplexDataLab/OpenFake"
V1_REVISION: Final = "750279a9710d3d98bbc6d0d87250312cfdf922a0"
V2_REVISION: Final = "3247b942f18d39d1fd476b1077852af033d35d8c"
LABELS: Final = {"real": REAL_LABEL, "fake": FAKE_LABEL}
EXTRA_COLUMNS: Final = ("type", "release_date", "row_id", "image_sha256")


def _shards(pattern: str, revision: str) -> tuple[HubArtifact, ...]:
    return (HubArtifact(REPO, pattern, revision=revision, streamed=True),)


class _OpenFake(ParquetShardBuilder):
    """Shared conversion of OpenFake rows; generators are unknown in ``unknown_generator_splits``."""

    unknown_generator_splits: ClassVar[frozenset[str]] = frozenset()

    @override
    def record(
        self, spec: SplitSpec, row: Mapping[str, Any], image: str
    ) -> SampleRecord:
        if row["label"] not in LABELS:
            raise ValueError(
                f"{self.name}/{spec.name}: unknown label {row['label']!r} for {image}."
            )
        label = LABELS[row["label"]]
        extra = {
            column: str(row[column]) for column in EXTRA_COLUMNS if row.get(column)
        }
        if label == REAL_LABEL:
            extra["real_source"] = row["model"]
        known_generator = spec.name not in self.unknown_generator_splits
        return SampleRecord(
            image=image,
            label=label,
            generator=row["model"] if label == FAKE_LABEL and known_generator else None,
            extra=extra,
        )


class OpenFakeV1(_OpenFake):
    """
    OpenFake v1, arXiv 2509.09495, pinned to the ``v1.0`` tag of the Hub repository.

    Politically salient real photos from LAION against images of open and
    proprietary generators, among them Stable Diffusion, Flux, Midjourney,
    DALL-E 3, Imagen and GPT Image 1, plus community fine-tunes. Rows are
    stored as Parquet shards ``data/{train,test}-*.parquet`` and converted shard
    by shard. Records keep ``type`` and ``release_date``; real images carry
    their source in ``real_source``.
    """

    name = "openfake_v1"
    description = "OpenFake v1: 1.9M politically salient real and generated images."
    split_specs = (
        SplitSpec(
            name="train",
            role=SplitRole.TRAIN,
            artifacts=_shards("data/train-*.parquet", V1_REVISION),
            description="1.87M images; about 1 TB of Parquet shards.",
        ),
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=_shards("data/test-*.parquet", V1_REVISION),
            description="60k images of the training generators.",
        ),
    )
    default_train_splits = ("train",)


class OpenFakeV2(_OpenFake):
    """
    OpenFake v2 at the repaired release of the ``main`` branch.

    Training reals come from LAION and Pexels. The ``test`` split holds out
    both the generators and the real sources, DOCCI and ImageNet, so it
    measures generalization to unseen generators. ``reddit`` is an in-the-wild
    set from AI-generation and photography subreddits, where the generator of
    a fake is unknown. Records keep ``type``, ``release_date``, ``row_id`` and
    ``image_sha256``; real images carry their source in ``real_source``.
    Several v2 test generators appear in the v1 training split, so do not
    test v1-trained models on v2.
    """

    name = "openfake_v2"
    description = (
        "OpenFake v2: 2.5M images, an out-of-distribution test and a Reddit set."
    )
    split_specs = (
        SplitSpec(
            name="train",
            role=SplitRole.TRAIN,
            artifacts=_shards("core/train-*.parquet", V2_REVISION),
            description="2.3M images; about 3.3 TB of Parquet shards.",
        ),
        SplitSpec(
            name="validation",
            role=SplitRole.VAL,
            artifacts=_shards("core/validation-*.parquet", V2_REVISION),
            description="58k held-out images of the training generators and real sources.",
        ),
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=_shards("core/test-*.parquet", V2_REVISION),
            description="91k images of unseen generators against DOCCI and ImageNet reals.",
        ),
        SplitSpec(
            name="reddit",
            role=SplitRole.TEST,
            artifacts=_shards("reddit/test-*.parquet", V2_REVISION),
            description="36k in-the-wild images from subreddits; generators unknown.",
        ),
    )
    default_train_splits = ("train",)
    default_val_splits = ("validation",)
    unknown_generator_splits = frozenset({"reddit"})
