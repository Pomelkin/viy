import csv
from collections.abc import Iterator
from pathlib import Path
from pathlib import PurePosixPath
from typing import Final
from typing import override

from PIL import Image

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL
from viy.common import ImageArray
from viy.layout import SampleRecord

from .base import DatasetBuilder
from .base import RawSample
from .base import SplitRole
from .base import SplitSpec
from .hub import HubArtifact
from .video import frame_count
from .video import inner_frame_indices
from .video import read_frames


REPO: Final = "DF26/DF26"
REVISION: Final = "a0f2408b74048faf13b1a26e05be768789a2a29e"
METADATA: Final = "metadata/videos.csv"
FRAMES_PER_VIDEO: Final = 4
PNG_COMPRESS_LEVEL: Final = 1
LABELS: Final = {"real": REAL_LABEL, "fake": FAKE_LABEL}


def _artifact(pattern: str) -> HubArtifact:
    return HubArtifact(REPO, pattern, revision=REVISION, gated=True)


class DF26(DatasetBuilder):
    """
    DF26 talking-head deepfake videos, arXiv 2609.07369, for benchmarking only.

    The Hub repository is gated: request access on its page, then log in with
    ``hf auth login`` or set ``HF_TOKEN``. Its license allows evaluation only,
    so the dataset is refused by training and validation datasets.

    Every video becomes four frames evenly spaced inside it, skipping the
    first and the last frame, decoded once and saved as PNG under
    ``images/<video path>/frame_<index>.png``. Records carry the generator,
    the clip id shared by a real video and its fakes as ``source_id``, and the
    ``scenario``, ``video``, ``frame``, ``access`` and ``conditioning`` columns.
    """

    name = "df26"
    description = "DF26: 271 real and 2,420 generated talking-head videos, as frames."
    split_specs = (
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=(_artifact(METADATA), _artifact("real/*"), _artifact("fake/*")),
            description="2,691 videos from 10 generators, 4 frames each; gated on the Hub.",
        ),
    )
    benchmark_only = True

    @override
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        for row in _read_videos(raw_dir / METADATA):
            video = raw_dir / row["video_path"]
            indices = inner_frame_indices(frame_count(video), FRAMES_PER_VIDEO)
            stem = PurePosixPath(row["video_path"]).with_suffix("")
            images = [
                str(PurePosixPath("images", stem, f"frame_{index:04d}.png"))
                for index in indices
            ]
            targets = [self.layout.dir / image for image in images]
            if not all(target.is_file() for target in targets):
                frames = read_frames(video, indices)
                for target, frame in zip(targets, frames, strict=True):
                    _save_png(target, frame)
            for index, image, target in zip(indices, images, targets, strict=True):
                yield RawSample(target, _record(row, image, index))
        return


def _save_png(target: Path, frame: ImageArray) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    Image.fromarray(frame).save(
        temporary, format="PNG", compress_level=PNG_COMPRESS_LEVEL
    )
    temporary.replace(target)
    return


def _record(row: dict[str, str], image: str, frame: int) -> SampleRecord:
    label = LABELS[row["label"]]
    extra = {
        "scenario": row["scenario"],
        "video": row["video_path"],
        "frame": str(frame),
    }
    if label == FAKE_LABEL:
        extra["access"] = row["generator_access"]
        extra["conditioning"] = row["conditioning"]
    return SampleRecord(
        image=image,
        label=label,
        generator=row["generator"] if label == FAKE_LABEL else None,
        source_id=row["clip_id"],
        extra=extra,
    )


def _read_videos(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"DF26 metadata not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    unknown = sorted({row["label"] for row in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"{path}: unknown labels {unknown}.")
    return rows
