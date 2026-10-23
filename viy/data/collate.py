from collections.abc import Sequence
from typing import TypedDict

import torch

from viy.augs import AppliedOp

from .dataset import DetectionSample


class DetectionBatch(TypedDict):
    """
    A batch of detection samples.

    Attributes:
        images (torch.Tensor): Preprocessed images of shape ``(B, 3, H, W)``.
        labels (torch.Tensor): ``int64`` labels of shape ``(B,)``, ``0`` real and ``1`` generated.
        sources (list[str]): ``<dataset>/<split>`` of every image.
        indices (torch.Tensor): ``int64`` position of every record in its
            ``SplitDataset.records``, shape ``(B,)``.
        paths (list[str]): Image path of every sample relative to its dataset directory.
        augmentations (list[tuple[AppliedOp, ...]]): Applied augmentations of every image.
    """

    images: torch.Tensor
    labels: torch.Tensor
    sources: list[str]
    indices: torch.Tensor
    paths: list[str]
    augmentations: list[tuple[AppliedOp, ...]]


class DetectionCollator:
    """
    Collates :class:`DetectionSample` items into a :class:`DetectionBatch`.

    ``DetectionDataset.dataloader`` sets it up. For a DataLoader of your own,
    pass an instance as ``collate_fn``: augmentation traces have different
    lengths, so the default collate function of PyTorch cannot batch them.

    Example:
        >>> loader = DataLoader(train, batch_size=64, collate_fn=DetectionCollator())
    """

    def __call__(self, samples: Sequence[DetectionSample]) -> DetectionBatch:
        """
        Stack images and labels and keep per-sample metadata as lists.

        Args:
            samples (Sequence[DetectionSample]): Samples of one batch.

        Returns:
            DetectionBatch: The batch.

        Raises:
            ValueError: If the batch is empty or the images differ in shape.
        """
        if len(samples) == 0:
            raise ValueError("Cannot collate an empty batch.")
        shapes = {tuple(sample["image"].shape) for sample in samples}
        if len(shapes) > 1:
            raise ValueError(
                f"Images of one batch must have equal shapes, got {sorted(shapes)}; "
                "check the preprocessing output size."
            )
        return DetectionBatch(
            images=torch.stack([sample["image"] for sample in samples]),
            labels=torch.tensor(
                [sample["label"] for sample in samples], dtype=torch.int64
            ),
            sources=[sample["source"] for sample in samples],
            indices=torch.tensor(
                [sample["index"] for sample in samples], dtype=torch.int64
            ),
            paths=[sample["path"] for sample in samples],
            augmentations=[sample["augmentations"] for sample in samples],
        )
