from abc import ABC
from abc import abstractmethod

import torch
from kostyl import StrictPydanticBaseModel
from PIL import Image
from pydantic import Field


class ModelMeta(StrictPydanticBaseModel):
    """
    Description of a benchmarked model stored with every result.

    Attributes:
        name (str): Model name; results are stored under it.
        revision (str | None): Checkpoint or code revision.
        n_parameters (int | None): Number of parameters used at inference.
        reference (str | None): Paper or repository URL.
    """

    name: str = Field(min_length=1)
    revision: str | None = None
    n_parameters: int | None = Field(default=None, ge=0)
    reference: str | None = None


class DetectorModel(ABC):
    """
    Interface a detector implements to be benchmarked.

    ``preprocess`` runs in DataLoader worker processes, so it should only
    transform the image and must not touch the network or the GPU.
    ``predict`` receives a stacked batch and runs on whatever device the model
    chooses.

    Example:
        >>> class MyDetector(DetectorModel):
        ...     def __init__(self) -> None:
        ...         self.net = load_my_net().eval().cuda()
        ...         self.prep = Preprocessor(image_size=256)
        ...
        ...     @property
        ...     def meta(self) -> ModelMeta:
        ...         return ModelMeta(name="my-detector")
        ...
        ...     def preprocess(self, image: Image.Image) -> torch.Tensor:
        ...         return self.prep(image)
        ...
        ...     def predict(self, batch: torch.Tensor) -> torch.Tensor:
        ...         return self.net(batch.cuda()).sigmoid().squeeze(1)
    """

    @property
    @abstractmethod
    def meta(self) -> ModelMeta:
        """Description of the model."""
        raise NotImplementedError

    @property
    def threshold(self) -> float:
        """Decision threshold on the scores for threshold-dependent metrics."""
        return 0.5

    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Convert one RGB image into a model input.

        Args:
            image (Image.Image): Decoded RGB image.

        Returns:
            torch.Tensor: Input tensor; tensors of a batch must have equal shapes.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of preprocessed images.

        Args:
            batch (torch.Tensor): Stacked outputs of :meth:`preprocess`.

        Returns:
            torch.Tensor: Probability that each image is generated, shape ``(B,)``.
        """
        raise NotImplementedError
