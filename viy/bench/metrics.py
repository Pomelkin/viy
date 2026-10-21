from typing import Final

import numpy as np
import numpy.typing as npt
from kostyl import StrictPydanticBaseModel
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL


type Labels = npt.NDArray[np.int64]
type Scores = npt.NDArray[np.float64]

LOW_FPR_TARGETS: Final = (0.01, 0.001)


class BinaryMetrics(StrictPydanticBaseModel):
    """
    Detection metrics of one set of images.

    Threshold-free metrics are ``None`` when the set holds a single class.

    Attributes:
        num_samples (int): Number of images.
        num_real (int): Number of real images.
        num_fake (int): Number of generated images.
        auc (float | None): ROC AUC.
        average_precision (float | None): Average precision with fakes as positives.
        accuracy (float): Accuracy at the threshold.
        balanced_accuracy (float | None): Mean of the true positive and true negative rates.
        f1 (float | None): F1 score of the fake class at the threshold.
        fpr (float | None): Share of real images flagged as generated.
        fnr (float | None): Share of generated images missed.
        tpr_at_1pct_fpr (float | None): Detection rate at 1% false positives.
        tpr_at_01pct_fpr (float | None): Detection rate at 0.1% false positives.
        mean_score_real (float | None): Mean score of real images.
        mean_score_fake (float | None): Mean score of generated images.
    """

    num_samples: int
    num_real: int
    num_fake: int
    auc: float | None
    average_precision: float | None
    accuracy: float
    balanced_accuracy: float | None
    f1: float | None
    fpr: float | None
    fnr: float | None
    tpr_at_1pct_fpr: float | None
    tpr_at_01pct_fpr: float | None
    mean_score_real: float | None
    mean_score_fake: float | None


def binary_metrics(labels: Labels, scores: Scores, threshold: float) -> BinaryMetrics:
    """
    Compute detection metrics.

    Args:
        labels (Labels): ``0`` for real images, ``1`` for generated ones.
        scores (Scores): Higher means more likely generated.
        threshold (float): Images with ``score >= threshold`` are flagged as generated.

    Returns:
        BinaryMetrics: Metrics of the set.
    """
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError(
            f"Expected equal non-empty shapes, got {labels.shape} and {scores.shape}."
        )
    real = labels == REAL_LABEL
    fake = labels == FAKE_LABEL
    flagged = scores >= threshold
    fpr = _mean(flagged[real])
    fnr = _mean(~flagged[fake])
    both = bool(real.any() and fake.any())
    tpr_at = _tpr_at_fpr(labels, scores) if both else (None, None)
    precision = _mean(fake[flagged])
    recall = None if fnr is None else 1.0 - fnr
    return BinaryMetrics(
        num_samples=int(labels.size),
        num_real=int(real.sum()),
        num_fake=int(fake.sum()),
        auc=float(roc_auc_score(labels, scores)) if both else None,
        average_precision=float(average_precision_score(labels, scores))
        if both
        else None,
        accuracy=float((flagged == fake).mean()),
        balanced_accuracy=None if fpr is None or fnr is None else 1.0 - (fpr + fnr) / 2,
        f1=_f1(precision, recall),
        fpr=fpr,
        fnr=fnr,
        tpr_at_1pct_fpr=tpr_at[0],
        tpr_at_01pct_fpr=tpr_at[1],
        mean_score_real=_mean(scores[real]),
        mean_score_fake=_mean(scores[fake]),
    )


def _mean(values: npt.NDArray[np.bool_] | Scores) -> float | None:
    return float(values.mean()) if values.size else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if recall is None:
        return None
    if precision is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _tpr_at_fpr(labels: Labels, scores: Scores) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    low, lower = (float(tpr[fpr <= target].max()) for target in LOW_FPR_TARGETS)
    return low, lower
