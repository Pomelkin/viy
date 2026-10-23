from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

import numpy as np
import numpy.typing as npt
from kostyl import StrictPydanticBaseModel
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL
from viy.registry import Registry


type Labels = npt.NDArray[np.int64]
type Scores = npt.NDArray[np.float64]

CALIBRATION_BINS: Final = 15


class ScoredSet:
    """
    Labels and scores of one set of images, with quantities shared by metrics.

    Fakes are the positive class, and images with ``score >= threshold`` are
    flagged as generated.
    """

    def __init__(self, labels: Labels, scores: Scores, threshold: float) -> None:
        """
        Initialize the set.

        Args:
            labels (Labels): ``0`` for real images, ``1`` for generated ones.
            scores (Scores): Higher means more likely generated.
            threshold (float): Decision threshold of threshold-dependent metrics.
        """
        if labels.shape != scores.shape or labels.size == 0:
            raise ValueError(
                f"Expected equal non-empty shapes, got {labels.shape} and {scores.shape}."
            )
        self.labels = labels
        self.scores = scores
        self.threshold = threshold
        self.real = labels == REAL_LABEL
        self.fake = labels == FAKE_LABEL
        self.flagged = scores >= threshold
        return

    @property
    def both_classes(self) -> bool:
        """Whether the set holds real and generated images."""
        return bool(self.real.any() and self.fake.any())

    @property
    def probabilities(self) -> bool:
        """Whether every score lies in ``[0, 1]``."""
        return bool(self.scores.min() >= 0.0 and self.scores.max() <= 1.0)

    @cached_property
    def roc(self) -> tuple[Scores, Scores]:
        """False and true positive rates of the ROC curve."""
        fpr, tpr, _ = roc_curve(self.labels, self.scores, drop_intermediate=False)
        return fpr, tpr


@dataclass(frozen=True, slots=True)
class Metric:
    """
    A metric of the benchmark pool.

    Attributes:
        name (str): Name used in configs and results.
        description (str): One-line meaning shown by ``viy bench metrics``.
        compute (Callable[[ScoredSet], float | None]): Value on a set of images,
            ``None`` where the metric is undefined, e.g. AUC of a single class.
        higher_is_better (bool | None): Direction used to find the worst slice;
            ``None`` marks a descriptive value that is not aggregated over slices.
        threshold (bool): Whether the value depends on the decision threshold.
    """

    name: str
    description: str
    compute: Callable[[ScoredSet], float | None]
    higher_is_better: bool | None = True
    threshold: bool = False


METRICS: Registry[Metric] = Registry("metric")


def register_metric(metric: Metric, replace: bool = False) -> None:
    """
    Add a metric to the pool, so benchmarks can select it by name.

    Args:
        metric (Metric): Metric to add.
        replace (bool): Replace a metric of the same name, e.g. when a notebook
            cell runs again.
    """
    METRICS.add(metric.name, metric, replace=replace)
    return


class SliceMetrics(StrictPydanticBaseModel):
    """
    Selected metrics of one set of images.

    Attributes:
        num_samples (int): Number of images.
        num_real (int): Number of real images.
        num_fake (int): Number of generated images.
        metrics (dict[str, float | None]): Value of every selected metric.
    """

    num_samples: int
    num_real: int
    num_fake: int
    metrics: dict[str, float | None]


def compute_metrics(
    labels: Labels, scores: Scores, threshold: float, metrics: Sequence[str]
) -> SliceMetrics:
    """
    Compute the selected metrics of a set of images.

    Args:
        labels (Labels): ``0`` for real images, ``1`` for generated ones.
        scores (Scores): Higher means more likely generated.
        threshold (float): Images with ``score >= threshold`` are flagged as generated.
        metrics (Sequence[str]): Names of metrics from :data:`METRICS`.

    Returns:
        SliceMetrics: Counts and metric values.
    """
    scored = ScoredSet(labels, scores, threshold)
    return SliceMetrics(
        num_samples=int(labels.size),
        num_real=int(scored.real.sum()),
        num_fake=int(scored.fake.sum()),
        metrics={name: METRICS.get(name).compute(scored) for name in metrics},
    )


def _mean(values: npt.NDArray[np.bool_] | Scores) -> float | None:
    return float(values.mean()) if values.size else None


def _auc(scored: ScoredSet) -> float | None:
    if not scored.both_classes:
        return None
    return float(roc_auc_score(scored.labels, scored.scores))


def _average_precision(scored: ScoredSet) -> float | None:
    if not scored.both_classes:
        return None
    return float(average_precision_score(scored.labels, scored.scores))


def _eer(scored: ScoredSet) -> float | None:
    if not scored.both_classes:
        return None
    fpr, tpr = scored.roc
    gap = (1.0 - tpr) - fpr
    crossing = int(np.argmax(gap <= 0))
    if gap[crossing] == 0:
        return float(fpr[crossing])
    share = gap[crossing - 1] / (gap[crossing - 1] - gap[crossing])
    return float(fpr[crossing - 1] + share * (fpr[crossing] - fpr[crossing - 1]))


def _tpr_at_fpr(target: float) -> Callable[[ScoredSet], float | None]:
    def compute(scored: ScoredSet) -> float | None:
        if not scored.both_classes:
            return None
        fpr, tpr = scored.roc
        return float(tpr[fpr <= target].max())

    return compute


def _fpr_at_tpr(target: float) -> Callable[[ScoredSet], float | None]:
    def compute(scored: ScoredSet) -> float | None:
        if not scored.both_classes:
            return None
        fpr, tpr = scored.roc
        return float(fpr[tpr >= target].min())

    return compute


def _fpr(scored: ScoredSet) -> float | None:
    return _mean(scored.flagged[scored.real])


def _fnr(scored: ScoredSet) -> float | None:
    return _mean(~scored.flagged[scored.fake])


def _recall(scored: ScoredSet) -> float | None:
    return _mean(scored.flagged[scored.fake])


def _precision(scored: ScoredSet) -> float | None:
    return _mean(scored.fake[scored.flagged])


def _accuracy(scored: ScoredSet) -> float | None:
    return float((scored.flagged == scored.fake).mean())


def _balanced_accuracy(scored: ScoredSet) -> float | None:
    fpr, fnr = _fpr(scored), _fnr(scored)
    if fpr is None or fnr is None:
        return None
    return 1.0 - (fpr + fnr) / 2


def _f1(scored: ScoredSet) -> float | None:
    recall, precision = _recall(scored), _precision(scored)
    if recall is None:
        return None
    if precision is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mcc(scored: ScoredSet) -> float | None:
    if not scored.both_classes:
        return None
    return float(matthews_corrcoef(scored.labels, scored.flagged.astype(np.int64)))


def _log_loss(scored: ScoredSet) -> float | None:
    if not scored.probabilities:
        return None
    return float(
        log_loss(scored.labels, scored.scores, labels=[REAL_LABEL, FAKE_LABEL])
    )


def _brier(scored: ScoredSet) -> float | None:
    if not scored.probabilities:
        return None
    return float(np.mean((scored.scores - scored.labels) ** 2))


def _ece(scored: ScoredSet) -> float | None:
    if not scored.probabilities:
        return None
    bins = np.minimum(
        (scored.scores * CALIBRATION_BINS).astype(np.int64), CALIBRATION_BINS - 1
    )
    error = 0.0
    for index in np.unique(bins):
        members = bins == index
        gap = abs(scored.labels[members].mean() - scored.scores[members].mean())
        error += members.mean() * gap
    return float(error)


def _mean_score_real(scored: ScoredSet) -> float | None:
    return _mean(scored.scores[scored.real])


def _mean_score_fake(scored: ScoredSet) -> float | None:
    return _mean(scored.scores[scored.fake])


for _metric in (
    Metric("auc", "ROC AUC", _auc),
    Metric(
        "average_precision",
        "area under the precision-recall curve, fakes as positives",
        _average_precision,
    ),
    Metric(
        "eer", "equal error rate, where FPR equals FNR", _eer, higher_is_better=False
    ),
    Metric(
        "tpr_at_5pct_fpr",
        "share of fakes detected at 5% false positives",
        _tpr_at_fpr(0.05),
    ),
    Metric(
        "tpr_at_1pct_fpr",
        "share of fakes detected at 1% false positives",
        _tpr_at_fpr(0.01),
    ),
    Metric(
        "tpr_at_01pct_fpr",
        "share of fakes detected at 0.1% false positives",
        _tpr_at_fpr(0.001),
    ),
    Metric(
        "fpr_at_95pct_tpr",
        "share of reals flagged when 95% of fakes are detected",
        _fpr_at_tpr(0.95),
        higher_is_better=False,
    ),
    Metric("accuracy", "share of correct decisions", _accuracy, threshold=True),
    Metric(
        "balanced_accuracy",
        "mean of the detection rate and the true negative rate",
        _balanced_accuracy,
        threshold=True,
    ),
    Metric(
        "precision",
        "share of flagged images that are generated",
        _precision,
        threshold=True,
    ),
    Metric(
        "recall", "share of generated images flagged, the TPR", _recall, threshold=True
    ),
    Metric("f1", "F1 score of the fake class", _f1, threshold=True),
    Metric("mcc", "Matthews correlation coefficient", _mcc, threshold=True),
    Metric(
        "fpr",
        "share of real images flagged as generated",
        _fpr,
        higher_is_better=False,
        threshold=True,
    ),
    Metric(
        "fnr",
        "share of generated images missed",
        _fnr,
        higher_is_better=False,
        threshold=True,
    ),
    Metric(
        "log_loss",
        "binary cross-entropy; needs scores in [0, 1]",
        _log_loss,
        higher_is_better=False,
    ),
    Metric(
        "brier",
        "mean squared error of the scores; needs scores in [0, 1]",
        _brier,
        higher_is_better=False,
    ),
    Metric(
        "ece",
        f"expected calibration error over {CALIBRATION_BINS} score bins; needs scores in [0, 1]",
        _ece,
        higher_is_better=False,
    ),
    Metric(
        "mean_score_real",
        "mean score of real images",
        _mean_score_real,
        higher_is_better=None,
    ),
    Metric(
        "mean_score_fake",
        "mean score of generated images",
        _mean_score_fake,
        higher_is_better=None,
    ),
):
    register_metric(_metric)

DEFAULT_METRICS: Final = (
    "auc",
    "average_precision",
    "eer",
    "tpr_at_1pct_fpr",
    "tpr_at_01pct_fpr",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "fpr",
    "fnr",
    "mean_score_real",
    "mean_score_fake",
)
