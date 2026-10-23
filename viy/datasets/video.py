from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from viy.common import ImageArray


def frame_count(path: Path) -> int:
    """Number of frames recorded in the container of the video at ``path``."""
    capture = _open(path)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def inner_frame_indices(total: int, count: int) -> list[int]:
    """
    ``count`` evenly spaced frame indices that skip the first and the last frame.

    Image-to-video generators start from a real frame, so the first frames of
    a generated video are the least representative of the generator.

    Args:
        total (int): Number of frames in the video.
        count (int): Number of frames to pick.

    Returns:
        list[int]: Increasing indices ``step * (i + 1)`` with ``step = total // (count + 1)``.
    """
    step = total // (count + 1)
    if step == 0:
        raise ValueError(f"A video of {total} frames is too short for {count} frames.")
    return [step * (index + 1) for index in range(count)]


def read_frames(path: Path, indices: Sequence[int]) -> list[ImageArray]:
    """
    Decode the frames ``indices`` of the video at ``path`` as RGB arrays.

    The video is decoded sequentially, which is exact for any codec, unlike
    seeking to a frame.

    Args:
        path (Path): Video file.
        indices (Sequence[int]): Increasing frame indices.

    Returns:
        list[ImageArray]: Frames in the order of ``indices``.
    """
    capture = _open(path)
    wanted = set(indices)
    frames: dict[int, ImageArray] = {}
    for index in range(max(indices) + 1):
        if not capture.grab():
            capture.release()
            raise ValueError(
                f"{path}: the video ends at frame {index}, before frame {max(indices)}."
            )
        if index in wanted:
            _, frame = capture.retrieve()
            frames[index] = np.asarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
            )
    capture.release()
    return [frames[index] for index in indices]


def _open(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open the video {path}.")
    return capture
