import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from kostyl.utils import setup_logger


logger = setup_logger(fmt="detailed")


@dataclass(frozen=True, slots=True)
class HubArtifact:
    """
    Files of a Hugging Face dataset repository needed to build a split.

    Attributes:
        repo_id (str): Dataset repository, e.g. ``deepfakesMSU/NTIRE-RobustAIGenDetection-val``.
        pattern (str): Glob of the repository files to download.
        subdir (str): Subdirectory of the raw directory the files are placed in.
        extract (bool): Unpack downloaded zip archives in place and delete them.
    """

    repo_id: str
    pattern: str
    subdir: str = ""
    extract: bool = False


def fetch_artifact(artifact: HubArtifact, raw_dir: Path) -> None:
    """
    Download an artifact into ``raw_dir`` unless it was fetched before.

    Args:
        artifact (HubArtifact): Files to download.
        raw_dir (Path): Raw directory of the dataset.
    """
    target = raw_dir / artifact.subdir
    marker = target / f".fetched-{re.sub(r'[^A-Za-z0-9]+', '_', artifact.pattern)}"
    if marker.is_file():
        return
    target.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading '{artifact.pattern}' of {artifact.repo_id} into {target}")
    snapshot_download(
        repo_id=artifact.repo_id,
        repo_type="dataset",
        allow_patterns=[artifact.pattern],
        local_dir=target,
    )
    if artifact.extract:
        for archive in sorted(target.glob(artifact.pattern)):
            if archive.suffix == ".zip":
                logger.info(f"Extracting {archive}")
                with zipfile.ZipFile(archive) as handle:
                    handle.extractall(target)
                archive.unlink()
    marker.touch()
    return
