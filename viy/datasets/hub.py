import re
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import get_token
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from kostyl.utils import setup_logger

from viy.progress import track


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
        revision (str | None): Commit, branch or tag to download; ``None`` follows ``main``.
        gated (bool): The repository needs an accepted access request and a token;
            :meth:`DatasetBuilder.check_available` checks both before any download.
        streamed (bool): Files are fetched one at a time by the dataset builder
            while it converts them, instead of all at once before preparation.
    """

    repo_id: str
    pattern: str
    subdir: str = ""
    extract: bool = False
    revision: str | None = None
    gated: bool = False
    streamed: bool = False


def check_access(repo_id: str) -> None:
    """
    Make sure the current Hugging Face credentials can read the dataset ``repo_id``.

    The token comes from the ``HF_TOKEN`` environment variable or from
    ``hf auth login``.

    Raises:
        PermissionError: If no Hugging Face token is configured.
        huggingface_hub.errors.GatedRepoError: If access to the dataset was not granted.
    """
    if get_token() is None:
        raise PermissionError(
            f"'{repo_id}' is a gated dataset on the Hugging Face Hub. Request access at "
            f"https://huggingface.co/datasets/{repo_id}, then log in with `hf auth login` "
            "or set the HF_TOKEN environment variable."
        )
    HfApi().auth_check(repo_id, repo_type="dataset")
    return


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
        revision=artifact.revision,
        allow_patterns=[artifact.pattern],
        local_dir=target,
    )
    if artifact.extract:
        for archive in sorted(target.glob(artifact.pattern)):
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as handle:
                    for member in track(
                        handle.infolist(), f"Extracting {archive.name}"
                    ):
                        handle.extract(member, target)
                archive.unlink()
    marker.touch()
    return


def list_artifact_files(artifact: HubArtifact) -> list[str]:
    """Repository paths matching the artifact pattern at its revision, sorted."""
    files = HfApi().list_repo_files(
        artifact.repo_id, repo_type="dataset", revision=artifact.revision
    )
    return sorted(path for path in files if fnmatch(path, artifact.pattern))


def download_artifact_file(artifact: HubArtifact, path: str, raw_dir: Path) -> Path:
    """
    Download one file of an artifact into ``raw_dir``.

    Args:
        artifact (HubArtifact): Artifact the file belongs to.
        path (str): Repository path of the file.
        raw_dir (Path): Raw directory of the dataset.

    Returns:
        Path: Local copy of the file.
    """
    return Path(
        hf_hub_download(
            artifact.repo_id,
            path,
            repo_type="dataset",
            revision=artifact.revision,
            local_dir=raw_dir / artifact.subdir,
        )
    )
