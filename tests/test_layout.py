from pathlib import Path

import pytest

from viy.layout import FileCheck
from viy.layout import LayoutError
from viy.layout import LinkMode
from viy.layout import SampleRecord
from viy.layout import check_split
from viy.layout import link_files
from viy.layout import read_records
from viy.layout import write_records


def test_records_round_trip(tmp_path: Path) -> None:
    records = [
        SampleRecord(
            image="images/a.jpg", label=0, distortions=(), distortion_scales=()
        ),
        SampleRecord(
            image="images/b.jpg",
            label=1,
            generator="hidream",
            distortions=("jpeg", "watermark", "downscale"),
            distortion_scales=(28, "pixelseal", 0.6726230195985183),
            source_id="src-1",
            extra={"domain": "texture"},
        ),
        SampleRecord(image="images/c.png", label=1),
    ]
    path = tmp_path / "splits" / "train.csv"
    write_records(path, records)
    loaded = read_records(path)
    assert loaded == records
    assert loaded[1].extra == {"domain": "texture"}
    assert loaded[0].is_distorted is False
    assert loaded[1].is_distorted is True
    assert loaded[2].is_distorted is None
    assert loaded[2].distortions is None


def test_minimal_csv_is_a_valid_split(tmp_path: Path) -> None:
    path = tmp_path / "custom.csv"
    path.write_text("image,label\nimages/x.jpg,0\nimages/y.jpg,1\n", encoding="utf-8")
    records = read_records(path)
    assert [record.label for record in records] == [0, 1]
    assert records[0].generator is None


def test_csv_errors_name_the_image(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("image,label\nimages/x.jpg,0\nimages/y.jpg,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="images/y.jpg"):
        read_records(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image": "x.jpg", "label": 3},
        {"image": "/abs/x.jpg", "label": 0},
        {"image": "../x.jpg", "label": 0},
        {
            "image": "x.jpg",
            "label": 0,
            "distortions": ("jpeg",),
            "distortion_scales": (),
        },
        {"image": "x.jpg", "label": 0, "distortions": ("jpeg",)},
    ],
)
def test_invalid_records(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SampleRecord(**kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode", list(LinkMode))
def test_link_files_is_idempotent(tmp_path: Path, mode: LinkMode) -> None:
    source = tmp_path / "raw" / "a.jpg"
    source.parent.mkdir()
    source.write_bytes(b"jpeg")
    target = tmp_path / "layout" / "images" / "a.jpg"
    link_files([(source, target)], mode)
    link_files([(source, target)], mode)
    assert target.read_bytes() == b"jpeg"
    assert target.is_symlink() == (mode == LinkMode.SYMLINK)


def test_link_files_requires_sources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        link_files([(tmp_path / "missing.jpg", tmp_path / "out.jpg")], LinkMode.SYMLINK)


def test_check_split(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"x")
    present = SampleRecord(image="images/a.jpg", label=0)
    missing = SampleRecord(image="images/b.jpg", label=1)
    check_split([present], "ds/ok", tmp_path)
    with pytest.raises(LayoutError, match="empty"):
        check_split([], "ds/empty", tmp_path)
    with pytest.raises(LayoutError, match="more than once"):
        check_split([present, present], "ds/dup", tmp_path)
    with pytest.raises(LayoutError, match="missing"):
        check_split([present, missing], "ds/missing", tmp_path)
    check_split([present, missing], "ds/unchecked", tmp_path, FileCheck.NONE)
