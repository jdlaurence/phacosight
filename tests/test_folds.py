"""Path remapping tests against the actual committed fold CSVs."""

from pathlib import Path

import pytest

from phacosight.data.folds import _relocate, fold_case_ids, load_fold

REPO = Path(__file__).resolve().parents[1]
FIVEFOLD = REPO / "upstream" / "TrainIDs_SemanticSegmentation_FiveFold" / "TrainIDs_Cataract_1k_Anatomy_Instruments"
FOURFOLD = REPO / "upstream" / "TrainIDs_Semantic Segmentation" / "TrainIDs_Cataract_1k_Instruments"


def test_relocate_cluster_absolute_path():
    p = _relocate(
        "/storage/homefs/ng22l920/Datasets/Cat3K_30Vids_Segmentation/"
        "30videos_FinalRevision/case_5015/img/case5015_01.png",
        Path("/data/cat1k"),
    )
    assert p == Path("/data/cat1k/case_5015/img/case5015_01.png")


def test_relocate_repo_relative_path_with_mask_override():
    p = _relocate(
        "semantic_segmentation_images_annotations/Images_and_Supervisely_Annotations/"
        "case_5014/mask_instruments/case5014_01.png",
        Path("/data/cat1k"),
        mask_dir="mask_anatomy_inst",
    )
    assert p == Path("/data/cat1k/case_5014/mask_anatomy_inst/case5014_01.png")


def test_relocate_rejects_unrecognized_path():
    with pytest.raises(ValueError):
        _relocate("some/random/file.png", Path("/data"))


@pytest.mark.parametrize("fold", range(5))
def test_fivefold_csvs_resolve_and_are_patient_disjoint(fold):
    train = load_fold(FIVEFOLD / f"Cat1k_anatomy_instrument_{fold}_train.csv", "/data/cat1k")
    test = load_fold(FIVEFOLD / f"Cat1k_anatomy_instrument_{fold}_test.csv", "/data/cat1k")
    assert len(train) > 0 and len(test) > 0
    # patient-wise separation is the whole point of these folds
    assert not (fold_case_ids(train) & fold_case_ids(test))


def test_fourfold_csvs_resolve():
    df = load_fold(FOURFOLD / "Cat1k_instrument_train_fold0.csv", "/data/cat1k")
    assert len(df) > 0
    assert all(str(p).startswith("/data/cat1k/case_") for p in df["imgs"])
