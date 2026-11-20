"""Roadmap step 2.2 Check: property-based tests prove each MR changes only its declared
columns and never leaves the observed training range. Also covers the 2.1 interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shortcutcheck.mrs import (
    MetadataSwap,
    MetamorphicRelation,
    NotApplicableError,
    NotFittedError,
    SdkShift,
    SizeScale,
    YearShift,
    changed_columns,
    changed_rows,
    core_mrs,
)

VALID = (2008, 2021)
META = ["year", "minSdkVersion", "targetSdkVersion", "apk_size"]
YEARS = [1980, *range(2008, 2022), 2107]  # 1980 and 2107 are the invalid values seen in Dada


@st.composite
def app_frames(
    draw: st.DrawFn, min_rows: int = 6, sdk_known: bool = True, anchor: bool = True
) -> pd.DataFrame:
    n = draw(st.integers(min_rows, 40))
    years = draw(st.lists(st.sampled_from(YEARS), min_size=n, max_size=n))
    sizes = draw(st.lists(st.integers(5_000, 200_000_000), min_size=n, max_size=n))
    if sdk_known:
        mins = draw(st.lists(st.integers(1, 30), min_size=n, max_size=n))
        extra = draw(st.lists(st.integers(0, 10), min_size=n, max_size=n))
        targets = [m + e for m, e in zip(mins, extra, strict=True)]
    else:
        mins = targets = [0] * n
    frame = pd.DataFrame(
        {
            "year": years,
            "minSdkVersion": mins,
            "targetSdkVersion": targets,
            "apk_size": sizes,
            "android.permission.SEND_SMS": draw(
                st.lists(st.integers(0, 1), min_size=n, max_size=n)
            ),
            "api_call_count": draw(st.lists(st.integers(0, 500), min_size=n, max_size=n)),
        }
    )
    if anchor:  # training frames: two valid years and both classes, so every MR can be fitted
        frame.loc[0, "year"], frame.loc[1, "year"] = 2012, 2017
        if sdk_known:
            frame.loc[0, ["minSdkVersion", "targetSdkVersion"]] = [8, 19]
            frame.loc[1, ["minSdkVersion", "targetSdkVersion"]] = [14, 23]
    return frame.astype("int64")


def labels_for(frame: pd.DataFrame) -> pd.Series:
    y = pd.Series(np.arange(len(frame)) % 2, index=frame.index)
    return y


def fitted(frame: pd.DataFrame) -> list[MetamorphicRelation]:
    mrs = core_mrs(list(frame.columns), VALID)
    for mr in mrs:
        mr.fit(frame, labels_for(frame))
    return mrs


TEST_FRAMES = app_frames(min_rows=1, anchor=False)
SEEDS = st.integers(0, 2**32 - 1)
PROPERTY = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY
@given(train=app_frames(), test=TEST_FRAMES, seed=SEEDS)
def test_every_core_mr_changes_only_its_declared_columns(
    train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> None:
    for mr in fitted(train):
        before = test.copy()
        predicted = np.random.default_rng(seed).integers(0, 2, len(test))
        after = mr.transform(test, rng=np.random.default_rng(seed), predicted=predicted)
        assert changed_columns(test, after) <= set(mr.columns_touched), mr.mr_id
        pd.testing.assert_frame_equal(test, before)  # input never modified
        assert list(after.dtypes) == list(test.dtypes)


@PROPERTY
@given(train=app_frames(), test=TEST_FRAMES, seed=SEEDS)
def test_every_core_mr_stays_inside_the_observed_training_range(
    train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> None:
    for mr in fitted(train):
        predicted = np.random.default_rng(seed).integers(0, 2, len(test))
        after = mr.transform(test, rng=np.random.default_rng(seed), predicted=predicted)
        moved = changed_rows(test, after, mr.columns_touched)
        for col in mr.columns_touched:
            observed = mr.ranges[col]
            values = after.loc[moved, col]
            assert ((values >= observed.low) & (values <= observed.high)).all(), (mr.mr_id, col)
        assert mr.is_valid(after)[moved].all(), mr.mr_id


@PROPERTY
@given(train=app_frames(), test=TEST_FRAMES, seed=SEEDS)
def test_mr1_always_moves_to_a_different_valid_year(
    train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> None:
    mr = YearShift("year", VALID).fit(train)
    after = mr.transform(test, rng=np.random.default_rng(seed))
    assert set(after["year"]) <= set(mr.valid_years)  # never creates 1980 or 2107
    assert (after["year"] != test["year"]).all()
    assert (after["year"] >= VALID[0]).all() and (after["year"] <= VALID[1]).all()


@PROPERTY
@given(train=app_frames(), test=TEST_FRAMES, seed=SEEDS)
def test_mr4_copies_a_real_training_app_of_the_other_class(
    train: pd.DataFrame, test: pd.DataFrame, seed: int
) -> None:
    mr = MetadataSwap(META, "year", VALID).fit(train, labels_for(train))
    predicted = np.random.default_rng(seed).integers(0, 2, len(test))
    after = mr.transform(test, rng=np.random.default_rng(seed), predicted=predicted)
    y = labels_for(train)
    for i in range(len(test)):
        donor_class = 1 - predicted[i]
        pool = train.loc[y == donor_class, META]
        pool = pool[(pool["year"] >= VALID[0]) & (pool["year"] <= VALID[1])]
        row = after.iloc[i][META].to_numpy()
        assert (pool.to_numpy() == row).all(axis=1).any()


@PROPERTY
@given(train=app_frames(), seed=SEEDS)
def test_same_seed_gives_the_same_perturbation(train: pd.DataFrame, seed: int) -> None:
    for mr in fitted(train):
        predicted = np.zeros(len(train), dtype=int)
        a = mr.transform(train, rng=np.random.default_rng(seed), predicted=predicted)
        b = mr.transform(train, rng=np.random.default_rng(seed), predicted=predicted)
        pd.testing.assert_frame_equal(a, b)


def test_mr3_scales_between_half_and_double_before_clipping() -> None:
    train = pd.DataFrame({"apk_size": [1, 10**9]})
    test = pd.DataFrame({"apk_size": [10**6] * 500})
    after = SizeScale().fit(train).transform(test, rng=np.random.default_rng(0))
    ratio = after["apk_size"] / test["apk_size"]
    assert ratio.between(0.5, 2.0).all()
    assert (ratio < 1).any() and (ratio > 1).any()


def test_mr2_is_not_applicable_when_sdk_versions_are_missing() -> None:
    # Dada stores unknown SDK versions as 0; in every Dada split over 99.9% are 0 (D10).
    frame = pd.DataFrame({"minSdkVersion": [0] * 99 + [8], "targetSdkVersion": [0] * 99 + [19]})
    with pytest.raises(NotApplicableError, match="known SDK versions"):
        SdkShift().fit(frame)


def test_mr2_draws_observed_pairs_when_sdk_versions_are_known() -> None:
    train = pd.DataFrame({"minSdkVersion": [8, 14, 21, 0], "targetSdkVersion": [19, 23, 28, 0]})
    after = SdkShift().fit(train).transform(train.iloc[:3], rng=np.random.default_rng(1))
    pairs = {(8, 19), (14, 23), (21, 28)}
    got = set(map(tuple, after.to_numpy().tolist()))
    assert got <= pairs
    assert (after.to_numpy() != train.iloc[:3].to_numpy()).any(axis=1).all()


def test_mr1_needs_two_valid_training_years() -> None:
    with pytest.raises(NotApplicableError, match="two valid years"):
        YearShift().fit(pd.DataFrame({"year": [1980, 2012, 2107]}))


def test_mr4_needs_labels_and_predictions() -> None:
    frame = pd.DataFrame({"year": [2012, 2017], "apk_size": [10, 20]})
    with pytest.raises(NotApplicableError, match="training labels"):
        MetadataSwap(["year", "apk_size"]).fit(frame)
    mr = MetadataSwap(["year", "apk_size"]).fit(frame, pd.Series([0, 1]))
    with pytest.raises(ValueError, match="predicted class"):
        mr.transform(frame, rng=np.random.default_rng(0))


def test_mr4_never_uses_donors_with_invalid_years() -> None:
    train = pd.DataFrame({"year": [1980, 2012, 1980, 2017], "apk_size": [1, 2, 3, 4]})
    mr = MetadataSwap(["year", "apk_size"]).fit(train, pd.Series([0, 0, 1, 1]))
    after = mr.transform(train, rng=np.random.default_rng(0), predicted=np.array([1, 1, 0, 0]))
    assert after["year"].tolist() == [2012, 2012, 2017, 2017]


def test_interface_errors() -> None:
    with pytest.raises(NotFittedError):
        YearShift().transform(pd.DataFrame({"year": [2012]}), rng=np.random.default_rng(0))
    with pytest.raises(NotApplicableError, match="not in the data"):
        YearShift().fit(pd.DataFrame({"apk_size": [1]}))
    with pytest.raises(ValueError, match="straddle 1"):
        SizeScale(factor=(1.0, 2.0))


def test_core_mrs_only_builds_what_the_columns_support() -> None:
    assert [m.mr_id for m in core_mrs(META, VALID)] == ["MR1", "MR2", "MR3", "MR4"]
    assert [m.mr_id for m in core_mrs(["year", "x"], VALID)] == ["MR1", "MR4"]
    assert core_mrs(["x"], VALID) == []
    described = core_mrs(META, VALID)[0].describe()
    assert described["id"] == "MR1" and described["justification"]
