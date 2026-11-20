"""Roadmap step 2.3: MR5/MR6 grounded in official protection levels, with property tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shortcutcheck.mrs import (
    AddDangerousPermission,
    AddNormalPermission,
    NotApplicableError,
    changed_columns,
    secondary_mrs,
)
from shortcutcheck.mrs.permissions import (
    BY_COLUMN,
    LEVELS,
    PREFIX,
    mr5_columns,
    mr6_columns,
    render_markdown,
)
from shortcutcheck.runner import MRRun, run_mr

P = PREFIX
PERMS = [p.column for p in LEVELS]


def test_mapping_covers_every_dada_permission_once() -> None:
    assert len(LEVELS) == 34 and len(BY_COLUMN) == 34
    assert all(c.startswith(P) for c in PERMS)


def test_eligibility_follows_the_official_level_only() -> None:
    for c in mr5_columns():
        p = BY_COLUMN[c]
        assert p.level.split("|")[0] == "normal" and not p.deprecated and not p.removed
    for c in mr6_columns():
        assert BY_COLUMN[c].level.split("|")[0] == "dangerous" and not BY_COLUMN[c].removed
    assert not set(mr5_columns()) & set(mr6_columns())
    # Spot checks against AOSP main.
    assert P + "INTERNET" in mr5_columns() and P + "CAMERA" in mr6_columns()
    excluded = set(PERMS) - set(mr5_columns()) - set(mr6_columns())
    for name in (
        "GET_TASKS",  # normal but deprecated
        "WRITE_SMS",  # removed
        "READ_HISTORY_BOOKMARKS",  # not defined any more
        "SYSTEM_ALERT_WINDOW",  # signature|appop...
        "REQUEST_INSTALL_PACKAGES",
        "INSTALL_PACKAGES",
        "REBOOT",
        "INJECT_EVENTS",
        "MOUNT_FORMAT_FILESYSTEMS",
    ):
        assert P + name in excluded


def test_mapping_table_has_a_source_link_per_feature() -> None:
    text = render_markdown()
    for p in LEVELS:
        row = next(line for line in text.splitlines() if line.startswith(f"| `{p.name}` |"))
        assert (
            f"(https://developer.android.com/reference/android/Manifest.permission#{p.name})" in row
        )
    assert "android.googlesource.com" in text


@st.composite
def permission_frames(draw: st.DrawFn, n_min: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    n = draw(st.integers(n_min, 60))
    cols = {c: draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)) for c in PERMS}
    cols["api_calls"] = draw(st.lists(st.integers(0, 100), min_size=n, max_size=n))
    x = pd.DataFrame(cols).astype("int64")
    return x, pd.Series(np.arange(n) % 2)


def with_known_pools(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Make one normal permission goodware-common and frequent, so both MRs can be fitted."""
    x = x.copy()
    x[P + "ACCESS_NETWORK_STATE"] = (y == 0).astype("int64")
    x[P + "CAMERA"] = 1
    x.loc[x.index[:2], P + "CAMERA"] = 0
    return x


PROPERTY = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY
@given(data=permission_frames(), test=permission_frames(n_min=1), seed=st.integers(0, 2**32 - 1))
def test_mr5_mr6_add_exactly_one_eligible_permission(
    data: tuple[pd.DataFrame, pd.Series], test: tuple[pd.DataFrame, pd.Series], seed: int
) -> None:
    x, y = data
    x = with_known_pools(x, y)
    xt = test[0]
    predicted = np.random.default_rng(seed).integers(0, 2, len(xt))
    for mr in secondary_mrs(list(x.columns), min_apps=2):
        mr.fit(x, y)
        after = mr.transform(xt, rng=np.random.default_rng(seed), predicted=predicted)
        assert changed_columns(xt, after) <= set(mr.pool)
        diff = after[list(mr.columns_touched)].to_numpy() - xt[list(mr.columns_touched)].to_numpy()
        assert set(np.unique(diff)) <= {0, 1}  # only ever 0 -> 1
        assert (diff.sum(axis=1) <= 1).all()  # at most one permission per app
        assert mr.is_valid(after).all()
        if mr.mr_id == "MR5":
            assert (diff.sum(axis=1)[predicted == 0] == 0).all()  # goodware rows untouched
        # A row is left alone only if it already holds every pool permission it could get.
        could = (xt[mr.pool].to_numpy() == 0).any(axis=1)
        if mr.mr_id == "MR5":
            could &= predicted == 1
        assert (diff.sum(axis=1) == could.astype(int)).all()


def test_mr5_keeps_only_frequent_goodware_common_normal_permissions() -> None:
    n = 100
    y = pd.Series([0] * 50 + [1] * 50)
    x = pd.DataFrame({c: 0 for c in PERMS}, index=range(n))
    x[P + "ACCESS_NETWORK_STATE"] = (y == 0).astype(int)  # goodware-common
    x[P + "CHANGE_WIFI_STATE"] = (y == 1).astype(int)  # malware-common
    x.loc[:4, P + "NFC"] = 1  # goodware-only but held by 5 apps
    mr = AddNormalPermission(list(x.columns), min_apps=10).fit(x, y)
    assert mr.pool == [P + "ACCESS_NETWORK_STATE"]
    with pytest.raises(NotApplicableError, match="training labels"):
        AddNormalPermission(list(x.columns)).fit(x)


def test_mr6_needs_a_permission_seen_in_training() -> None:
    x = pd.DataFrame({c: 0 for c in PERMS}, index=range(20))
    with pytest.raises(NotApplicableError, match="held by 10"):
        AddDangerousPermission(list(x.columns)).fit(x)


def test_no_permission_columns_means_no_secondary_mrs() -> None:
    assert secondary_mrs(["year", "apk_size"]) == []


class ScoresCamera:
    """Calls an app goodware when it requests CAMERA: a shortcut MR6 should expose."""

    feature_names_in_ = np.array([P + "CAMERA", P + "SEND_SMS"])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.where(x[P + "CAMERA"] == 1, 0.1, 0.9)
        return np.column_stack([1 - p, p])


def test_mr6_counts_only_malware_to_goodware_flips() -> None:
    x = pd.DataFrame({c: 0 for c in PERMS}, index=range(40))
    x.loc[:19, P + "CAMERA"] = 1
    x[P + "SEND_SMS"] = 1
    mr = AddDangerousPermission([P + "CAMERA", P + "SEND_SMS"], min_apps=5).fit(x)
    run = run_mr(ScoresCamera(), mr, x, seed=3)
    assert run.direction == "to_goodware"
    # Rows 20-39 lack CAMERA, are predicted malware, and gain CAMERA: every one flips.
    assert run.n_counted == 20 and run.violation_rate == 1.0


def test_direction_on_a_hand_computed_example() -> None:
    kw = {
        "mr_id": "MR6",
        "p_before": np.array([0.9, 0.2, 0.9, 0.2]),
        "p_after": np.array([0.1, 0.8, 0.95, 0.1]),
        "perturbed": np.ones(4, bool),
        "valid": np.ones(4, bool),
    }
    # Flips: row 0 (malware -> goodware) and row 1 (goodware -> malware).
    assert MRRun(**kw).n_violations == 2  # type: ignore[arg-type]
    assert MRRun(**kw, direction="to_goodware").n_violations == 1  # type: ignore[arg-type]
