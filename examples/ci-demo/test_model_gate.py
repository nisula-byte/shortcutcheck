"""The release gate as an ordinary test (roadmap 4.2 demo).

The model under test is named in MODEL (``robust`` or ``shortcut``), as a CI job would set it.
"""

from __future__ import annotations

import os

import pytest

MODEL = os.environ.get("DEMO_MODEL", "robust")


@pytest.mark.shortcutcheck(
    model=f"models/{MODEL}.joblib",
    data="data/apps.csv",
    train="data/train.csv",
    mrs="core",
    threshold=0.05,
)
def test_detector_has_no_metadata_shortcuts(shortcutcheck_report):
    shortcutcheck_report.assert_passed()
