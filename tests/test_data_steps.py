"""Steps 1.3-1.5: registration, raw-data protection, label parsing, audit, dex_date validity."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from shortcutcheck.config import Config
from shortcutcheck.data import audit
from shortcutcheck.data.dada import discover_sets, read_labels
from shortcutcheck.data.register import register_zip, sha256_of
from shortcutcheck.data.safe_io import RawDataWriteError, guard_write, write_text
from shortcutcheck.data.validity import check_dex_date, decision, to_year

HASH_A = "a" * 64
HASH_B = "b" * 64


# --- 1.3 -----------------------------------------------------------------------------------


def test_register_records_checksum(project: Config, synthetic_zip: Path) -> None:
    data_md = (project.root / "DATA.md").read_text()
    assert sha256_of(synthetic_zip) in data_md


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_raw_files_are_read_only(project: Config) -> None:
    files = [p for p in project.raw_dir.rglob("*") if p.is_file()]
    assert files
    assert all(not (p.stat().st_mode & stat.S_IWUSR) for p in files)


def test_register_refuses_second_extraction(project: Config, synthetic_zip: Path) -> None:
    with pytest.raises(FileExistsError):
        register_zip(synthetic_zip, project.raw_dir, project.root / "DATA.md")


def test_guard_refuses_raw_paths(project: Config) -> None:
    with pytest.raises(RawDataWriteError):
        guard_write(project.raw_dir / "anything.csv", project.raw_root)
    with pytest.raises(RawDataWriteError):
        write_text(project.raw_root / "x.txt", "no", project.raw_root)
    assert write_text(project.root / "ok.txt", "yes", project.raw_root).exists()


# --- label files ---------------------------------------------------------------------------


@pytest.mark.parametrize("header", [True, False])
def test_read_labels_with_and_without_header(tmp_path: Path, header: bool) -> None:
    path = tmp_path / "x.goodmal.csv"
    body = f"{HASH_A},0\n{HASH_B},1\n"
    path.write_text(("sha256,goodmal\n" if header else "") + body)
    labels = read_labels(path)
    assert labels.set_index("sha256")["label"].to_dict() == {HASH_A: 0, HASH_B: 1}


def test_read_labels_rejects_unknown_values(tmp_path: Path) -> None:
    path = tmp_path / "x.goodmal.csv"
    path.write_text(f"{HASH_A},maybe\n")
    with pytest.raises(ValueError, match="unrecognised"):
        read_labels(path)


# --- 1.4 -----------------------------------------------------------------------------------


def test_audit_passes_when_both_years_have_enough(project: Config) -> None:
    sets = discover_sets(project.raw_dir)
    assert {s.name for s in sets} == {"SYN-DR-AG", "SYN_ACT14", "SYN_ACT17"}
    summaries, verdicts = audit.run_audit(
        sets, project.id_column, project.rq3_candidates, project.rq3_min_per_class_per_year
    )
    by_name = {v.candidate: v for v in verdicts}
    assert by_name["AndroCT"].passes
    assert not by_name["AZL"].passes  # no matching sets in the synthetic data
    text = audit.render_markdown(summaries, verdicts, "synthetic")
    assert "**Check: PASS.** RQ3 can use: AndroCT." in text


def test_audit_fails_below_minimum(project: Config) -> None:
    sets = discover_sets(project.raw_dir)
    _, verdicts = audit.run_audit(sets, project.id_column, project.rq3_candidates, 10**6)
    assert not any(v.passes for v in verdicts)
    assert "VirusShare-per-year fallback" in audit.render_markdown([], verdicts, "x")


# --- 1.5 -----------------------------------------------------------------------------------


def test_to_year_handles_mixed_encodings() -> None:
    values = pd.Series([2014, 0, "2016-03-01 10:00:00", 1420070400, 1420070400000, None, "junk"])
    years = to_year(values).tolist()
    assert years[0] == 2014
    assert pd.isna(years[1])
    assert years[2] == 2016
    assert years[3] == 2015 and years[4] == 2015
    assert pd.isna(years[5]) and pd.isna(years[6])


def test_dex_date_counts_and_decision() -> None:
    values = pd.Series([2014, 2015, 0, 1980, 2030, 2016, 2013, 2012, 2011, 2010])
    report = check_dex_date(values, first_year=2008, last_year=2021)
    assert (report.missing_or_zero, report.before_android, report.after_collection) == (1, 1, 1)
    assert report.exactly_1980 == 1
    assert report.invalid_share == pytest.approx(0.3)
    assert decision(report, 0.20).startswith("DROP")
    assert decision(report, 0.50).startswith("KEEP")
