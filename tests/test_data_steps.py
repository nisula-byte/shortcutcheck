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


def test_read_labels_dada_v31_three_columns(tmp_path: Path) -> None:
    """Dada v3.1 format: sha256,malware,dataset. The label must come from 'malware'."""
    path = tmp_path / "ACT14-training.goodmal.csv"
    path.write_text(
        f"sha256,malware,dataset\n{HASH_A},1,androct-malware\n{HASH_B},0,androct-goodware\n"
    )
    labels = read_labels(path)
    assert labels.set_index("sha256")["label"].to_dict() == {HASH_A: 1, HASH_B: 0}


def test_read_labels_rejects_unknown_values(tmp_path: Path) -> None:
    path = tmp_path / "x.goodmal.csv"
    path.write_text(f"{HASH_A},maybe\n")
    with pytest.raises(ValueError, match="unrecognised"):
        read_labels(path)


# --- 1.4 -----------------------------------------------------------------------------------


def test_audit_counts_by_app_year(project: Config) -> None:
    sets = discover_sets(project.raw_dir)
    assert {s.name for s in sets} == {"SYN-DR-AG", "SYN_ACT14", "SYN_ACT17"}
    summaries, verdicts, tables = audit.run_audit(
        sets, project.id_column, project.rq3_candidates, project.rq3_min_per_class_per_year
    )
    by_name = {v.candidate: v for v in verdicts}
    syn = by_name["SYN-2014-vs-2017"]
    assert syn.passes
    assert syn.sources == ["SYN_ACT14", "SYN_ACT17"]
    assert not by_name["SYN-missing"].passes  # no matching sets
    text = audit.render_markdown(summaries, verdicts, "synthetic", tables)
    assert "**Check: PASS for SYN-2014-vs-2017.**" in text
    assert "### Apps per year: SYN_ACT14, SYN_ACT17" in text


def test_audit_ignores_set_names_and_uses_app_year(tmp_path: Path) -> None:
    """Dada's ACT14/ACT17 are time barriers: a set called '14' can hold 2017 apps."""
    folder = tmp_path / "raw"
    folder.mkdir()
    rows = []
    for i in range(40):
        year = 2014 if i < 20 else 2017
        rows.append({"sha256": f"{i:064x}", "year": year, "f": i % 2})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "X14-training.characteristics.csv", index=False)
    labels = frame[["sha256"]].assign(malware=[i % 2 for i in range(40)], dataset="x")
    labels.to_csv(folder / "X14-training.goodmal.csv", index=False)
    cand = {"X": {"labelled": ["X14-*"], "early_years": [2014], "late_years": [2017]}}
    _, verdicts, _ = audit.run_audit(discover_sets(folder), "sha256", cand, minimum=10)
    assert verdicts[0].early_counts == (10, 10)
    assert verdicts[0].late_counts == (10, 10)
    assert verdicts[0].passes


def test_audit_fails_below_minimum(project: Config) -> None:
    sets = discover_sets(project.raw_dir)
    _, verdicts, _ = audit.run_audit(sets, project.id_column, project.rq3_candidates, 10**6)
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


def test_second_machine_registration_is_safe(
    project: Config, synthetic_zip: Path, tmp_path: Path
) -> None:
    """Registering the same zip again adds no DATA.md row; a different zip is refused."""
    from shortcutcheck.data.register import recorded_digests, register_zip

    data_md = project.root / "DATA.md"
    before = data_md.read_text()
    assert len(recorded_digests(data_md)) == 1
    register_zip(synthetic_zip, tmp_path / "elsewhere", data_md)  # same zip, new machine
    assert data_md.read_text() == before
    other = tmp_path / "other.zip"
    other.write_bytes(synthetic_zip.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="not the data"):
        register_zip(other, tmp_path / "elsewhere2", data_md)
