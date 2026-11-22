"""Command-line entry point. Phase 1 commands live under ``shortcutcheck data`` and
``shortcutcheck baselines``; the release-gate ``run`` command arrives in Phase 4."""

from __future__ import annotations

import io
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from shortcutcheck import gate as gate_mod
from shortcutcheck import pilot as pilot_mod
from shortcutcheck.addenda import (
    AddendumNotCommittedError,
    addendum_d,
    addendum_e,
    fix_poc,
    mechanism,
    phase_a,
    phase_b,
    require_committed,
)
from shortcutcheck.analysis import figures as fig_mod
from shortcutcheck.analysis import h1 as h1_mod
from shortcutcheck.analysis import h2 as h2_mod
from shortcutcheck.analysis import h3 as h3_mod
from shortcutcheck.config import Config, load_config
from shortcutcheck.data import audit as audit_mod
from shortcutcheck.data.dada import DadaSet, discover_sets, match_sets, read_set
from shortcutcheck.data.dedup import DedupReport, deduplicate
from shortcutcheck.data.register import register_zip, sha256_of
from shortcutcheck.data.safe_io import write_frame, write_text
from shortcutcheck.data.splits import (
    PART,
    load_part,
    make_rq3_split,
    make_rq12_split,
    save_split,
    verify_against_index,
    write_processed,
)
from shortcutcheck.data.validity import check_dex_date, decision
from shortcutcheck.evaluation import addendum_d_run as d_run
from shortcutcheck.evaluation import addendum_e_run as e_run
from shortcutcheck.evaluation import experiment as engine_mod
from shortcutcheck.evaluation import rq3_ground_truth as rq3_truth
from shortcutcheck.evaluation import rq3_signals as rq3_sig
from shortcutcheck.evaluation.preregistration import analysis_status, check_preregistration
from shortcutcheck.experiments.compare import compare_tables
from shortcutcheck.experiments.spec import load_experiment
from shortcutcheck.mrs import applicability as appl_mod
from shortcutcheck.mrs import permissions as perm_mod
from shortcutcheck.mrs import realism_report as real_mod
from shortcutcheck.provenance import git_commit
from shortcutcheck.runner import known_answers as ka_mod
from shortcutcheck.synthetic import make_synthetic_dada, write_synthetic_config
from shortcutcheck.training import baselines as base_mod
from shortcutcheck.training import pool as pool_mod
from shortcutcheck.training.features import select

app = typer.Typer(help="ShortcutCheck: metamorphic shortcut testing for malware detectors.")
data_app = typer.Typer(help="Phase 1 data steps (roadmap 1.3-1.7).")
app.add_typer(data_app, name="data")
mrs_app = typer.Typer(help="Phase 2 metamorphic relation steps (roadmap 2.1-2.6).")
app.add_typer(mrs_app, name="mrs")
exp_app = typer.Typer(help="Phase 3 experiments, one YAML config each (roadmap 3.1).")
app.add_typer(exp_app, name="experiment")
analyse_app = typer.Typer(help="Judge H1-H3 as pre-registered (roadmap 3.2-3.8).")
app.add_typer(analyse_app, name="analyse")
addendum_app = typer.Typer(help="Pre-registered addenda (PREREGISTRATION-ADDENDUM-*.md).")
app.add_typer(addendum_app, name="addendum")

ConfigOpt = typer.Option(None, "--config", help="Path to configs/phase1.yaml.")


def _cfg(config: Path | None) -> Config:
    return load_config(config)


def _sets(cfg: Config) -> list[DadaSet]:
    sets = discover_sets(cfg.raw_dir)
    if not sets:
        raise typer.BadParameter(
            f"No *.characteristics.csv found under {cfg.raw_dir}. Run `shortcutcheck data "
            "register <zip>` first."
        )
    return sets


@data_app.command("register")
def register(zip_path: Path, config: Path | None = ConfigOpt) -> None:
    """Step 1.3: hash the Dada zip, extract it read-only, record it in DATA.md."""
    cfg = _cfg(config)
    digest = register_zip(zip_path, cfg.raw_dir, cfg.root / "DATA.md")
    typer.echo(f"Registered {zip_path.name}\nSHA-256 {digest}\nExtracted to {cfg.raw_dir}")


@data_app.command("audit")
def audit(config: Path | None = ConfigOpt) -> None:
    """Step 1.4: write data_audit.md with per-set counts and the RQ3 sufficiency check."""
    cfg = _cfg(config)
    sets = _sets(cfg)
    summaries, verdicts, tables = audit_mod.run_audit(
        sets,
        cfg.id_column,
        cfg.rq3_candidates,
        cfg.rq3_min_per_class_per_year,
        cfg.android_first_year,
        cfg.collection_year,
    )
    text = audit_mod.render_markdown(
        summaries, verdicts, cfg.raw_dir.relative_to(cfg.root).as_posix(), tables
    )
    write_text(cfg.root / "data_audit.md", text, cfg.raw_root)
    typer.echo(text)
    if not any(v.passes for v in verdicts):
        raise typer.Exit(code=1)


@data_app.command("validity")
def validity(config: Path | None = ConfigOpt) -> None:
    """Step 1.5: dex_date validity per set and overall, with the keep/drop decision."""
    cfg = _cfg(config)
    frames, lines = [], [
        "# dex_date validity (roadmap step 1.5)",
        "",
        f"Invalid = missing/zero, before {cfg.android_first_year}, or after "
        f"{cfg.collection_year}. Generated by `shortcutcheck data validity`.",
        "",
        "| Set | Rows | Missing/zero | Before Android | After collection | Exactly 1980 "
        "| Invalid share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in _sets(cfg):
        frame = read_set(s, cfg.id_column)
        if "dex_date" not in frame.columns:
            continue
        frames.append(frame["dex_date"])
        r = check_dex_date(frame["dex_date"], cfg.android_first_year, cfg.collection_year)
        lines.append(
            f"| {s.name} | {r.total} | {r.missing_or_zero} | {r.before_android} "
            f"| {r.after_collection} | {r.exactly_1980} | {r.invalid_share:.1%} |"
        )
    if not frames:
        raise typer.BadParameter("No set has a dex_date column.")
    overall = check_dex_date(
        pd.concat(frames, ignore_index=True), cfg.android_first_year, cfg.collection_year
    )
    verdict = decision(overall, cfg.dex_date_invalid_threshold)
    lines += ["", f"**Overall: {verdict}**", ""]
    text = "\n".join(lines)
    write_text(cfg.reports_dir / "dex_date_validity.md", text, cfg.raw_root)
    typer.echo(text)


def _combined(cfg: Config, patterns: list[str]) -> tuple[pd.DataFrame, dict[str, str], DedupReport]:
    sets = _sets(cfg)
    chosen = {s.name: s for p in patterns for s in match_sets(sets, p)}
    if not chosen:
        raise typer.BadParameter(f"No Dada set matches {patterns}.")
    frame = pd.concat([read_set(s, cfg.id_column) for s in chosen.values()], ignore_index=True)
    clean, report = deduplicate(frame, cfg.id_column)
    sources = {name: sha256_of(s.features_path) for name, s in sorted(chosen.items())}
    return clean, sources, report


def _build_rq12(
    cfg: Config, sets: list[str]
) -> tuple[pd.DataFrame, dict[str, object], DedupReport]:
    """The RQ1/RQ2 split, computed deterministically (step 1.7). Used by split and rebuild."""
    clean, sources, report = _combined(cfg, sets)
    split = make_rq12_split(clean, cfg.split_seed, cfg.rq12_ratios)
    extra: dict[str, object] = {
        "kind": "rq12",
        "seed": cfg.split_seed,
        "ratios": cfg.rq12_ratios,
        "sources": sources,
        "dedup": asdict(report),
    }
    return split, extra, report


def _exclude_other_pools(
    cfg: Config, cand: dict[str, Any], clean: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Drop apps that occur (any year) in the pools of the candidates named in
    ``exclude_pools_of``, so a replication shares no app with them (Addendum E)."""
    others: list[str] = list(cand.get("exclude_pools_of") or [])
    if not others:
        return clean, 0
    seen: set[str] = set()
    for other in others:
        pool, _ = audit_mod.candidate_pool(
            _sets(cfg), cfg.rq3_candidates[str(other)], cfg.id_column, dedup=False
        )
        seen |= set(pool[cfg.id_column])
    drop = clean[cfg.id_column].isin(seen)
    return clean[~drop].reset_index(drop=True), int(drop.sum())


def _build_rq3(cfg: Config, candidate: str) -> tuple[pd.DataFrame, dict[str, object], DedupReport]:
    """The RQ3 split, computed deterministically (step 1.7). Used by split and rebuild."""
    cand = cfg.rq3_candidates[candidate]
    raw_pool, sources = audit_mod.candidate_pool(_sets(cfg), cand, cfg.id_column, dedup=False)
    clean, report = deduplicate(raw_pool, cfg.id_column)
    clean, excluded = _exclude_other_pools(cfg, cand, clean)
    first, last = cfg.android_first_year, cfg.collection_year
    early_years = [int(y) for y in cand["early_years"]]
    late_years = [int(y) for y in cand["late_years"]]
    early = audit_mod.window(clean, early_years, first, last)
    late = audit_mod.window(clean, late_years, first, last)
    split, overlap = make_rq3_split(
        early, late, cfg.split_seed, cfg.rq3_val_fraction, cfg.id_column
    )
    extra: dict[str, object] = {
        "kind": "rq3",
        "candidate": candidate,
        "seed": cfg.split_seed,
        "val_fraction": cfg.rq3_val_fraction,
        "early_years": early_years,
        "late_years": late_years,
        "sources": {n: sha256_of(s.features_path) for n, s in _named(cfg, sources)},
        "late_removed_for_hash_overlap": overlap,
        "dedup": asdict(report),
    }
    if cand.get("exclude_pools_of"):
        extra["excluded_pools_of"] = list(cand["exclude_pools_of"])
        extra["removed_for_overlap_with_excluded_pools"] = excluded
    return split, extra, report


def _build_holdout(
    cfg: Config, candidate: str, years: list[int]
) -> tuple[pd.DataFrame, dict[str, object], DedupReport]:
    """A held-out time window: every app of ``years`` locked as test data (no train/val).

    Apps already in the primary RQ3 split built from the same candidate are removed, so the
    holdout shares no app with any model's training, validation or test data.
    """
    cand = cfg.rq3_candidates[candidate]
    raw_pool, sources = audit_mod.candidate_pool(_sets(cfg), cand, cfg.id_column, dedup=False)
    clean, report = deduplicate(raw_pool, cfg.id_column)
    clean, excluded = _exclude_other_pools(cfg, cand, clean)
    held = audit_mod.window(clean, years, cfg.android_first_year, cfg.collection_year)
    used, _, _ = _build_rq3(cfg, candidate)
    overlap = held[cfg.id_column].isin(set(used[cfg.id_column]))
    split = held[~overlap].assign(**{PART: "test"}).reset_index(drop=True)
    extra: dict[str, object] = {
        "kind": "holdout",
        "candidate": candidate,
        "years": years,
        "sources": {n: sha256_of(s.features_path) for n, s in _named(cfg, sources)},
        "removed_for_overlap_with_rq3_split": int(overlap.sum()),
        "dedup": asdict(report),
    }
    if cand.get("exclude_pools_of"):
        extra["excluded_pools_of"] = list(cand["exclude_pools_of"])
        extra["removed_for_overlap_with_excluded_pools"] = excluded
    return split, extra, report


def _build_dada(
    cfg: Config, train_sets: list[str], test_sets: list[str]
) -> tuple[pd.DataFrame, dict[str, object], DedupReport]:
    """Dada's own protocol: its training file trains, its test file tests. No re-splitting.

    Each part is deduplicated as in step 1.6. Test apps that also occur in the training part
    are removed (counted in the manifest). There is no validation part.
    """
    train, train_sources, report = _combined(cfg, train_sets)
    parts = [train.assign(**{PART: "train"})]
    sources = dict(train_sources)
    overlap = 0
    if test_sets:
        test, test_sources, _ = _combined(cfg, test_sets)
        seen = test[cfg.id_column].isin(set(train[cfg.id_column]))
        overlap = int(seen.sum())
        parts.append(test[~seen].assign(**{PART: "test"}))
        sources.update(test_sources)
    split = pd.concat(parts, ignore_index=True)
    extra: dict[str, object] = {
        "kind": "dada",
        "train_sets": train_sets,
        "test_sets": test_sets,
        "sources": sources,
        "test_removed_for_overlap_with_train": overlap,
        "dedup": asdict(report),
    }
    return split, extra, report


@data_app.command("split-dada")
def split_dada(
    name: str = typer.Option(..., help="Split name, e.g. dada-VS-AG_deb."),
    train: list[str] = typer.Option(
        ..., "--train", help="Dada training file(s), e.g. VS-AG-training."
    ),
    test: list[str] | None = typer.Option(None, "--test", help="Dada test file(s), if any."),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum B: freeze a split that follows Dada's own train/test files exactly."""
    cfg = _cfg(config)
    split, extra, report = _build_dada(cfg, list(train), list(test or []))
    paths = save_split(cfg, name, split, extra)
    _write_dedup(cfg, name, report)
    counts = split[PART].value_counts().to_dict()
    typer.echo(f"Frozen Dada-protocol split '{name}' {counts} -> {paths.index_dir}.")


@data_app.command("split-holdout")
def split_holdout(
    years: list[int] = typer.Option(..., "--years", help="Years to hold out, e.g. 2015 2016."),
    name: str = typer.Option(..., help="Split name, e.g. rq3-AndroCT-holdout-2015-16."),
    candidate: str | None = typer.Option(
        None, help="An rq3_candidates entry (default: rq3_primary in config)."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Freeze a time window as a locked, test-only split (readable only via load_test())."""
    cfg = _cfg(config)
    candidate = candidate or cfg.rq3_primary
    if candidate not in cfg.rq3_candidates:
        raise typer.BadParameter(f"Unknown candidate '{candidate}'.")
    split, extra, report = _build_holdout(cfg, candidate, sorted(years))
    paths = save_split(cfg, name, split, extra)
    _write_dedup(cfg, name, report)
    typer.echo(
        f"Frozen holdout '{name}' ({len(split)} apps, years {sorted(years)}) -> "
        f"{paths.index_dir}. Locked: only load_test() can read it. Commit the index files."
    )


@data_app.command("rebuild")
def rebuild(config: Path | None = ConfigOpt) -> None:
    """Rebuild data/processed on a new machine from the raw data and the frozen splits.

    Each split is recomputed with the same code and seed, then checked against its committed
    index files and manifest (sources, apps, class counts). Only if everything matches are the
    processed files written. Frozen index files are never changed.
    """
    cfg = _cfg(config)
    manifests = sorted(cfg.splits_dir.glob("*/manifest.json"))
    if not manifests:
        raise typer.BadParameter(f"No frozen splits under {cfg.splits_dir}.")
    for path in manifests:
        name = path.parent.name
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["kind"] == "rq12":
            # The configured order matters: sets are concatenated before the seeded shuffle.
            sets = cfg.rq12_splits.get(name) or list(manifest["sources"])
            split, extra, _ = _build_rq12(cfg, sets)
        elif manifest["kind"] == "dada":
            split, extra, _ = _build_dada(
                cfg, list(manifest["train_sets"]), list(manifest["test_sets"])
            )
        elif manifest["kind"] == "holdout":
            years = [int(y) for y in manifest["years"]]
            split, extra, _ = _build_holdout(cfg, manifest["candidate"], years)
        else:
            split, extra, _ = _build_rq3(cfg, manifest["candidate"])
        verify_against_index(cfg, name, split, extra["sources"])  # type: ignore[arg-type]
        write_processed(cfg, name, split)
        typer.echo(f"Rebuilt '{name}': matches its frozen index files and manifest.")


def _write_dedup(cfg: Config, name: str, report: DedupReport) -> None:
    lines = [f"# Deduplication: `{name}` (roadmap step 1.6)", "", "| Measure | Count |"]
    lines += ["| --- | ---: |"] + [f"| {k} | {v} |" for k, v in asdict(report).items()]
    write_text(cfg.reports_dir / f"{name}_dedup.md", "\n".join(lines) + "\n", cfg.raw_root)


@data_app.command("split-rq12")
def split_rq12(
    name: str = typer.Option(..., help="Split name, e.g. rq12-DR-AG."),
    sets: list[str] | None = typer.Option(
        None, "--sets", help="Dada set names or globs (default: rq12_splits[name] in config)."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 1.7: stratified 70/15/15 split, frozen."""
    cfg = _cfg(config)
    sets = sets or cfg.rq12_splits.get(name)
    if not sets:
        raise typer.BadParameter(f"No --sets given and '{name}' is not in rq12_splits.")
    split, extra, report = _build_rq12(cfg, sets)
    paths = save_split(cfg, name, split, extra)
    _write_dedup(cfg, name, report)
    typer.echo(f"Frozen split '{name}' -> {paths.index_dir}. Commit the index files.")


@data_app.command("split-rq3")
def split_rq3(
    candidate: str | None = typer.Option(
        None, help="An rq3_candidates entry (default: rq3_primary in config)."
    ),
    name: str | None = typer.Option(None, help="Split name (default: rq3-<candidate>)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 1.7: early-window apps -> train/val, late-window apps held out entirely, frozen."""
    cfg = _cfg(config)
    candidate = candidate or cfg.rq3_primary
    if candidate not in cfg.rq3_candidates:
        raise typer.BadParameter(f"Unknown candidate '{candidate}'.")
    split, extra, report = _build_rq3(cfg, candidate)
    split_name = name or f"rq3-{candidate}"
    paths = save_split(cfg, split_name, split, extra)
    _write_dedup(cfg, split_name, report)
    typer.echo(
        f"Frozen split '{split_name}' -> {paths.index_dir} "
        f"({extra['late_removed_for_hash_overlap']} late apps removed for "
        "hash overlap). Commit the index files."
    )


def _named(cfg: Config, names: list[str]) -> list[tuple[str, DadaSet]]:
    by_name = {s.name: s for s in _sets(cfg)}
    return [(n, by_name[n]) for n in names]


@app.command("baselines")
def baselines(
    split: str = typer.Option(..., help="Frozen split name."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 1.8: RF and LR, full and behaviour-only, 5 seeds, plus the shuffled-label check."""
    cfg = _cfg(config)
    commit = git_commit(cfg.root)
    results, check = base_mod.run_baselines(cfg, split, commit)
    write_frame(results, cfg.reports_dir / f"{split}_baselines.csv", cfg.raw_root)
    text = base_mod.render_markdown(split, base_mod.summarise(results), check, commit)
    write_text(cfg.reports_dir / f"{split}_baselines.md", text, cfg.raw_root)
    typer.echo(text)
    if not check.passes:
        raise typer.Exit(code=1)


@mrs_app.command("applicability")
def mrs_applicability(
    split: list[str] | None = typer.Option(None, help="Frozen split(s); default: all five."),
    config: Path | None = ConfigOpt,
) -> None:
    """Steps 2.2-2.3: which MRs apply to each split (training partition only)."""
    cfg = _cfg(config)
    rows = [appl_mod.check_split(cfg, s) for s in (split or appl_mod.default_splits(cfg))]
    text = appl_mod.render_markdown(rows, git_commit(cfg.root))
    write_text(cfg.reports_dir / "mr_applicability.md", text, cfg.raw_root)
    typer.echo(text)


@mrs_app.command("permissions")
def mrs_permissions(config: Path | None = ConfigOpt) -> None:
    """Step 2.3: the official protection level of every permission feature, and its MR."""
    cfg = _cfg(config)
    text = perm_mod.render_markdown()
    write_text(cfg.reports_dir / "permission_levels.md", text, cfg.raw_root)
    typer.echo(text)


@mrs_app.command("realism")
def mrs_realism(
    split: list[str] | None = typer.Option(None, help="Frozen split(s); default: all five."),
    max_eval: int = typer.Option(5000, help="Validation apps sampled per split."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 2.4: share of perturbed apps excluded by the realism check, per MR and split."""
    cfg = _cfg(config)
    commit = git_commit(cfg.root)
    names = split or appl_mod.default_splits(cfg)
    results = [real_mod.check_split(cfg, s, commit, max_eval=max_eval) for s in names]
    text = real_mod.render_markdown(results, commit)
    write_text(cfg.reports_dir / "realism.md", text, cfg.raw_root)
    typer.echo(text)


@mrs_app.command("known-answers")
def mrs_known_answers(
    split: list[str] | None = typer.Option(None, help="Frozen split(s); default: all five."),
    max_eval: int = typer.Option(5000, help="Validation apps sampled per split."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 2.5: oracle models with known answers; exits non-zero if any answer is wrong."""
    cfg = _cfg(config)
    commit = git_commit(cfg.root)
    answers = [
        a
        for s in (split or appl_mod.default_splits(cfg))
        for a in ka_mod.check_split(cfg, s, commit, max_eval=max_eval)
    ]
    text = ka_mod.render_markdown(answers, commit)
    write_text(cfg.reports_dir / "known_answers.md", text, cfg.raw_root)
    typer.echo(text)
    if not all(a.passes for a in answers if a.testable):
        raise typer.Exit(code=1)


@app.command("pilot")
def pilot(
    split: str = typer.Option("rq12-DR-AG", help="Frozen split (plan: the Drebin mixed set)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 2.7: pilot RQ1 as stated in docs/pilot_plan.md; exits non-zero if Gate 2 fails."""
    cfg = _cfg(config)
    result = pilot_mod.run_pilot(cfg, split, git_commit(cfg.root))
    out = cfg.reports_dir
    write_frame(result.rows, out / f"pilot_rq1_{split}.csv", cfg.raw_root)
    write_frame(result.detection, out / f"pilot_rq1_{split}_detection.csv", cfg.raw_root)
    text = pilot_mod.render_markdown(result)
    write_text(cfg.reports_dir / f"pilot_rq1_{split}.md", text, cfg.raw_root)
    typer.echo(text)
    if not pilot_mod.gate2(result).passes:
        raise typer.Exit(code=1)


@app.command("pilot-compare")
def pilot_compare(
    split: str = typer.Option("rq12-DR-AG", help="Frozen split the pilot ran on."),
    ref: str = typer.Option("HEAD", help="Git revision holding the reference results."),
    config: Path | None = ConfigOpt,
) -> None:
    """Gate 2 reproduction: compare a fresh `make pilot` with the committed results."""
    cfg = _cfg(config)
    rel = f"reports/pilot_rq1_{split}.csv"
    committed = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=cfg.root, capture_output=True, text=True, check=True
    ).stdout
    result = pilot_mod.compare(pd.read_csv(io.StringIO(committed)), pd.read_csv(cfg.root / rel))
    typer.echo(f"Rows compared: {result.n_rows}; identical: {result.identical_rows}")
    typer.echo(f"Largest difference in any rate: {result.max_rate_diff:.4f}")
    typer.echo(f"Same Gate 2 verdict: {'yes' if result.same_gate else 'NO'}")
    for line in result.worst:
        typer.echo(f"  {line}")
    typer.echo(f"Verdict: {result.verdict}")
    if result.verdict == "DIFFERENT":
        raise typer.Exit(code=1)


@app.command("pilot-reproduction")
def pilot_reproduction(
    folder: Path = typer.Argument(..., help="Folder with a second machine's pilot CSVs."),
    label: str = typer.Option(..., help="Machine description, e.g. 'macOS 26, Apple M-series'."),
    split: str = typer.Option("rq12-DR-AG", help="Frozen split the pilot ran on."),
    config: Path | None = ConfigOpt,
) -> None:
    """Gate 2: write a per-algorithm comparison of a second machine's pilot run."""
    cfg = _cfg(config)
    name = f"pilot_rq1_{split}"
    text = pilot_mod.render_reproduction(
        pd.read_csv(cfg.reports_dir / f"{name}.csv"),
        pd.read_csv(folder / f"{name}.csv"),
        pd.read_csv(cfg.reports_dir / f"{name}_detection.csv"),
        pd.read_csv(folder / f"{name}_detection.csv"),
        label,
    )
    write_text(folder / "comparison.md", text, cfg.raw_root)
    typer.echo(text)


RESULT_FILES = ("rows.csv", "detection.csv", "inputs.csv")


def _run_summary(
    exp_path: Path, result: engine_mod.ExperimentResult, opened: tuple[str, ...]
) -> str:
    lines = [
        f"# Experiment `{result.experiment}`: run record",
        "",
        f"Generated by `shortcutcheck experiment run {exp_path.as_posix()}`. Git commit "
        f"`{result.commit}`. Per-model MLflow run IDs are in `rows.csv`. Hypotheses are judged "
        "by `shortcutcheck analyse`, not here.",
        "",
        "Test partitions opened (once each, logged in `logs/test_access.jsonl`): "
        + (", ".join(f"`{s}`" for s in opened) or "none")
        + ".",
        "",
        "## Inputs per run and target",
        "",
        "| Run | Evaluated on | Partition | Role | Apps | Excluded (trained on) | Inputs | "
        "MR sample | MRs |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in result.inputs.itertuples():
        lines.append(
            f"| {r.run} | {r.eval_split} | {r.partition} | {r.role} | {r.n_partition} | "
            f"{r.n_excluded} | {r.n_inputs} | {r.n_mr_sample} | {r.mrs} |"
        )
    return "\n".join(lines) + "\n"


@exp_app.command("run")
def experiment_run(
    exp_config: Path = typer.Argument(..., help="configs/experiments/<name>.yaml"),
    out: Path | None = typer.Option(None, help="Output folder (default reports/<experiment>)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 3.1: run one experiment config and write its result tables."""
    cfg = _cfg(config)
    exp = load_experiment(exp_config, cfg)
    if exp.engine == "pool":
        folder = out or cfg.reports_dir / exp.experiment
        commit = git_commit(cfg.root)
        model_dir = (out / "models") if out else cfg.root / "data" / "models" / exp.experiment
        pool = pool_mod.build_pool(
            cfg, exp, commit, model_dir, progress=lambda m: typer.echo(m, err=True)
        )
        write_frame(pool, folder / "pool_models.csv", cfg.raw_root)
        x, y = select(
            load_part(cfg, str(exp.pool["split"]), "train"),
            "full",
            cfg.metadata_columns,
            cfg.id_column,
            cfg.dropped_features,
        )
        skew = exp.pool["skew"]
        plan = pool_mod.skew_plan(x, y, str(skew["column"]), [float(v) for v in skew["levels"]])
        text = pool_mod.render_pool(pool, plan.median, commit)
        write_text(folder / "pool.md", text, cfg.raw_root)
        typer.echo(text)
        typer.echo("Pool built (step 3.5). Signals (3.6) and ground truth (3.7) come next.")
        return
    result = engine_mod.run_experiment(
        cfg, exp, git_commit(cfg.root), progress=lambda m: typer.echo(m, err=True)
    )
    folder = out or cfg.reports_dir / exp.experiment
    tables = (result.rows, result.detection, result.inputs)
    for name, frame in zip(RESULT_FILES, tables, strict=True):
        write_frame(frame, folder / name, cfg.raw_root)
    text = _run_summary(exp_config, result, exp.test_splits)
    write_text(folder / "run.md", text, cfg.raw_root)
    typer.echo(text)


@exp_app.command("signals")
def experiment_signals(
    exp_config: Path = typer.Argument(..., help="configs/experiments/rq3.yaml"),
    out: Path | None = typer.Option(None, help="Output folder (default reports/<experiment>)."),
    workers: int = typer.Option(2, help="Worker processes (results do not depend on it)."),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 3.6: label-free signals for every pool model, plus an identical-rerun check."""
    cfg = _cfg(config)
    exp = load_experiment(exp_config, cfg)
    folder = out or cfg.reports_dir / exp.experiment
    model_dir = cfg.root / "data" / "models" / exp.experiment
    commit = git_commit(cfg.root)
    ctx = rq3_sig.build_context(cfg, exp, model_dir)  # opens 2017-19 once, inputs only
    table = rq3_sig.compute_signals(
        exp, ctx, workers=workers, progress=lambda m: typer.echo(m, err=True)
    )
    table["git_commit"] = commit
    write_frame(table, folder / "signals.csv", cfg.raw_root)
    text = rq3_sig.render_signals(table, commit)
    # 3.1 Check for the signals: recompute a fixed subset in one process, same inputs.
    subset = (
        table.groupby("algorithm", sort=False).head(rq3_sig.RECHECK_PER_ALGORITHM)["model"].tolist()
    )
    again = rq3_sig.compute_signals(exp, ctx, workers=1, only=subset)
    first = table[table["model"].isin(subset)].drop(columns="git_commit").reset_index(drop=True)
    c = compare_tables(first, again)
    verdict = "IDENTICAL" if c.identical else "DIFFERENT"
    text += (
        f"\n## Identical-rerun check\n\n{len(subset)} models ({', '.join(subset)}) recomputed "
        f"in a single process on the same inputs: {c.identical_rows}/{c.n_rows} rows identical "
        f"over {len(c.columns)} columns (largest difference {c.max_abs_diff:.3g}) -> "
        f"**{verdict}**.\n"
    )
    write_text(folder / "signals.md", text, cfg.raw_root)
    typer.echo(text)
    if not c.identical:
        raise typer.Exit(code=1)


@exp_app.command("ground-truth")
def experiment_ground_truth(
    exp_config: Path = typer.Argument(..., help="configs/experiments/rq3.yaml"),
    out: Path | None = typer.Option(None, help="Output folder (default reports/<experiment>)."),
    dry_run: bool = typer.Option(
        False, help="Use the validation partition in place of the test partition (no test access)."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 3.7: F1 drop per pool model (one load_test() per model) and the pool table."""
    cfg = _cfg(config)
    exp = load_experiment(exp_config, cfg)
    if dry_run and out is None:
        typer.echo("--dry-run needs --out, so its numbers can never land in reports/.")
        raise typer.Exit(code=2)
    folder = out or cfg.reports_dir / exp.experiment
    signals = pd.read_csv(
        folder / "signals.csv" if not dry_run else cfg.reports_dir / exp.experiment / "signals.csv"
    )
    model_dir = cfg.root / "data" / "models" / exp.experiment
    commit = git_commit(cfg.root)
    before = rq3_truth.n_test_accesses(cfg)
    truth = rq3_truth.ground_truth(
        cfg, exp, model_dir, dry_run=dry_run, progress=lambda m: typer.echo(m, err=True)
    )
    n_log = rq3_truth.n_test_accesses(cfg) - before
    pool = rq3_truth.assemble_pool(signals, truth, commit)
    write_frame(truth, folder / "ground_truth.csv", cfg.raw_root)
    write_frame(pool, folder / "pool.csv", cfg.raw_root)
    text = rq3_truth.render_truth(truth, pool, n_log, commit)
    if dry_run:
        text = "# DRY RUN on the validation partition: not results\n\n" + text
    write_text(folder / "ground_truth.md", text, cfg.raw_root)
    typer.echo(text)


@exp_app.command("compare")
def experiment_compare(
    first: Path = typer.Argument(..., help="Folder (or CSV) from one run."),
    second: Path = typer.Argument(..., help="Folder (or CSV) from another run, same config."),
    report: Path | None = typer.Option(None, help="Also append the result to this Markdown file."),
) -> None:
    """Step 3.1 Check: are two runs with the same seeds identical? Exits 1 if not."""
    out: list[str] = [f"`{first.as_posix()}` vs `{second.as_posix()}`:", ""]
    pairs = (
        [(first / n, second / n) for n in RESULT_FILES[:2]] if first.is_dir() else [(first, second)]
    )
    ok = True
    for a, b in pairs:
        c = compare_tables(pd.read_csv(a), pd.read_csv(b))
        verdict = "IDENTICAL" if c.identical else "DIFFERENT"
        out.append(
            f"- {a.name}: {c.identical_rows}/{c.n_rows} rows identical over {len(c.columns)} "
            f"columns (largest difference {c.max_abs_diff:.3g}) -> {verdict}"
        )
        out += [f"  - {problem}" for problem in c.problems]
        ok &= c.identical
    out += ["", f"Verdict: **{'IDENTICAL' if ok else 'DIFFERENT'}**", ""]
    typer.echo("\n".join(out))
    if report is not None:
        with report.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(out) + "\n")
    if not ok:
        raise typer.Exit(code=1)


def _header(cfg: Config, command: str, source: Path, commits: list[str]) -> str:
    runs = ", ".join(f"`{c}`" for c in sorted(set(commits))) or "unknown"
    return (
        f"Generated by `{command}` from `{source.relative_to(cfg.root).as_posix()}` "
        f"(experiment run at commit {runs}; analysis at commit `{git_commit(cfg.root)}`). "
        f"Rules: `PREREGISTRATION.md`. {analysis_status(cfg.root)}"
    )


def _rows(cfg: Config, experiment: str) -> tuple[Path, pd.DataFrame, list[str]]:
    folder = cfg.reports_dir / experiment
    path = folder / "rows.csv"
    if not path.exists():
        typer.echo(f"No {path}; run `make {experiment}` first.")
        raise typer.Exit(code=2)
    run_md = (folder / "run.md").read_text(encoding="utf-8") if (folder / "run.md").exists() else ""
    commits = [part.split("`")[1] for part in run_md.split("Git commit ")[1:2]]
    return path, pd.read_csv(path), commits


@analyse_app.command("rq1")
def analyse_rq1(config: Path | None = ConfigOpt) -> None:
    """Step 3.3: judge H1 from reports/rq1/rows.csv; writes h1.md and a figure."""
    cfg = _cfg(config)
    path, rows, commits = _rows(cfg, "rq1")
    detection = pd.read_csv(path.parent / "detection.csv")
    result = h1_mod.judge_h1(rows)
    text = h1_mod.render_h1(
        result, rows, detection, _header(cfg, "shortcutcheck analyse rq1", path, commits)
    )
    write_text(path.parent / "h1.md", text, cfg.raw_root)
    fig_mod.rq1_violation_rates(rows, path.parent / "rq1_violation_rates.svg", h1_mod.CORE_MRS)
    typer.echo(text)


@analyse_app.command("rq2")
def analyse_rq2(config: Path | None = ConfigOpt) -> None:
    """Step 3.4: judge H2 from reports/rq2/rows.csv; writes h2.md, h2_tests.csv, a figure."""
    cfg = _cfg(config)
    path, rows, commits = _rows(cfg, "rq2")
    result = h2_mod.judge_h2(rows)
    text = h2_mod.render_h2(result, rows, _header(cfg, "shortcutcheck analyse rq2", path, commits))
    write_text(path.parent / "h2.md", text, cfg.raw_root)
    write_frame(result.tests, path.parent / "h2_tests.csv", cfg.raw_root)
    fig_mod.rq2_pairs(rows, path.parent / "rq2_mr4_pairs.svg")
    typer.echo(text)


@analyse_app.command("rq3")
def analyse_rq3(config: Path | None = ConfigOpt) -> None:
    """Step 3.8: judge H3a-H3c and the Gate 3 outcome from reports/rq3/pool.csv."""
    cfg = _cfg(config)
    path = cfg.reports_dir / "rq3" / "pool.csv"
    if not path.exists():
        typer.echo(f"No {path}; the RQ3 pool is built in steps 3.5-3.7.")
        raise typer.Exit(code=2)
    pool = pd.read_csv(path)
    result = h3_mod.judge_h3(pool)
    commits = sorted(set(pool["git_commit"].astype(str))) if "git_commit" in pool else []
    header = _header(cfg, "shortcutcheck analyse rq3", path, commits)
    text = h3_mod.render_h3(result, header, h3_mod.secondary(pool))
    write_text(path.parent / "h3.md", text, cfg.raw_root)
    write_frame(result.comparisons, path.parent / "h3_comparisons.csv", cfg.raw_root)
    fig_mod.rq3_scatter(pool, path.parent / "rq3_mr4_vs_f1_drop.svg")
    typer.echo(text)


def _addendum_a(cfg: Config) -> str:
    try:
        return require_committed(cfg.root, phase_a.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err


@addendum_app.command("a1")
def addendum_a1(config: Path | None = ConfigOpt) -> None:
    """Addendum A1: why debiased training increased metadata reliance (training data only)."""
    cfg = _cfg(config)
    prereg = _addendum_a(cfg)
    measures = phase_a.a1_measures(cfg)
    text = phase_a.render_a1(
        measures, phase_a.judge_a1(measures), phase_a.header("a1", git_commit(cfg.root), prereg)
    )
    folder = cfg.reports_dir / "addendum_a"
    write_frame(measures, folder / "a1_measures.csv", cfg.raw_root)
    write_text(folder / "a1.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("a2")
def addendum_a2(config: Path | None = ConfigOpt) -> None:
    """Addendum A2: violation rate vs how much metadata helps each RQ1 model (no new data)."""
    cfg = _cfg(config)
    prereg = _addendum_a(cfg)
    rows = pd.read_csv(cfg.reports_dir / "rq1" / "rows.csv")
    det = pd.read_csv(cfg.reports_dir / "rq1" / "detection.csv")
    results = {
        "MR4 vs F1 gain": phase_a.judge_a2(phase_a.a2_table(rows, det)),
        "MR1 vs F1 gain": phase_a.judge_a2(phase_a.a2_table(rows, det, mr="MR1")),
        "MR4 vs ROC-AUC gain": phase_a.judge_a2(phase_a.a2_table(rows, det, metric="roc_auc")),
    }
    text = phase_a.render_a2(results, phase_a.header("a2", git_commit(cfg.root), prereg))
    folder = cfg.reports_dir / "addendum_a"
    write_frame(phase_a.a2_table(rows, det), folder / "a2_table.csv", cfg.raw_root)
    write_text(folder / "a2.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("a3")
def addendum_a3(config: Path | None = ConfigOpt) -> None:
    """Addendum A3: RQ3 pre-deployment MR4 with plain-Gower realism (validation data only)."""
    cfg = _cfg(config)
    prereg = _addendum_a(cfg)
    exp = load_experiment(Path("configs/experiments/rq3.yaml"), cfg)
    plain = phase_a.a3_rates(cfg, exp, cfg.root / "data" / "models" / exp.experiment)
    pool = pd.read_csv(cfg.reports_dir / "rq3" / "pool.csv")
    corr = phase_a.a3_correlations(pool, plain)
    text = phase_a.render_a3(corr, plain, phase_a.header("a3", git_commit(cfg.root), prereg))
    folder = cfg.reports_dir / "addendum_a"
    write_frame(plain, folder / "a3_plain_gower.csv", cfg.raw_root)
    write_text(folder / "a3.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("b")
def addendum_b(
    analyse_only: bool = typer.Option(False, help="Only re-analyse reports/rq2-dada/rows.csv."),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum B: RQ2 under Dada's own train/test protocol (opens Dada's test files once)."""
    cfg = _cfg(config)
    try:
        prereg = require_committed(cfg.root, phase_b.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err
    folder = cfg.reports_dir / "rq2-dada"
    if not analyse_only:
        exp = load_experiment(Path(phase_b.CONFIG), cfg)
        result = engine_mod.run_experiment(
            cfg, exp, git_commit(cfg.root), progress=lambda m: typer.echo(m, err=True)
        )
        tables = (result.rows, result.detection, result.inputs)
        for name, frame in zip(RESULT_FILES, tables, strict=True):
            write_frame(frame, folder / name, cfg.raw_root)
        write_text(
            folder / "run.md",
            _run_summary(Path(phase_b.CONFIG), result, exp.test_splits),
            cfg.raw_root,
        )
    rows = pd.read_csv(folder / "rows.csv")
    result_b = phase_b.judge(rows)
    head = (
        f"Generated by `shortcutcheck addendum b` at commit `{git_commit(cfg.root)}` from "
        f"`reports/rq2-dada/rows.csv`. Rules: `{phase_b.FILE}` (committed in `{prereg[:7]}` "
        "before this experiment ran). The H2 verdict (judged under D5) is not changed."
    )
    text = phase_b.render(result_b, h2_mod.h2_tests(rows), head)
    write_frame(result_b.table, folder / "addendum_b_tests.csv", cfg.raw_root)
    write_text(folder / "addendum_b.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("c")
def addendum_c(
    analyse_only: bool = typer.Option(False, help="Only re-analyse reports/addendum_c/rows.csv."),
    dry_run: bool = typer.Option(
        False, help="Tiny run (one split, LR only, one seed) into a scratch folder; not a result."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum C: metamorphic augmentation fix, proof of concept (validation data only)."""
    cfg = _cfg(config)
    try:
        prereg = require_committed(cfg.root, fix_poc.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err
    folder = cfg.reports_dir / ("addendum_c_dry_run" if dry_run else "addendum_c")
    commit = git_commit(cfg.root)
    if not analyse_only:
        kwargs: dict[str, object] = {}
        if dry_run:
            kwargs = {"splits": ("rq12-DR-AG_deb",), "algorithms": ("lr",), "seeds": (11,)}
        rows = fix_poc.run(
            cfg,
            commit,
            progress=lambda m: typer.echo(m, err=True),
            log_mlflow=not dry_run,
            **kwargs,  # type: ignore[arg-type]
        )
        write_frame(rows, folder / "rows.csv", cfg.raw_root)
    rows = pd.read_csv(folder / "rows.csv")
    result = fix_poc.judge(rows)
    text = fix_poc.render(rows, result, fix_poc.header(commit, prereg))
    write_frame(result.tests, folder / "held_out_tests.csv", cfg.raw_root)
    write_text(folder / "addendum_c.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("d")
def addendum_d_cmd(
    part: str = typer.Option("all", help="h4, h5 or all."),
    analyse_only: bool = typer.Option(False, help="Only re-analyse the saved tables."),
    dry_run: bool = typer.Option(
        False,
        help="Same layout on validation partitions, LR and seed 11 only, 4 pool models; "
        "no test access; not a result.",
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum D: the fix on test inputs (H4) and decay on the 2015-16 holdout (H5)."""
    if part not in ("h4", "h5", "all"):
        raise typer.BadParameter("part must be h4, h5 or all")
    cfg = _cfg(config)
    try:
        prereg = require_committed(cfg.root, addendum_d.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err
    folder = cfg.reports_dir / ("addendum_d_dry_run" if dry_run else "addendum_d")
    commit = git_commit(cfg.root)
    small: dict[str, object] = {"algorithms": ("lr",), "seeds": (11,)} if dry_run else {}
    say = lambda m: typer.echo(m, err=True)  # noqa: E731
    if part in ("h4", "all"):
        if not analyse_only:
            rows = d_run.run_h4(cfg, commit, dry_run=dry_run, progress=say, **small)  # type: ignore[arg-type]
            write_frame(rows, folder / "h4_rows.csv", cfg.raw_root)
        rows = pd.read_csv(folder / "h4_rows.csv")
        r4 = addendum_d.judge_h4(rows)
        write_frame(r4.held_out, folder / "h4_held_out_tests.csv", cfg.raw_root)
        text = addendum_d.render_h4(rows, r4, addendum_d.header("--part h4", commit, prereg))
        write_text(folder / "h4.md", text, cfg.raw_root)
        typer.echo(text)
    if part in ("h5", "all"):
        if not analyse_only:
            exp = load_experiment(Path("configs/experiments/rq3.yaml"), cfg)
            only = None
            if dry_run:
                only = ["lr-q0-h0-s11", "rf-q0-h0-s11", "lgbm-q0.5-h0-s11", "mlp-q0.75-h0-s11"]
            h5a, h5b = d_run.run_h5(
                cfg,
                exp,
                commit,
                cfg.root / "data" / "models" / exp.experiment,
                dry_run=dry_run,
                pool_only=only,
                progress=say,
                **small,  # type: ignore[arg-type]
            )
            write_frame(h5a, folder / "h5a_rows.csv", cfg.raw_root)
            write_frame(h5b, folder / "h5b_pool.csv", cfg.raw_root)
        h5a = pd.read_csv(folder / "h5a_rows.csv")
        h5b = pd.read_csv(folder / "h5b_pool.csv")
        ra, rb = addendum_d.judge_h5a(h5a), addendum_d.judge_h5b(h5b)
        write_frame(rb.comparisons, folder / "h5c_comparisons.csv", cfg.raw_root)
        text = addendum_d.render_h5(h5a, ra, rb, addendum_d.header("--part h5", commit, prereg))
        write_text(folder / "h5.md", text, cfg.raw_root)
        typer.echo(text)


@addendum_app.command("e")
def addendum_e_cmd(
    analyse_only: bool = typer.Option(False, help="Only re-analyse the saved tables."),
    dry_run: bool = typer.Option(
        False,
        help="Same layout on the AZL validation partition, LR and RF, seed 11, 2 settings; "
        "no test access; not a result.",
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum E: replication on AZL, algorithm-adjusted H5b, fix vs behaviour-only."""
    cfg = _cfg(config)
    try:
        prereg = require_committed(cfg.root, addendum_e.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err
    folder = cfg.reports_dir / ("addendum_e_dry_run" if dry_run else "addendum_e")
    commit = git_commit(cfg.root)
    if not analyse_only:
        small: dict[str, object] = (
            {"algorithms": ("lr", "rf"), "seeds": (11,), "settings": 2} if dry_run else {}
        )
        arms, pool, androct = e_run.run_e(
            cfg, commit, dry_run=dry_run, progress=lambda m: typer.echo(m, err=True), **small  # type: ignore[arg-type]
        )
        write_frame(arms, folder / "e_arms.csv", cfg.raw_root)
        write_frame(pool, folder / "e_pool.csv", cfg.raw_root)
        write_frame(androct, folder / "e_androct_behaviour.csv", cfg.raw_root)
    arms = pd.read_csv(folder / "e_arms.csv")
    pool = pd.read_csv(folder / "e_pool.csv")
    androct = pd.read_csv(folder / "e_androct_behaviour.csv")
    result = addendum_e.judge(arms, pool)
    d_table = cfg.reports_dir / "addendum_d" / "h5b_pool.csv"
    adj = None
    if d_table.exists():
        t = pd.read_csv(d_table)
        adj = addendum_e.judge_adjusted(t[t["skew"] == 0], "AndroCT un-injected (addendum D)")
    e5_d = None
    d_rows = cfg.reports_dir / "addendum_d" / "h5a_rows.csv"
    if len(androct) and d_rows.exists():
        e5_d = addendum_e.judge_e5(e_run.androct_e5_rows(pd.read_csv(d_rows), androct))
    write_frame(result.e5, folder / "e5_tests.csv", cfg.raw_root)
    write_frame(result.e2.comparisons, folder / "e4_comparisons.csv", cfg.raw_root)
    text = addendum_e.render(
        arms, pool, result, addendum_e.header(commit, prereg), androct=adj, androct_e5=e5_d
    )
    write_text(folder / "addendum_e.md", text, cfg.raw_root)
    typer.echo(text)


@addendum_app.command("f")
def addendum_f_cmd(
    analyse_only: bool = typer.Option(False, help="Only re-analyse the saved tables."),
    dry_run: bool = typer.Option(
        False, help="One split, LR and RF, seed 11, into a scratch folder; not a result."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Addendum F: mechanism of the fix (training and validation data only)."""
    cfg = _cfg(config)
    try:
        prereg = require_committed(cfg.root, mechanism.FILE)
    except AddendumNotCommittedError as err:
        typer.echo(str(err))
        raise typer.Exit(code=2) from err
    folder = cfg.reports_dir / ("addendum_f_dry_run" if dry_run else "addendum_f")
    commit = git_commit(cfg.root)
    if not analyse_only:
        small: dict[str, object] = (
            {"splits": ("rq12-DR-AG_deb",), "algorithms": ("lr", "rf"), "seeds": (11,)}
            if dry_run
            else {}
        )
        rows, sep = mechanism.run(cfg, progress=lambda m: typer.echo(m, err=True), **small)  # type: ignore[arg-type]
        write_frame(rows, folder / "f_rows.csv", cfg.raw_root)
        write_frame(sep, folder / "f_separability.csv", cfg.raw_root)
    rows = pd.read_csv(folder / "f_rows.csv")
    sep = pd.read_csv(folder / "f_separability.csv")
    result = mechanism.judge(rows, sep)
    text = mechanism.render(rows, sep, result, mechanism.header(commit, prereg))
    write_text(folder / "addendum_f.md", text, cfg.raw_root)
    typer.echo(text)


@app.command("run")
def run_gate(
    model: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Trained model (joblib). Load only files you trust."
    ),
    data: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="CSV of apps to test; labels are not needed."
    ),
    train: Path = typer.Option(
        ..., "--train", exists=True, dir_okay=False, help="Labelled CSV the model was trained on."
    ),
    mrs: str = typer.Option("core", help="core (MR1, MR3, MR4), all, or a list such as MR1,MR4."),
    threshold: float = typer.Option(
        gate_mod.DEFAULT_THRESHOLD, help="Fail when any violation rate exceeds this."
    ),
    sample: int = typer.Option(
        gate_mod.DEFAULT_SAMPLE, help="Test at most this many inputs (uniform sample); 0 = all."
    ),
    seed: int = typer.Option(gate_mod.DEFAULT_SEED, help="Seed for sampling and the relations."),
    realism: bool = typer.Option(True, "--realism/--no-realism", help="Drop unrealistic inputs."),
    report: Path | None = typer.Option(
        None, help="Write the full JSON report (for the dashboard)."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Step 4.1: metamorphic gate. Exit code 0 = pass, 1 = a relation exceeds the threshold."""
    cfg = gate_mod.resolve_settings(config)
    try:
        result = gate_mod.check_files(
            model,
            data,
            train,
            cfg,
            mrs=gate_mod.parse_mrs(mrs),
            threshold=threshold,
            sample=sample,
            seed=seed,
            realism=realism,
        )
    except gate_mod.GateError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(code=2) from err
    typer.echo(result.render())
    if report is not None:
        result.write(report)
        typer.echo(f"\nreport written to {report}")
    if not result.passed:
        raise typer.Exit(code=1)


@app.command("dashboard")
def dashboard_cmd(
    report: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="JSON report from `shortcutcheck run --report`."
    ),
    port: int = typer.Option(8501, help="Port for the local Streamlit server."),
) -> None:
    """Step 4.3: open a gate report in the Streamlit dashboard."""
    import subprocess
    import sys

    from shortcutcheck import dashboard_app_path

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_app_path(),
        "--server.port",
        str(port),
    ]
    raise typer.Exit(code=subprocess.call([*cmd, "--", str(report)]))


@app.command("prereg-check")
def prereg_check(config: Path | None = ConfigOpt) -> None:
    """Step 3.2 Check: PREREGISTRATION.md was committed before any test access."""
    cfg = _cfg(config)
    result = check_preregistration(cfg)
    typer.echo(result.describe())
    if not result.passes:
        raise typer.Exit(code=1)


@app.command("synthetic")
def synthetic(
    out_dir: Path,
    seed: int = 0,
    scale: float = typer.Option(1.0, help="Multiply set sizes (use <1 for quick tests)."),
    config_out: Path | None = typer.Option(
        None, help="Also write a copy of the config whose RQ3 candidates match this data."
    ),
    config: Path | None = ConfigOpt,
) -> None:
    """Write a synthetic Dada-like zip for a dry run. Never report results from it."""
    typer.echo(f"Wrote {make_synthetic_dada(out_dir, seed, scale)}")
    if config_out is not None:
        base = config or load_config().root / "configs" / "phase1.yaml"
        typer.echo(f"Wrote {write_synthetic_config(base, config_out)}")


if __name__ == "__main__":
    app()
