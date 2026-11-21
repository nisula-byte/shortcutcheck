"""Roadmap step 3.2: the pre-registered analysis, checked on cases with known answers."""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from shortcutcheck.analysis import figures, h1, h2, h3
from shortcutcheck.analysis.stats import (
    bootstrap_p_greater,
    cliffs_delta,
    cluster_bootstrap_rhos,
    holm,
    spearman,
    wilcoxon_greater,
)
from shortcutcheck.config import Config
from shortcutcheck.evaluation.preregistration import analysis_status, check_preregistration

ALGOS = ["lr", "svm", "rf", "lgbm", "mlp"]
SEEDS = [11, 22, 33, 44, 55]

# ---- statistics ------------------------------------------------------------------------


def test_holm_hand_computed() -> None:
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> monotone 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_cliffs_delta_hand_computed() -> None:
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0
    assert cliffs_delta([1, 3], [2]) == 0.0


def test_wilcoxon_greater() -> None:
    assert wilcoxon_greater([0.0] * 25).p_greater == 1.0
    d = np.arange(1, 26) / 100
    w = wilcoxon_greater(d)
    assert w.p_greater == pytest.approx(1 / 2**25)  # exact: every rank positive
    mixed = np.array([0.3, -0.1, 0.2, 0.05, -0.02, 0.4, 0.1, 0.0, 0.25, -0.3])
    w = wilcoxon_greater(mixed)
    ref = stats.wilcoxon(mixed, alternative="greater", zero_method="wilcox")
    assert w.p_greater == pytest.approx(ref.pvalue) and w.n_nonzero == 9


def test_spearman_and_bootstrap_p() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert bootstrap_p_greater(np.ones(99)) == pytest.approx(1 / 100)
    assert bootstrap_p_greater(-np.ones(99)) == 1.0


def test_cluster_bootstrap_keeps_clusters_and_is_seeded() -> None:
    t = pd.DataFrame({"c": np.repeat(np.arange(10), 5), "x": np.arange(50.0)})
    t["y"] = t["x"] * 2
    a = cluster_bootstrap_rhos(t, ["x"], "y", "c", 50, seed=3)
    assert np.allclose(a["x"], 1.0)  # perfectly monotone in every resample
    t["noise"] = np.random.default_rng(0).normal(size=50)
    b1 = cluster_bootstrap_rhos(t, ["noise"], "y", "c", 50, seed=3)
    b2 = cluster_bootstrap_rhos(t, ["noise"], "y", "c", 50, seed=3)
    assert b1.equals(b2) and b1["noise"].std() > 0


# ---- H1 ------------------------------------------------------------------------------------


def rq1_rows(rates: dict[tuple[str, str], float], low: float = 0.03) -> pd.DataFrame:
    """rates[(split, algorithm)] = MR4 rate for every seed (full features); others 0."""
    out = []
    for split, a, s, fs, mr in itertools.product(
        h1.BIASED_SPLITS, ALGOS, SEEDS, ["full", "behaviour"], ["MR1", "MR4", "MR6"]
    ):
        rate = rates.get((split, a), 0.0) if (fs == "full" and mr == "MR4") else 0.0
        if mr == "MR6":
            rate = 0.5  # secondary MRs never count
        out.append(
            {
                "run": split,
                "train_split": split,
                "role": "own",
                "algorithm": a,
                "feature_set": fs,
                "seed": s,
                "mr": mr,
                "stratum": "all",
                "rate": rate,
                "rate_low": low if rate > 0 else 0.0,
                "n_violations": int(rate * 100),
                "max_abs_delta": rate,
                "n_counted": 100,
                "rate_unfiltered": rate,
                "delta": rate,
            }
        )
    return pd.DataFrame(out)


def three_each() -> dict[tuple[str, str], float]:
    return {(s, a): 0.08 for s in h1.BIASED_SPLITS for a in ("lr", "rf", "lgbm")}


def test_h1_supported_partial_and_not() -> None:
    assert h1.judge_h1(rq1_rows(three_each())).verdict == "SUPPORTED"
    one_split = {k: v for k, v in three_each().items() if k[0] == "rq12-DR-AG"}
    r = h1.judge_h1(rq1_rows(one_split))
    assert r.verdict == "PARTIALLY SUPPORTED" and r.splits[0].algorithms_showing == [
        "lgbm",
        "lr",
        "rf",
    ]
    assert h1.judge_h1(rq1_rows({})).verdict == "NOT SUPPORTED"


def test_h1_thresholds_are_strict() -> None:
    assert h1.judge_h1(rq1_rows({k: 0.049 for k in three_each()})).verdict == "NOT SUPPORTED"
    assert h1.judge_h1(rq1_rows(three_each(), low=0.01)).verdict == "NOT SUPPORTED"
    df = rq1_rows(three_each())
    gone = (df["algorithm"] == "lr") & (df["seed"] == 33) & (df["mr"] == "MR4")
    df.loc[gone & (df["feature_set"] == "full"), ["rate", "rate_low"]] = np.nan
    r = h1.judge_h1(df)  # a missing seed disqualifies lr: 2 of 5 left
    assert r.verdict == "NOT SUPPORTED" and "lr" not in r.splits[0].algorithms_showing


def test_h1_control_failure_stops_the_judgement() -> None:
    df = rq1_rows(three_each())
    df.loc[(df["feature_set"] == "behaviour") & (df["mr"] == "MR1"), "n_violations"] = 1
    r = h1.judge_h1(df)
    assert not r.control_ok and r.verdict.startswith("NOT JUDGED")
    df = rq1_rows(three_each())
    df.loc[(df["feature_set"] == "behaviour") & (df["mr"] == "MR6"), "n_violations"] = 9
    assert h1.judge_h1(df).control_ok  # permissions are behaviour: MR6 may move them


# ---- H2 ------------------------------------------------------------------------------------


def rq2_rows(gap: dict[str, float], size_gap: dict[str, float]) -> pd.DataFrame:
    """Biased MR4 rate = debiased + gap[family] (+ a seed-specific wobble)."""
    out = []
    rng = np.random.default_rng(1)
    for family, role, (a, s) in itertools.product(
        h2.FAMILIES, h2.INPUTS, itertools.product(ALGOS, SEEDS)
    ):
        base = 0.02 + rng.uniform(0, 0.01)
        for run, rate in (
            (family, base + gap[family]),
            (f"{family}-size-matched", base + size_gap[family]),
            (f"{family}_deb", base),
        ):
            for mr in ("MR1", "MR4"):
                out.append(
                    {
                        "run": run,
                        "role": role,
                        "algorithm": a,
                        "seed": s,
                        "mr": mr,
                        "feature_set": "full",
                        "stratum": "all",
                        "eval_split": "x",
                        "rate": rate if mr == "MR4" else base,
                    }
                )
    return pd.DataFrame(out)


def test_h2_supported_and_robust() -> None:
    r = h2.judge_h2(rq2_rows({"DR-AG": 0.05, "VS-AG": 0.04}, {"DR-AG": 0.03, "VS-AG": 0.0}))
    assert r.verdict == "SUPPORTED"
    assert r.robust == {"DR-AG": True, "VS-AG": False}
    primary = r.tests[(r.tests["comparison"] == "published") & (r.tests["inputs"] == "h2-deb")]
    assert len(primary) == 4  # 2 families x 2 MRs present: one Holm family
    mr4 = primary[primary["mr"] == "MR4"]
    assert (mr4["n_pairs"] == 25).all() and (mr4["cliffs_delta"] == 1.0).all()
    # Holm runs over the 4 tests of this (comparison, inputs) family, not over all tests.
    assert np.allclose(primary["p_holm"], holm(primary["p_greater"].tolist()))
    assert (mr4["p_holm"] < h2.ALPHA).all()


def test_h2_partial_and_not() -> None:
    assert (
        h2.judge_h2(rq2_rows({"DR-AG": 0.05, "VS-AG": 0.0}, {"DR-AG": 0, "VS-AG": 0})).verdict
        == "PARTIALLY SUPPORTED"
    )
    assert (
        h2.judge_h2(rq2_rows({"DR-AG": -0.05, "VS-AG": 0.0}, {"DR-AG": 0, "VS-AG": 0})).verdict
        == "NOT SUPPORTED"
    )


# ---- H3 ------------------------------------------------------------------------------------


def pool(kind: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for algorithm, (skew, n_settings) in itertools.product(
        ALGOS, zip([0.0, 0.25, 0.5, 0.75], [6, 2, 2, 2], strict=True)
    ):
        for setting in range(n_settings):
            level = rng.uniform(0, 1)  # configuration-level shortcut reliance
            for s in SEEDS:
                drop = level + rng.normal(0, 0.05) + (skew * 3 if kind == "injected" else 0)
                mr4 = {
                    "strong": level + rng.normal(0, 0.05),
                    "tie": level + rng.normal(0, 0.05),
                    "injected": rng.uniform(0, 1) + skew * 3,
                    "none": rng.uniform(0, 1),
                }[kind]
                out.append(
                    {
                        "model": f"{algorithm}-{skew}-{setting}-{s}",
                        "algorithm": algorithm,
                        "skew": skew,
                        "setting": setting,
                        "seed": s,
                        "mr4_pre": mr4,
                        "shap_share": (
                            (level + rng.normal(0, 0.05)) if kind == "tie" else rng.uniform(0, 1)
                        ),
                        "confidence_drop": rng.uniform(0, 1),
                        "psi": rng.uniform(0, 1),
                        "atc_drop": rng.uniform(0, 1),
                        "f1_drop": drop,
                    }
                )
    return pd.DataFrame(out)


@pytest.mark.parametrize(
    ("kind", "outcome"),
    [
        ("strong", "Strong signal"),
        ("tie", "Signal, no edge"),
        ("injected", "Injected only"),
        ("none", "No signal"),
    ],
)
def test_h3_maps_each_case_to_its_gate3_outcome(kind: str, outcome: str) -> None:
    r = h3.judge_h3(pool(kind), resamples=300)
    assert r.headline.n_models == 150 and r.headline.n_configurations == 30
    assert r.full_pool.n_configurations == 60
    assert r.gate3 == outcome


def test_h3_needs_every_column_and_reports_missing_models() -> None:
    with pytest.raises(ValueError):
        h3.judge_h3(pool("strong").drop(columns=["psi"]), resamples=50)
    p = pool("strong")
    p.loc[0, "shap_share"] = np.nan
    assert h3.judge_h3(p, resamples=50).missing == 1


# ---- reports and figures -------------------------------------------------------------------


def test_reports_render_and_figures_are_byte_identical(tmp_path: Path) -> None:
    rows = rq1_rows(three_each())
    det = rows[["train_split", "algorithm", "feature_set", "seed"]].drop_duplicates()
    det = det.assign(f1=0.9, pr_auc=0.9, roc_auc=0.9, fpr_at_95tpr=0.1)
    text = h1.render_h1(h1.judge_h1(rows), rows, det, "hdr")
    assert "**H1: SUPPORTED.**" in text and "| rq12-VS-AG | 3: lgbm, lr, rf | holds |" in text
    r2 = rq2_rows({"DR-AG": 0.05, "VS-AG": 0.04}, {"DR-AG": 0.0, "VS-AG": 0.0})
    assert "**H2: SUPPORTED.**" in h2.render_h2(h2.judge_h2(r2), r2, "hdr")
    p = pool("strong")
    r3 = h3.judge_h3(p, resamples=300)  # 50 resamples cannot reach Holm p < 0.05 (min 1/51)
    assert "Strong signal" in h3.render_h3(r3, "hdr", h3.secondary(p, resamples=50))
    for name, draw in (
        ("a", lambda path: figures.rq1_violation_rates(rows, path, h1.CORE_MRS)),
        ("b", lambda path: figures.rq2_pairs(r2, path)),
        ("c", lambda path: figures.rq3_scatter(p, path)),
    ):
        first, second = tmp_path / f"{name}1.svg", tmp_path / f"{name}2.svg"
        draw(first)
        draw(second)
        assert first.read_bytes() == second.read_bytes()


# ---- step 3.2 Check ------------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _log(cfg: Config, split: str, utc: str) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.logs_dir / "test_access.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": utc, "split": split, "purpose": "x"}) + "\n")


def test_preregistration_check(project: Config) -> None:
    root = project.root
    _git(root, "init", "-q")
    assert not check_preregistration(project).passes  # nothing committed yet
    assert "not yet pre-registered" in analysis_status(root)

    analysis = root / "src" / "shortcutcheck" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "h1.py").write_text("RATE_MIN = 0.05\n")
    (root / "PREREGISTRATION.md").write_text("# plan\n")
    _git(root, "add", "PREREGISTRATION.md", "src")
    _git(root, "commit", "-q", "-m", "prereg")
    rq3 = f"rq3-{project.rq3_primary}"

    result = check_preregistration(project)
    assert result.passes and result.first_rq3_access is None
    _log(project, rq3, "2100-01-01T00:00:00+00:00")
    assert check_preregistration(project).passes  # 2017 labels opened after the commit
    _log(project, "rq12-DR-AG", "2000-01-01T00:00:00+00:00")
    result = check_preregistration(project)
    assert not result.passes and "rq12-DR-AG" in result.describe()

    assert "identical to the pre-registered version" in analysis_status(root)
    (analysis / "h1.py").write_text("RATE_MIN = 0.03\n")
    assert "CHANGED since the pre-registered version" in analysis_status(root)


def test_analyse_cli(project: Config) -> None:
    from typer.testing import CliRunner

    from shortcutcheck.cli import app

    runner = CliRunner()
    folder = project.reports_dir / "rq1"
    folder.mkdir(parents=True)
    rows = rq1_rows(three_each())
    rows.to_csv(folder / "rows.csv", index=False)
    det = rows[["train_split", "algorithm", "feature_set", "seed"]].drop_duplicates()
    det.assign(f1=0.9, pr_auc=0.9, roc_auc=0.9, fpr_at_95tpr=0.1).to_csv(
        folder / "detection.csv", index=False
    )
    (folder / "run.md").write_text("# x\n\nGenerated by y. Git commit `abc1234`. More.\n")
    result = runner.invoke(app, ["analyse", "rq1"])
    assert result.exit_code == 0, result.output
    text = (folder / "h1.md").read_text()
    assert "experiment run at commit `abc1234`" in text and "**H1: SUPPORTED.**" in text
    assert "not yet pre-registered" in text
    assert (folder / "rq1_violation_rates.svg").exists()
    assert runner.invoke(app, ["analyse", "rq2"]).exit_code == 2  # no rq2 results yet
    assert runner.invoke(app, ["analyse", "rq3"]).exit_code == 2  # pool built in 3.5-3.7
    assert runner.invoke(app, ["prereg-check"]).exit_code == 1  # nothing committed
