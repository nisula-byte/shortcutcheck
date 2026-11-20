"""Known-answer tests on the frozen splits (roadmap step 2.5 Check).

For every split, three kinds of oracle model are fitted on the training partition and every
applicable MR is run on a seeded, stratified sample of validation apps (no labels needed).
Each (oracle, MR) pair has a known answer:

* ``exact 0``: the oracle cannot see what the MR changes, so no input may flip and no
  probability may move at all.
* ``expected p``: a stump on the column the MR changes; the observed rate must match the rate
  computed from the MR's sampling rule, within 4 standard errors (Poisson-binomial).

Runs without the realism filter: whether an input is excluded depends on its new value, which
would bias the comparison. Correctness of the filter is tested separately (step 2.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from shortcutcheck.config import Config
from shortcutcheck.data.dada import split_feature_groups
from shortcutcheck.data.splits import load_part
from shortcutcheck.mrs.base import MetamorphicRelation, NotApplicableError
from shortcutcheck.mrs.core import MetadataSwap, SizeScale, YearShift, core_mrs
from shortcutcheck.mrs.realism_report import stratified_sample
from shortcutcheck.mrs.secondary import secondary_mrs
from shortcutcheck.runner.engine import MRRun, ProbabilisticModel, run_mr
from shortcutcheck.runner.oracles import (
    Constant,
    Stump,
    expected_mr1,
    expected_mr3,
    expected_mr4,
)
from shortcutcheck.training.baselines import _setup_mlflow, build_model
from shortcutcheck.training.features import select

Z_TOLERANCE = 4.0


@dataclass(frozen=True)
class Answer:
    split: str
    oracle: str
    mr_id: str
    kind: str  # "exact 0" or "expected"
    n_counted: int
    observed: float
    expected: float
    tolerance: float
    max_abs_delta: float

    @property
    def testable(self) -> bool:
        """False when the MR changed no input (MR5 on a model that calls nothing malware)."""
        return self.n_counted > 0

    @property
    def passes(self) -> bool:
        if not self.testable:
            return True
        if self.kind == "exact 0":
            return self.observed == 0.0 and self.max_abs_delta == 0.0
        return abs(self.observed - self.expected) <= self.tolerance


def _expected(model: Stump, mr: MetamorphicRelation, run: MRRun, x: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(x[model.column], errors="coerce").to_numpy(dtype=float)
    if isinstance(mr, YearShift):
        return expected_mr1(model, mr, values)
    if isinstance(mr, SizeScale):
        return expected_mr3(model, mr, values)
    if isinstance(mr, MetadataSwap):
        return expected_mr4(model, mr, values)
    raise TypeError(mr)


def judge(
    split: str,
    oracle: str,
    model: ProbabilisticModel,
    mr: MetamorphicRelation,
    x: pd.DataFrame,
    seed: int,
) -> Answer:
    run = run_mr(model, mr, x, seed=seed)
    counted = run.counted
    n = int(counted.sum())
    observed = run.violation_rate if n else float("nan")
    max_delta = float(np.abs(run.p_after - run.p_before)[counted].max()) if n else float("nan")
    touches = isinstance(model, Stump) and model.column in mr.columns_touched
    if not touches:
        return Answer(split, oracle, mr.mr_id, "exact 0", n, observed, 0.0, 0.0, max_delta)
    assert isinstance(model, Stump)
    p = _expected(model, mr, run, x)[counted]
    expected = float(p.mean()) if n else float("nan")
    se = math.sqrt(float(np.sum(p * (1 - p)))) / n if n else float("nan")
    return Answer(
        split,
        oracle,
        mr.mr_id,
        "expected",
        n,
        observed,
        expected,
        Z_TOLERANCE * se + 1e-9,
        max_delta,
    )


def _mrs(columns: list[str], cfg: Config) -> list[MetamorphicRelation]:
    valid = (cfg.android_first_year, cfg.collection_year)
    return [*core_mrs(columns, valid), *secondary_mrs(columns)]


def check_split(cfg: Config, split: str, commit: str, max_eval: int = 5000) -> list[Answer]:
    seed = cfg.seeds[0]
    args = (cfg.metadata_columns, cfg.id_column, cfg.dropped_features)
    x_train, y_train = select(load_part(cfg, split, "train"), "full", *args)
    x_val, y_val = select(load_part(cfg, split, "val"), "full", *args)
    x_val = x_val.reindex(columns=x_train.columns)
    x_eval = x_val.iloc[stratified_sample(y_val, max_eval, seed)].reset_index(drop=True)

    _, behaviour = split_feature_groups(list(x_train.columns), cfg.metadata_columns)
    oracles: dict[str, tuple[ProbabilisticModel, set[str]]] = {}
    for name, column in (("year stump", "year"), ("size stump", "apk_size")):
        if column in x_train.columns:
            oracles[name] = (Stump(column).fit(x_train, y_train), set())
    oracles |= {
        # Behaviour features include permissions, so MR5/MR6 are not known answers for it.
        "behaviour-only RF": (
            build_model("rf", seed).fit(x_train[behaviour], y_train),
            {"MR5", "MR6"},
        ),
        "constant": (Constant().fit(x_train, y_train), set()),
    }
    answers = []
    for mr in _mrs(list(x_train.columns), cfg):
        try:
            mr.fit(x_train, y_train)
        except NotApplicableError:
            continue
        for name, (model, skip) in oracles.items():
            if mr.mr_id not in skip:
                answers.append(judge(split, name, model, mr, x_eval, seed))

    _setup_mlflow(cfg, f"phase2-known-answers-{split}")
    with mlflow.start_run(run_name=f"known-answers-s{seed}"):
        mlflow.set_tags({"step": "2.5", "split": split, "git_commit": commit})
        mlflow.log_params({"seed": seed, "max_eval": max_eval, "z_tolerance": Z_TOLERANCE})
        mlflow.log_metrics(
            {
                "n_answers": float(len(answers)),
                "n_pass": float(sum(a.passes for a in answers)),
                **{
                    f"{a.oracle.replace(' ', '_')}_{a.mr_id}_observed": a.observed
                    for a in answers
                    if not math.isnan(a.observed)
                },
            }
        )
    return answers


def render_markdown(answers: list[Answer], commit: str) -> str:
    tested = [a for a in answers if a.testable]
    n_pass = sum(a.passes for a in tested)
    lines = [
        "# Known-answer tests (roadmap step 2.5)",
        "",
        f"Generated by `shortcutcheck mrs known-answers`. Git commit `{commit}`. Do not edit "
        "by hand. Oracles are fitted on each split's training partition and run on a seeded, "
        "class-stratified sample of up to 5,000 validation apps, without the realism filter.",
        "",
        "- **exact 0**: the oracle cannot see what the MR changes. It passes only if no input "
        "flips and no probability moves at all.",
        "- **expected**: a stump on the column the MR changes. The expected rate is computed "
        "from the MR's sampling rule, and it passes if the observed rate is within "
        f"{Z_TOLERANCE:g} standard errors.",
        "",
        "- **not testable**: the MR changed no input, e.g. MR5 only changes apps the model "
        "calls malware, and a constant model may call none malware.",
        "",
        f"**Check: {'PASS' if n_pass == len(tested) else 'FAIL'}** "
        f"({n_pass} of {len(tested)} testable known answers; "
        f"{len(answers) - len(tested)} not testable).",
        "",
        "| Split | Oracle | MR | Kind | Counted | Observed | Expected | Tolerance "
        "| Max abs. change in p | Result |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for a in answers:
        exp = "0" if a.kind == "exact 0" else f"{a.expected:.4f}"
        tol = "exact" if a.kind == "exact 0" else f"±{a.tolerance:.4f}"
        if not a.testable:
            lines.append(
                f"| {a.split} | {a.oracle} | {a.mr_id} | {a.kind} | 0 | – | {exp} | {tol} | – "
                "| not testable |"
            )
            continue
        delta = "0" if a.max_abs_delta == 0 else f"{a.max_abs_delta:.2e}"
        lines.append(
            f"| {a.split} | {a.oracle} | {a.mr_id} | {a.kind} | {a.n_counted} "
            f"| {a.observed:.4f} | {exp} | {tol} | {delta} "
            f"| {'pass' if a.passes else '**FAIL**'} |"
        )
    return "\n".join(lines) + "\n"
