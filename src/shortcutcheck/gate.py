"""ShortcutCheck gate (roadmap 4.1): test a trained detector for metadata sensitivity.

This is the reusable tool behind ``shortcutcheck run``, the pytest plugin and the dashboard.
It needs the trained model, the training data it was trained on (to fit the metamorphic
relations and the realism check) and unlabelled inputs to test. No labels of the inputs are
used. The gate fails when the violation rate of any applicable relation exceeds the threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd

from shortcutcheck.config import Config, load_config
from shortcutcheck.data.dada import LABEL
from shortcutcheck.evaluation.experiment import fit_mrs
from shortcutcheck.mrs.base import MetamorphicRelation
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.runner.engine import malware_probability, run_mr
from shortcutcheck.runner.stats import summarise
from shortcutcheck.training.features import numeric_features

CORE = ("MR1", "MR3", "MR4")
ALL = (*CORE, "MR5", "MR6")
KNOWN = {*ALL, "MR2"}
REPORT_VERSION = 1
DEFAULT_THRESHOLD = 0.05
DEFAULT_SAMPLE = 5000
DEFAULT_SEED = 11
MR_NAMES = {
    "MR1": "build year moved to another valid year",
    "MR2": "SDK version pair moved",
    "MR3": "package size rescaled (x0.5 to x2)",
    "MR4": "all metadata swapped with an app of the other predicted class",
    "MR5": "one normal permission added (malware must stay malware)",
    "MR6": "one dangerous permission added (malware must stay malware)",
}


class Settings(Protocol):
    """What the gate needs to know about the feature table (a project Config satisfies it)."""

    @property
    def metadata_columns(self) -> list[str]: ...
    @property
    def id_column(self) -> str: ...
    @property
    def dropped_features(self) -> list[str]: ...
    @property
    def android_first_year(self) -> int: ...
    @property
    def collection_year(self) -> int: ...


@dataclass(frozen=True)
class GateSettings:
    """Built-in defaults for Dada-format feature tables, used when no project config exists."""

    metadata_columns: list[str] = field(
        default_factory=lambda: [
            "apk_size",
            "APK size",
            "dex_date",
            "year",
            "Year",
            "minSdkVersion",
            "targetSdkVersion",
        ]
    )
    id_column: str = "sha256"
    dropped_features: list[str] = field(
        default_factory=lambda: [
            "dex_date",
            "APK size",
            "Year",
            "Internet Permission",
            "External storage",
            "Uses Play Services",
            "Generates UUIDs",
            "Vibrate phone",
            "NFC",
            "Bluetooth",
            "Uses HTTP",
            "Uses JSON",
            "Specify User-Agent",
        ]
    )
    android_first_year: int = 2008
    collection_year: int = 2021


def resolve_settings(config: Path | None = None) -> Settings:
    """The given project config, else this project's config if found, else built-in defaults."""
    if config is not None:
        return load_config(config)
    try:
        return load_config()
    except (FileNotFoundError, RuntimeError, OSError, KeyError):
        return GateSettings()


class GateError(ValueError):
    """Bad input to the gate (unknown relation, missing labels, unusable model)."""


def tool_version() -> str:
    try:
        return version("shortcutcheck")
    except PackageNotFoundError:  # pragma: no cover - only outside an installed package
        return "unknown"


def parse_mrs(spec: str) -> tuple[str, ...]:
    """``core``, ``all`` or a comma-separated list such as ``MR1,MR4``."""
    key = spec.strip().lower()
    if key == "core":
        return CORE
    if key == "all":
        return ALL
    wanted = tuple(dict.fromkeys(s.strip().upper() for s in spec.split(",") if s.strip()))
    unknown = [m for m in wanted if m not in KNOWN]
    if not wanted or unknown:
        raise GateError(f"unknown relation(s) {unknown or spec!r}; use core, all or e.g. MR1,MR4")
    return wanted


def features(frame: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Numeric model features of a Dada-style table (labels and dropped columns removed)."""
    x = numeric_features(frame.drop(columns=[LABEL], errors="ignore"), cfg.id_column)
    return x.drop(columns=[c for c in cfg.dropped_features if c in x.columns])


def labels(frame: pd.DataFrame) -> pd.Series:
    if LABEL not in frame.columns:
        raise GateError(f"the training table needs a {LABEL!r} column (1 = malware)")
    return frame[LABEL].astype(int)


def load_model(path: Path) -> Any:
    """Load a joblib/pickle model. Only load files you trust: unpickling can run code."""
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise GateError(f"{path} has no predict_proba(); ShortcutCheck needs probabilities")
    return model


# ---- report -------------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """JSON-safe copy: NaN becomes null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _num(value: Any) -> float:
    return math.nan if value is None else float(value)


@dataclass
class MRResult:
    mr: str
    description: str
    applicable: bool
    rate: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    n_counted: int = 0
    n_violations: int = 0
    n_unrealistic: int = 0
    rate_unfiltered: float = math.nan
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
        }


@dataclass
class Example:
    """One input whose prediction flipped, with the metadata the relation changed."""

    mr: str
    row: int
    app_id: str
    p_before: float
    p_after: float
    changes: dict[str, list[float]]


@dataclass
class FeatureRate:
    """MR4 restricted to one metadata field: flips when only that field takes donor values."""

    feature: str
    rate: float
    n_counted: int
    n_violations: int


@dataclass
class GateReport:
    model: str
    data: str
    train: str
    threshold: float
    seed: int
    n_inputs: int
    n_checked: int
    realism: bool
    mrs: list[MRResult]
    examples: list[Example] = field(default_factory=list)
    feature_rates: list[FeatureRate] = field(default_factory=list)
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    version: str = field(default_factory=tool_version)
    report_version: int = REPORT_VERSION

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.mrs)

    @property
    def failing(self) -> list[str]:
        return [r.mr for r in self.mrs if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["passed"] = self.passed
        return dict(_clean(out))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> GateReport:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("report_version") != REPORT_VERSION:
            raise GateError(f"{path}: unsupported report version {d.get('report_version')}")
        mrs = [
            MRResult(
                mr=r["mr"],
                description=r["description"],
                applicable=r["applicable"],
                rate=_num(r["rate"]),
                ci_low=_num(r["ci_low"]),
                ci_high=_num(r["ci_high"]),
                n_counted=r["n_counted"],
                n_violations=r["n_violations"],
                n_unrealistic=r["n_unrealistic"],
                rate_unfiltered=_num(r["rate_unfiltered"]),
                passed=r["passed"],
            )
            for r in d["mrs"]
        ]
        return cls(
            model=d["model"],
            data=d["data"],
            train=d["train"],
            threshold=d["threshold"],
            seed=d["seed"],
            n_inputs=d["n_inputs"],
            n_checked=d["n_checked"],
            realism=d["realism"],
            mrs=mrs,
            examples=[Example(**e) for e in d.get("examples", [])],
            feature_rates=[
                FeatureRate(f["feature"], _num(f["rate"]), f["n_counted"], f["n_violations"])
                for f in d.get("feature_rates", [])
            ],
            created_utc=d["created_utc"],
            version=d["version"],
            report_version=d["report_version"],
        )

    def render(self) -> str:
        lines = [
            f"ShortcutCheck {self.version}: {'PASS' if self.passed else 'FAIL'} "
            f"(threshold {self.threshold:g}, {self.n_checked} of {self.n_inputs} inputs checked, "
            f"realism check {'on' if self.realism else 'off'})",
            "",
            f"{'MR':<5}{'rate':>8}  {'95% CI':<17}{'counted':>9}{'flips':>7}  result",
        ]
        for r in self.mrs:
            if not r.applicable:
                lines.append(f"{r.mr:<5}{'n/a':>8}  {'':<17}{'':>9}{'':>7}  not applicable")
                continue
            ci = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
            verdict = "ok" if r.passed else f"FAIL (> {self.threshold:g})"
            lines.append(
                f"{r.mr:<5}{r.rate:>8.3f}  {ci:<17}{r.n_counted:>9}{r.n_violations:>7}  {verdict}"
            )
        if self.feature_rates:
            lines += ["", "MR4 by metadata field (only that field changed):"]
            for f in (f for f in self.feature_rates if f.n_counted):
                rate = "n/a" if math.isnan(f.rate) else f"{f.rate:.3f}"
                lines.append(f"  {f.feature:<20}{rate:>8}  ({f.n_violations}/{f.n_counted})")
        return "\n".join(lines)


# ---- the check ----------------------------------------------------------------------------


def _sample(n: int, size: int, seed: int) -> np.ndarray:
    if size <= 0 or n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


def _examples(
    mr: MetamorphicRelation,
    model: Any,
    x: pd.DataFrame,
    run: Any,
    ids: pd.Series,
    seed: int,
    limit: int,
) -> list[Example]:
    flipped = np.flatnonzero(run.violated & run.counted)[:limit]
    if not len(flipped):
        return []
    # Same seed and predicted labels as run_mr, so the transform is identical.
    predicted = (run.p_before >= run.threshold).astype(int)
    x_new = mr.transform(x, rng=np.random.default_rng(seed), predicted=predicted)
    out = []
    for i in flipped:
        changed = {
            c: [float(x.iloc[i][c]), float(x_new.iloc[i][c])]
            for c in mr.columns_touched
            if c in x.columns and x.iloc[i][c] != x_new.iloc[i][c]
        }
        out.append(
            Example(
                mr.mr_id,
                int(i),
                str(ids.iloc[i]),
                float(run.p_before[i]),
                float(run.p_after[i]),
                changed,
            )
        )
    return out


def _feature_rates(
    mr4: MetamorphicRelation,
    model: Any,
    x: pd.DataFrame,
    realism: RealismCheck | None,
    seed: int,
    threshold: float = 0.5,
) -> list[FeatureRate]:
    p_before = malware_probability(model, x)
    predicted = (p_before >= threshold).astype(int)
    donor = mr4.transform(x, rng=np.random.default_rng(seed), predicted=predicted)
    out = []
    for col in mr4.columns_touched:
        if col not in x.columns:
            continue
        x_one = x.copy()
        x_one[col] = donor[col].to_numpy()
        changed = (x_one[col] != x[col]).to_numpy()
        counted = changed.copy()
        if realism is not None and changed.any():
            counted[changed] = realism.is_realistic(x_one.loc[changed])
        p_after = malware_probability(model, x_one)
        flips = ((p_after >= threshold) != (p_before >= threshold)) & counted
        n = int(counted.sum())
        out.append(FeatureRate(col, float(flips.sum() / n) if n else math.nan, n, int(flips.sum())))
    return out


def check(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_data: pd.DataFrame,
    cfg: Settings,
    *,
    mrs: tuple[str, ...] = CORE,
    threshold: float = DEFAULT_THRESHOLD,
    sample: int = DEFAULT_SAMPLE,
    seed: int = DEFAULT_SEED,
    realism: bool = True,
    ids: pd.Series | None = None,
    n_examples: int = 20,
    names: tuple[str, str, str] = ("model", "data", "train"),
) -> GateReport:
    """Run the metamorphic relations on (a sample of) ``x_data`` and judge them."""
    if not 0 <= threshold <= 1:
        raise GateError("threshold must be between 0 and 1")
    x_data = x_data.reindex(columns=x_train.columns)
    keep = _sample(len(x_data), sample, seed)
    xs = x_data.iloc[keep].reset_index(drop=True)
    id_series = (
        pd.Series(ids).iloc[keep].reset_index(drop=True)
        if ids is not None
        else pd.Series([str(i) for i in keep])
    )
    meta = [c for c in cfg.metadata_columns if c in x_train.columns]
    check_real = RealismCheck(seed=seed, metadata_columns=meta).fit(x_train) if realism else None
    # fit_mrs reads only the Settings fields of a Config.
    fitted = {m.mr_id: m for m in fit_mrs(cast(Config, cfg), x_train, y_train, mrs)}
    results, examples = [], []
    for mr_id in mrs:
        if mr_id not in fitted:
            results.append(MRResult(mr_id, MR_NAMES.get(mr_id, mr_id), applicable=False))
            continue
        mr = fitted[mr_id]
        run = run_mr(model, mr, xs, seed=seed, realism=check_real)
        s = summarise(run, seed=seed)
        if s.n_counted == 0:
            results.append(MRResult(mr_id, MR_NAMES.get(mr_id, mr_id), applicable=False))
            continue
        results.append(
            MRResult(
                mr_id,
                MR_NAMES.get(mr_id, mr_id),
                applicable=True,
                rate=s.rate,
                ci_low=s.rate_wilson.low,
                ci_high=s.rate_wilson.high,
                n_counted=s.n_counted,
                n_violations=s.n_violations,
                n_unrealistic=s.n_unrealistic,
                rate_unfiltered=s.rate_unfiltered,
                passed=bool(s.rate <= threshold),
            )
        )
        examples += _examples(mr, model, xs, run, id_series, seed, n_examples)
    feature_rates = (
        _feature_rates(fitted["MR4"], model, xs, check_real, seed) if "MR4" in fitted else []
    )
    return GateReport(
        model=names[0],
        data=names[1],
        train=names[2],
        threshold=threshold,
        seed=seed,
        n_inputs=len(x_data),
        n_checked=len(xs),
        realism=realism,
        mrs=results,
        examples=examples,
        feature_rates=feature_rates,
    )


def check_files(
    model_path: Path,
    data_path: Path,
    train_path: Path,
    cfg: Settings,
    **kwargs: Any,
) -> GateReport:
    """``check`` on files: a joblib model, an input CSV and the labelled training CSV."""
    model = load_model(model_path)
    train = pd.read_csv(train_path, low_memory=False)
    data = pd.read_csv(data_path, low_memory=False)
    ids = data[cfg.id_column].astype(str) if cfg.id_column in data.columns else None
    return check(
        model,
        features(train, cfg),
        labels(train),
        features(data, cfg),
        cfg,
        ids=ids,
        names=(str(model_path), str(data_path), str(train_path)),
        **kwargs,
    )
