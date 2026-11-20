"""Synthetic Dada-like data for tests and for a dry run before the real zip arrives.

The generated sets mimic Dada's file layout and its known problems: a build-year shortcut in
the early set, duplicate hashes, hashes with conflicting labels, zero and 1980 ``dex_date``
values, and both headed and headerless label files. The numbers mean nothing; never report
results computed on this data.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SYNTHETIC_CANDIDATES = {
    "SYN-2014-vs-2017": {"labelled": ["SYN_ACT1*"], "early_years": [2014], "late_years": [2017]},
}

N_BEHAVIOUR = 30
INFORMATIVE = 6  # behaviour flags that genuinely depend on the label


def _hashes(rng: np.random.Generator, n: int) -> list[str]:
    return ["".join(rng.choice(list("0123456789abcdef"), 64)) for _ in range(n)]


def make_set(
    rng: np.random.Generator,
    n: int,
    malware_share: float,
    year_good: tuple[int, int],
    year_mal: tuple[int, int],
) -> pd.DataFrame:
    label = (rng.random(n) < malware_share).astype(int)
    year = np.where(
        label == 1,
        rng.integers(year_mal[0], year_mal[1] + 1, n),
        rng.integers(year_good[0], year_good[1] + 1, n),
    )
    dex = year.astype(object)
    bad = rng.random(n)
    dex[bad < 0.10] = 0
    dex[(bad >= 0.10) & (bad < 0.15)] = 1980
    frame = pd.DataFrame(
        {
            "sha256": _hashes(rng, n),
            "apk_size": rng.lognormal(15, 1, n).round(),
            "dex_date": dex,
            "year": year,
            "minSdkVersion": rng.integers(8, 22, n),
            "targetSdkVersion": rng.integers(15, 28, n),
        }
    )
    for j in range(N_BEHAVIOUR):
        informative = j < INFORMATIVE
        p = np.where(label == 1, 0.55, 0.30) if informative else np.full(n, 0.2)
        frame[f"flag_{j:02d}"] = (rng.random(n) < p).astype(int)
    frame["label"] = label
    return frame


def _add_noise(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dupes = frame.sample(frac=0.01, random_state=int(rng.integers(1 << 30)))
    conflicts = frame.sample(n=5, random_state=int(rng.integers(1 << 30))).copy()
    conflicts["label"] = 1 - conflicts["label"]
    return pd.concat([frame, dupes, conflicts], ignore_index=True)


def write_set(frame: pd.DataFrame, folder: Path, name: str, headed_labels: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["label"]).to_csv(folder / f"{name}.characteristics.csv", index=False)
    labels = frame[["sha256", "label"]]
    labels.to_csv(
        folder / f"{name}.goodmal.csv",
        index=False,
        header=["sha256", "goodmal"] if headed_labels else False,
    )
    (folder / f"{name}.sha256.txt").write_text("\n".join(frame["sha256"]) + "\n")


def make_synthetic_dada(out_dir: Path, seed: int = 0, scale: float = 1.0) -> Path:
    """Write a synthetic Dada-like folder and a zip of it. Returns the zip path."""
    rng = np.random.default_rng(seed)
    folder = out_dir / "datasets"

    def n(base: int) -> int:
        return max(60, int(base * scale))

    mixed = make_set(rng, n(3000), 0.30, (2013, 2016), (2010, 2013))
    write_set(_add_noise(mixed, rng), folder / "SYN-DR-AG", "SYN-DR-AG", headed_labels=True)
    early = make_set(rng, n(3000), 0.45, (2014, 2014), (2014, 2014))
    write_set(_add_noise(early, rng), folder / "SYN_ACT14", "SYN_ACT14", headed_labels=False)
    late = make_set(rng, n(2500), 0.45, (2017, 2017), (2017, 2017))
    write_set(late, folder / "SYN_ACT17", "SYN_ACT17", headed_labels=False)

    zip_path = out_dir / "synthetic-dada.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir))
    return zip_path


def write_synthetic_config(base_config: Path, out_path: Path) -> Path:
    """Copy a config, pointing its RQ3 candidates at the synthetic sets."""
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config["rq3_candidates"] = SYNTHETIC_CANDIDATES
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return out_path
