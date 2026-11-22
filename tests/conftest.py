from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from shortcutcheck.config import Config, load_config
from shortcutcheck.data.register import register_zip
from shortcutcheck.synthetic import SYNTHETIC_CANDIDATES as SYN_BASE
from shortcutcheck.synthetic import make_synthetic_dada

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[1]


SYNTHETIC_CANDIDATES = {
    **SYN_BASE,
    "SYN-missing": {
        "goodware": "NOPE-G",
        "malware": "NOPE-M",
        "early_years": [2014],
        "late_years": [2017],
    },
}


def make_project(tmp: Path, **overrides: object) -> Path:
    """A throwaway project root with the real config file, plus overrides.

    The RQ3 candidates are replaced by ones that match the synthetic data.
    """
    root = tmp / "project"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='tmp'\n")
    config = yaml.safe_load((REPO / "configs" / "phase1.yaml").read_text())
    config["rq3_candidates"] = SYNTHETIC_CANDIDATES
    config.update(overrides)
    (root / "configs" / "phase1.yaml").write_text(yaml.safe_dump(config))
    return root


@pytest.fixture(scope="session")
def synthetic_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_synthetic_dada(tmp_path_factory.mktemp("synthetic"), seed=0, scale=1.0)


@pytest.fixture()
def project(tmp_path: Path, synthetic_zip: Path, monkeypatch: pytest.MonkeyPatch) -> Config:
    """A project with the synthetic zip registered, as after roadmap step 1.3."""
    root = make_project(tmp_path, rq3_min_per_class_per_year=500)
    monkeypatch.setenv("SHORTCUTCHECK_ROOT", str(root))
    monkeypatch.chdir(root)
    cfg = load_config(root=root)
    register_zip(synthetic_zip, cfg.raw_dir, root / "DATA.md")
    yield cfg
    # Restore write permission so pytest can clean up the read-only raw folder.
    for path in [cfg.raw_root, *cfg.raw_root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    shutil.rmtree(root, ignore_errors=True)
