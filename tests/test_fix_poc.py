"""Addendum C: the augmentation and the decision rules, on data with known answers."""

from __future__ import annotations

import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from shortcutcheck.addenda import fix_poc
from shortcutcheck.mrs.core import MetadataSwap


def toy(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Malware: old years and small APKs; goodware: new years and big APKs."""
    rng = np.random.default_rng(seed)
    y = pd.Series(np.repeat([0, 1], n // 2))
    x = pd.DataFrame(
        {
            "year": np.where(y == 1, rng.integers(2010, 2013, n), rng.integers(2016, 2019, n)),
            "apk_size": np.where(y == 1, rng.integers(1, 100, n), rng.integers(500, 900, n)),
            "perm_a": rng.integers(0, 2, n),
            "perm_b": rng.integers(0, 2, n),
        }
    )
    return x, y


def fitted_mr4(x: pd.DataFrame, y: pd.Series) -> MetadataSwap:
    return MetadataSwap(["year", "apk_size"], "year", (2008, 2021)).fit(x, y)


def test_augment_copies_wear_the_other_class_metadata_and_keep_label_and_behaviour() -> None:
    x, y = toy()
    x_aug, y_aug, rejected = fix_poc.augment(x, y, fitted_mr4(x, y), None, seed=11)
    assert rejected == 0
    assert len(x_aug) == 2 * len(x)
    copy, labels = x_aug.iloc[len(x) :].reset_index(drop=True), y_aug.iloc[len(x) :]
    assert (labels.to_numpy() == y.to_numpy()).all()
    pd.testing.assert_frame_equal(copy[["perm_a", "perm_b"]], x[["perm_a", "perm_b"]])
    # Malware copies carry goodware metadata (new years, big APKs), and the reverse.
    mal = y.to_numpy() == 1
    assert (copy.loc[mal, "year"] >= 2016).all() and (copy.loc[mal, "apk_size"] >= 500).all()
    assert (copy.loc[~mal, "year"] <= 2012).all() and (copy.loc[~mal, "apk_size"] < 100).all()


def test_augment_is_reproducible_and_seed_dependent() -> None:
    x, y = toy()
    mr4 = fitted_mr4(x, y)
    a = fix_poc.augment(x, y, mr4, None, seed=11)[0]
    b = fix_poc.augment(x, y, mr4, None, seed=11)[0]
    c = fix_poc.augment(x, y, mr4, None, seed=22)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


class _RejectHalf:
    def is_realistic(self, x: pd.DataFrame) -> np.ndarray:
        return np.arange(len(x)) % 2 == 0


def test_augment_drops_unrealistic_copies_and_counts_them() -> None:
    x, y = toy(100)
    x_aug, y_aug, rejected = fix_poc.augment(x, y, fitted_mr4(x, y), _RejectHalf(), seed=11)  # type: ignore[arg-type]
    assert rejected == 50
    assert len(x_aug) == len(y_aug) == 150


def test_contrary_apps_are_the_ones_metadata_gets_wrong() -> None:
    x, y = toy(400)
    x_val, y_val = toy(40, seed=1)
    y_flip = y_val.copy()
    y_flip.iloc[:5] = 1 - y_flip.iloc[:5]  # these apps' metadata now points the wrong way
    mask = fix_poc.contrary_apps(x, y, x_val, y_flip)
    assert mask[:5].all() and not mask[5:].any()


def rows(effect: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """effect[split] = (MR4 augmented / baseline ratio, F1 drop, MR1/MR3 drop)."""
    rng = np.random.default_rng(5)
    out = []
    for split, (alg, seed) in itertools.product(effect, itertools.product("abcde", (11, 22, 33))):
        ratio, f1_drop, held = effect[split]
        base = {
            "rate_MR4": 0.2 + rng.uniform(0, 0.01),
            "f1": 0.9,
            "rate_MR1": 0.1,
            "rate_MR3": 0.05,
        }
        noise = rng.uniform(-0.002, 0.002)
        aug = {
            "rate_MR4": base["rate_MR4"] * ratio,
            "f1": 0.9 - f1_drop,
            "rate_MR1": 0.1 - held - noise,
            "rate_MR3": 0.05,
        }
        for arm, m in (("baseline", base), ("augmented", aug)):
            out.append({"split": split, "algorithm": alg, "seed": seed, "arm": arm, **m})
    frame = pd.DataFrame(out)
    frame["contrary_acc"] = 0.5
    frame["roc_auc"] = 0.95
    return frame


@pytest.mark.parametrize(
    ("effect", "verdict", "passing"),
    [
        ({s: (0.3, 0.01, 0.05) for s in "wxyz"}, "GO", 4),
        (
            {
                "w": (0.3, 0.01, 0.05),
                "x": (0.3, 0.01, 0.05),
                "y": (0.3, 0.05, 0.05),
                "z": (0.8, 0, 0.05),
            },
            "PARTIAL",
            2,
        ),
        ({s: (0.3, 0.01, 0.0) for s in "wxyz"}, "NO-GO", 0),  # held-out MRs do not fall
        ({s: (0.6, 0.0, 0.05) for s in "wxyz"}, "NO-GO", 0),  # MR4 not halved
    ],
)
def test_decision_rules(
    effect: dict[str, tuple[float, float, float]], verdict: str, passing: int
) -> None:
    result = fix_poc.judge(rows(effect))
    assert result.verdict == verdict
    assert result.n_passing == passing


def test_holm_covers_every_held_out_test() -> None:
    result = fix_poc.judge(rows({s: (0.3, 0.01, 0.05) for s in "wxyz"}))
    t = result.tests
    assert len(t) == 8 and set(t["mr"]) == {"MR1", "MR3"}
    # MR3 has no change: d = 0 everywhere, so its p is 1.
    assert (t[t["mr"] == "MR3"]["p_holm"] == 1.0).all()


def test_render_states_the_verdict() -> None:
    r = rows({s: (0.3, 0.01, 0.05) for s in "wxyz"})
    for m in ("rate_MR5", "rate_MR6"):
        r[m] = 0.1
    r["n_train"], r["n_rejected_copies"] = 100, 0
    text = fix_poc.render(r, fix_poc.judge(r), "head")
    assert "**Overall: GO**" in text and "| w |" in text


def test_module_never_touches_test_data() -> None:
    source = inspect.getsource(fix_poc)
    for forbidden in ("load_test", "locked_test", "holdout", '"test"'):
        assert forbidden not in source.replace("2015-16 holdout", "")
