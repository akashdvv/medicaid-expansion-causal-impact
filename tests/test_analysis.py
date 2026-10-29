import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402
import synth as S  # noqa: E402


def test_panel_is_balanced_and_coded():
    d = A.load()
    assert d.shape[0] == 46 * 12 and d.groupby("stfips").size().eq(12).all()
    assert set(d.g.unique()) == {0, 2014, 2015, 2016, 2017, 2019}
    assert d.loc[(d.g == 2014) & (d.year == 2013), "treated_now"].eq(0).all()
    assert d.loc[(d.g == 2014) & (d.year == 2014), "treated_now"].eq(1).all()


def test_two_by_two_matches_manual_arithmetic():
    r = A.two_by_two(A.load())
    assert np.isclose(r["did"], (r["treated_post"] - r["treated_pre"]) - (r["control_post"] - r["control_pre"]))


def test_twfe_recovers_known_effect_on_simulated_panel():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(30):
        g = 2014 if s < 15 else 0
        for y in range(2008, 2020):
            rows.append({"stfips": s, "year": y, "g": g, "W": 1.0,
                         "treated_now": int(g and y >= g),
                         "dins": 0.5 + 0.01 * s + 0.005 * (y - 2008) + 0.06 * int(g and y >= g) + rng.normal(0, 0.002)})
    est = A.twfe(pd.DataFrame(rows))["estimate"]
    assert abs(est - 0.06) < 0.005


def test_synthetic_control_weights_are_a_convex_combination():
    d = A.load()
    donors = sorted(d[d.g == 0].stfips.unique())
    r = S.synth(d, 22, donors, 2017)
    w = r["weights"].values
    assert (w >= -1e-9).all() and np.isclose(w.sum(), 1)
    assert r["rmspe_pre"] < 0.02        # synthetic Louisiana tracks the real one before expansion


def test_synth_recovers_exact_match():
    years = range(2008, 2020)
    donors = {1: np.linspace(0.5, 0.6, 12), 2: np.linspace(0.7, 0.8, 12)}
    treated = 0.5 * donors[1] + 0.5 * donors[2]
    rows = [{"stfips": s, "year": y, "dins": v} for s, vals in {**donors, 9: treated}.items() for y, v in zip(years, vals)]
    r = S.synth(pd.DataFrame(rows), 9, [1, 2], 2017)
    assert np.allclose(r["weights"].values, [0.5, 0.5], atol=1e-3)
