"""
Synthetic control for a single late-expanding state (Abadie, Diamond & Hainmueller).

Louisiana expanded in mid-2016 (first full year 2017). We build a "synthetic Louisiana" as a
weighted average of never-expanding states that tracks Louisiana's insured share before
expansion; the post-2017 gap between real and synthetic Louisiana is the estimated effect.
Inference: run the same procedure pretending each donor state was treated (placebos) and ask
how unusual Louisiana's post/pre error ratio is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_weights(y_treated_pre: np.ndarray, Y_donors_pre: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise pre-period squared error."""
    k = Y_donors_pre.shape[1]
    obj = lambda w: np.sum((y_treated_pre - Y_donors_pre @ w) ** 2)
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def synth(panel: pd.DataFrame, treated: int, donors: list[int], first_treated_year: int) -> dict:
    wide = panel.pivot(index="year", columns="stfips", values="dins")
    pre = wide.index < first_treated_year
    w = fit_weights(wide.loc[pre, treated].values, wide.loc[pre, donors].values)
    synthetic = wide[donors].values @ w
    gap = wide[treated].values - synthetic
    rmspe_pre = np.sqrt(np.mean(gap[pre] ** 2))
    rmspe_post = np.sqrt(np.mean(gap[~pre] ** 2))
    return {"years": wide.index.values, "actual": wide[treated].values, "synthetic": synthetic, "gap": gap,
            "weights": pd.Series(w, index=donors), "rmspe_pre": rmspe_pre, "rmspe_post": rmspe_post,
            "ratio": rmspe_post / rmspe_pre, "avg_post_gap": gap[~pre].mean()}


def placebo_test(panel: pd.DataFrame, treated: int, donors: list[int], first_treated_year: int) -> pd.DataFrame:
    rows = []
    for unit in [treated] + donors:
        pool = [d for d in donors if d != unit]
        r = synth(panel, unit, pool, first_treated_year)
        rows.append({"stfips": unit, "is_treated": unit == treated, "ratio": r["ratio"],
                     "avg_post_gap": r["avg_post_gap"], "rmspe_pre": r["rmspe_pre"], "gap": r["gap"]})
    out = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
