"""
Estimators for the effect of ACA Medicaid expansion on insurance coverage.

Data: share of low-income childless adults with health insurance, by state and year (ACS),
46 states, 2008-2019. Treatment: the year a state's expansion took effect (staggered adoption).

Estimators, from simplest to most robust:
  1. two_by_two          - the textbook DiD: 2014 expanders vs never-expanders, before vs after
  2. twfe                - two-way fixed effects regression (common in older papers; can be biased
                           when adoption is staggered and effects grow over time)
  3. callaway_santanna   - group-time ATTs that only compare treated states to valid controls
  4. honest_sensitivity  - how large a violation of parallel trends would overturn the result
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ehec_medicaid_expansion.csv"


def load() -> pd.DataFrame:
    """Read the panel. If the CSV isn't there yet, rebuild it from the moderndid package."""
    if not DATA.exists():
        import moderndid as md
        DATA.parent.mkdir(exist_ok=True)
        md.load_ehec().to_pandas().to_csv(DATA, index=False)
    d = pd.read_csv(DATA)
    d["g"] = d["yexp2"].fillna(0).astype(int)                 # 0 = not expanded by 2019
    d["treated_now"] = ((d["g"] > 0) & (d["year"] >= d["g"])).astype(int)
    d["cohort"] = d["g"].map(lambda g: "Not expanded by 2019" if g == 0 else f"Expanded {g}")
    return d


def weighted_trends(d: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted insured share per year for 2014 expanders vs never-expanders."""
    sub = d[d["g"].isin([0, 2014])]
    return (sub.groupby(["cohort", "year"])
               .apply(lambda x: np.average(x["dins"], weights=x["W"]), include_groups=False)
               .unstack(0))


def two_by_two(d: pd.DataFrame) -> dict:
    """(treated after - treated before) - (control after - control before), 2014 cohort only."""
    t = weighted_trends(d)
    pre, post = t.loc[2008:2013].mean(), t.loc[2014:2019].mean()
    treat, ctrl = "Expanded 2014", "Not expanded by 2019"
    change_t, change_c = post[treat] - pre[treat], post[ctrl] - pre[ctrl]
    return {"treated_pre": pre[treat], "treated_post": post[treat], "control_pre": pre[ctrl],
            "control_post": post[ctrl], "treated_change": change_t, "control_change": change_c,
            "did": change_t - change_c}


def twfe(d: pd.DataFrame) -> dict:
    """dins ~ treated_now + state FE + year FE, weighted, SEs clustered by state."""
    m = smf.wls("dins ~ treated_now + C(stfips) + C(year)", data=d, weights=d["W"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["stfips"]})
    b, se = m.params["treated_now"], m.bse["treated_now"]
    return {"estimate": b, "se": se, "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se}


def callaway_santanna(d: pd.DataFrame, control_group: str = "nevertreated", base_period: str = "varying"):
    import moderndid as md
    res = md.att_gt(data=d, yname="dins", tname="year", idname="stfips", gname="g", weightsname="W",
                    control_group=control_group, est_method="reg", base_period=base_period, boot=False)
    return res, md.aggte(res, type="simple"), md.aggte(res, type="dynamic")


def agg_summary(agg) -> dict:
    att, se = float(agg.overall_att), float(agg.overall_se)
    return {"estimate": att, "se": se, "ci_low": att - 1.96 * se, "ci_high": att + 1.96 * se}


def event_study_table(dyn) -> pd.DataFrame:
    e = np.asarray(dyn.event_times)
    est, se = np.asarray(dyn.att_by_event), np.asarray(dyn.se_by_event)
    crit = float(dyn.critical_values) if np.ndim(dyn.critical_values) == 0 else np.asarray(dyn.critical_values)
    return pd.DataFrame({"event_time": e, "estimate": est, "se": se,
                         "band_low": est - crit * se, "band_high": est + crit * se})


def event_study(d: pd.DataFrame, window: int = 5, base_period: str = "universal"):
    """Callaway-Sant'Anna event study restricted to +/- `window` years. Far-out leads are
    identified by only a handful of late-expanding states and are mostly noise."""
    import moderndid as md
    res = md.att_gt(data=d, yname="dins", tname="year", idname="stfips", gname="g", weightsname="W",
                    control_group="nevertreated", est_method="reg", base_period=base_period, boot=False)
    return md.aggte(res, type="dynamic", min_e=-window, max_e=window)


def honest_sensitivity(d: pd.DataFrame, event_times=(0, 2, 5), m_bar=(0.0, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Rambachan & Roth (2023) relative-magnitudes bounds: allow post-expansion violations of
    parallel trends up to M-bar x the largest pre-expansion violation; report the robust 95% CI.
    'Breakdown' = the largest M-bar at which the interval still excludes zero."""
    import moderndid as md
    dyn = event_study(d, base_period="universal")
    rows = []
    for e in event_times:
        hd = md.honest_did(dyn, event_time=e, sensitivity_type="relative_magnitude", m_bar_vec=list(m_bar))
        rows.append({"event_time": e, "Mbar": "exact parallel trends", "ci_low": float(hd.original_ci.lb),
                     "ci_high": float(hd.original_ci.ub)})
        ci = hd.robust_ci.to_pandas() if hasattr(hd.robust_ci, "to_pandas") else pd.DataFrame(hd.robust_ci)
        for _, r in ci.iterrows():
            rows.append({"event_time": e, "Mbar": float(r["Mbar"]), "ci_low": float(r["lb"]), "ci_high": float(r["ub"])})
    return pd.DataFrame(rows)


def breakdown_values(sens: pd.DataFrame) -> dict:
    out = {}
    for e, g in sens[sens["Mbar"] != "exact parallel trends"].groupby("event_time"):
        ok = g[g["ci_low"] > 0]["Mbar"].astype(float)
        out[int(e)] = float(ok.max()) if len(ok) else 0.0
    return out
