"""
metrics.py
==========
Standard catastrophe-model risk metrics computed from a Year Loss Table
(YLT), as produced by `loss_engine.LossEngine.simulate_year_loss_table`.

Definitions
-----------
AAL (Average Annual Loss)
    The mean of the annual loss distribution. = expected value of loss in
    any given year. Also called "pure premium" / "technical premium" base.

OEP (Occurrence Exceedance Probability) curve
    For a given loss threshold L, OEP(L) = P(the single largest loss event
    in a year exceeds L). Answers: "what's the chance my worst single
    event this year is bigger than L?" Used for per-occurrence
    reinsurance / capital purposes.

AEP (Aggregate Exceedance Probability) curve
    For a given loss threshold L, AEP(L) = P(the sum of all event losses
    in a year exceeds L). Answers: "what's the chance my total losses
    this year exceed L?" Used for aggregate covers / overall solvency.

Both curves are estimated empirically from the simulated YLT (the
standard Monte Carlo approach) and are usually reported either as:
    - exceedance probability at a given loss level, or
    - loss at a given return period (RP = 1 / exceedance probability)

Return period loss ("loss at the 1-in-100") is the inverse of exceedance
probability, and is what most audiences actually want to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_annual_loss(ylt: pd.DataFrame, loss_col: str = "aggregate_loss") -> float:
    """AAL = mean annual loss across all simulated years (unconditional
    mean, i.e. years with zero loss are included)."""
    return float(ylt[loss_col].mean())


def _ep_curve(losses: np.ndarray, n_years: int) -> pd.DataFrame:
    """
    Shared empirical exceedance-probability calculation.
    Given an array of annual losses (one value per simulated year, zeros
    included), return a DataFrame of unique loss thresholds with their
    exceedance probability and implied return period.
    """
    sorted_losses = np.sort(losses)[::-1]  # descending: largest loss first
    ranks = np.arange(1, len(sorted_losses) + 1)
    exceedance_prob = ranks / n_years  # P(annual loss >= this value)
    df = pd.DataFrame({"loss": sorted_losses, "exceedance_prob": exceedance_prob})
    # keep only the smallest rank (smallest EP, i.e. rarest occurrence) per unique loss value
    df = df.groupby("loss", as_index=False)["exceedance_prob"].min()
    # Sort ascending by loss. This makes exceedance_prob monotonically
    # DEcreasing as we go down the table (bigger loss -> rarer -> lower EP),
    # which is the convention `loss_at_return_period` relies on.
    df = df.sort_values("loss", ascending=True).reset_index(drop=True)
    df["return_period_years"] = 1.0 / df["exceedance_prob"]
    return df


def occurrence_ep_curve(ylt: pd.DataFrame) -> pd.DataFrame:
    """OEP curve from the YLT's per-year max single-event loss."""
    n_years = len(ylt)
    return _ep_curve(ylt["occurrence_loss"].values, n_years)


def aggregate_ep_curve(ylt: pd.DataFrame) -> pd.DataFrame:
    """AEP curve from the YLT's per-year summed loss."""
    n_years = len(ylt)
    return _ep_curve(ylt["aggregate_loss"].values, n_years)


def loss_at_return_period(ep_curve: pd.DataFrame, return_period_years: float) -> float:
    """
    Interpolate the EP curve to estimate the loss at a specific return
    period (e.g. 100, 200, 500-year loss). Uses log-linear interpolation
    on exceedance probability, which behaves better than linear
    interpolation across the long tail of a loss distribution.
    """
    target_ep = 1.0 / return_period_years
    ep = ep_curve["exceedance_prob"].values
    loss = ep_curve["loss"].values

    if target_ep >= ep[0]:
        return float(loss[0])
    if target_ep <= ep[-1]:
        return float(loss[-1])

    # ep is sorted descending; find bracketing points and interpolate in
    # log(ep) space (loss vs log-return-period is much closer to linear)
    log_ep = np.log(ep)
    idx = np.searchsorted(-log_ep, -np.log(target_ep))
    x0, x1 = log_ep[idx - 1], log_ep[idx]
    y0, y1 = loss[idx - 1], loss[idx]
    if x1 == x0:
        return float(y0)
    frac = (np.log(target_ep) - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def tail_value_at_risk(ylt: pd.DataFrame, loss_col: str = "aggregate_loss",
                        alpha: float = 0.99) -> float:
    """TVaR / CTE at level alpha: mean loss in the worst (1-alpha) fraction
    of simulated years. E.g. alpha=0.99 -> average of worst 1% of years."""
    losses = np.sort(ylt[loss_col].values)
    cutoff_idx = int(np.ceil(alpha * len(losses)))
    tail = losses[cutoff_idx:]
    if len(tail) == 0:
        tail = losses[-1:]
    return float(tail.mean())


def summary_metrics(ylt: pd.DataFrame, return_periods=(10, 50, 100, 200, 500, 1000)) -> dict:
    """Convenience wrapper: bundles AAL, standard return-period losses
    (OEP & AEP) and tail metrics into one dict for reporting."""
    oep = occurrence_ep_curve(ylt)
    aep = aggregate_ep_curve(ylt)
    out = {
        "AAL": average_annual_loss(ylt),
        "TVaR_99_aggregate": tail_value_at_risk(ylt, "aggregate_loss", 0.99),
        "TVaR_99_occurrence": tail_value_at_risk(ylt, "occurrence_loss", 0.99),
    }
    for rp in return_periods:
        out[f"OEP_{rp}yr"] = loss_at_return_period(oep, rp)
        out[f"AEP_{rp}yr"] = loss_at_return_period(aep, rp)
    return out
