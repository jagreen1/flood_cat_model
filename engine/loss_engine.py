"""
loss_engine.py
==============
Ties Hazard -> Vulnerability -> Exposure -> Financial Loss together and
produces the two core outputs of a catastrophe model:

    1. Event Loss Table (ELT)  - expected loss per stochastic event
    2. Year Loss Table  (YLT)  - simulated annual losses via Monte Carlo,
                                  which is what EP curves and AAL are
                                  actually built from.

Why Monte Carlo for the YLT?
-----------------------------
A stochastic event set only tells you *what could happen* and *how often*
(each event's annual_rate is a Poisson rate). To get a distribution of
*annual* losses (needed for OEP/AEP), the standard actuarial approach is
to simulate a large number of synthetic years: for each simulated year,
randomly determine which events occur (a Poisson process per event) and
sum/max their losses. With enough simulated years this converges to the
exact analytical annual loss distribution implied by the event set, while
remaining simple to reason about and extend (correlation, clustering,
multi-peril, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hazard import HazardEventSet
from .exposure import ExposurePortfolio
from .vulnerability import VulnerabilitySet, ANY, normalize_basement, floor_band
from .financial import apply_policy_terms


class LossEngine:
    def __init__(self, hazard: HazardEventSet, exposure: ExposurePortfolio,
                 vulnerability: VulnerabilitySet, seed: int = 42):
        self.hazard = hazard
        self.exposure = exposure
        self.vulnerability = vulnerability
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Step 1: Event Loss Table
    # ------------------------------------------------------------------ #
    def compute_event_loss_table(self, stochastic_damage: bool = False) -> pd.DataFrame:
        """
        Compute ground-up and net (insured) loss for every event in the
        hazard set, aggregated across the whole exposure portfolio.

        Each asset is matched to a vulnerability curve using its
        occupancy_type plus, where available, construction_type,
        basement, and floors - via VulnerabilitySet's hierarchical
        fallback (see vulnerability.py), so assets don't all need every
        attribute populated.

        Returns a DataFrame: event_id, annual_rate, gross_loss, net_loss
        """
        exp = self.exposure.df
        depths = self.hazard.depths_at_points(exp["lon"].values, exp["lat"].values)
        # depths shape: (n_events, n_assets)

        # Normalize the vulnerability-relevant attributes once so assets
        # that resolve to the same curve are grouped together (fast) and
        # curve lookups happen once per distinct combination, not once
        # per asset.
        group_keys = pd.DataFrame({
            "occupancy_type": exp["occupancy_type"],
            "construction_type": exp["construction_type"].apply(
                lambda v: v.strip().lower() if isinstance(v, str) and v.strip() else ANY
            ),
            "basement": exp["basement"].apply(normalize_basement),
            "floors_band": exp["floors"].apply(floor_band),
        })

        damage_ratio = np.zeros_like(depths, dtype=np.float64)
        for key_vals, sub in group_keys.groupby(list(group_keys.columns)):
            occ, constr, basement, floors_band = key_vals
            curve = self.vulnerability.get(
                occ, construction_type=constr, basement=basement, floors=floors_band
            )
            idx = sub.index.values
            if stochastic_damage:
                damage_ratio[:, idx] = curve.sample_damage_ratio(depths[:, idx], self.rng)
            else:
                damage_ratio[:, idx] = curve.damage_ratio(depths[:, idx])

        ground_up = damage_ratio * exp["value"].values[None, :]
        net = apply_policy_terms(ground_up, exp["deductible"].values, exp["limit"].values)

        self._per_asset_gross = ground_up  # cached for diagnostics / reuse
        self._per_asset_net = net

        elt = self.hazard.event_table[["event_id", "annual_rate"]].copy()
        elt["gross_loss"] = ground_up.sum(axis=1)
        elt["net_loss"] = net.sum(axis=1)
        self.elt = elt
        return elt

    # ------------------------------------------------------------------ #
    # Step 2: Year Loss Table via Monte Carlo
    # ------------------------------------------------------------------ #
    def simulate_year_loss_table(self, n_years: int = 100_000, loss_col: str = "net_loss",
                                  seed: int | None = None) -> pd.DataFrame:
        """
        Simulate `n_years` of synthetic annual experience from the ELT.

        Method (efficient Poisson-thinning simulation):
        - total_rate = sum of all event annual_rates
        - number of event occurrences across the whole n_years horizon is
          Poisson(total_rate * n_years)
        - each occurrence is independently assigned to a simulated year
          (uniformly, since a Poisson process has occurrence times
          uniformly distributed given the count) and to an event id (drawn
          proportional to that event's rate)
        - this reproduces, exactly, "for every event, occurs Poisson(rate)
          times per year" without looping over events x years.

        Returns a DataFrame: year, occurrence_loss (max single event loss
        that year), aggregate_loss (sum of all event losses that year),
        n_events (number of triggering events that year).
        """
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        elt = self.elt
        rates = elt["annual_rate"].values
        losses = elt[loss_col].values
        total_rate = rates.sum()

        expected_occurrences = total_rate * n_years
        n_occurrences = rng.poisson(expected_occurrences)

        if n_occurrences == 0:
            years = np.arange(1, n_years + 1)
            return pd.DataFrame({
                "year": years, "occurrence_loss": 0.0, "aggregate_loss": 0.0, "n_events": 0
            })

        # assign each occurrence to a simulated year (uniform over horizon)
        occurrence_years = rng.integers(1, n_years + 1, size=n_occurrences)
        # assign each occurrence to an event, weighted by its rate
        event_probs = rates / total_rate
        event_idx = rng.choice(len(elt), size=n_occurrences, p=event_probs)
        occurrence_losses = losses[event_idx]

        occ_df = pd.DataFrame({
            "year": occurrence_years,
            "event_loss": occurrence_losses,
        })

        grouped = occ_df.groupby("year")["event_loss"]
        year_summary = grouped.agg(
            occurrence_loss="max",
            aggregate_loss="sum",
            n_events="count",
        )

        all_years = pd.DataFrame({"year": np.arange(1, n_years + 1)}).set_index("year")
        ylt = all_years.join(year_summary).fillna(
            {"occurrence_loss": 0.0, "aggregate_loss": 0.0, "n_events": 0}
        ).reset_index()
        ylt["n_events"] = ylt["n_events"].astype(int)
        self.ylt = ylt
        return ylt
