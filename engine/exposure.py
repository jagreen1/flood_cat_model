"""
exposure.py
===========
The EXPOSURE layer of the cat model: "what is at risk, where, and how
much is it worth".

An ExposurePortfolio is just a table of assets (usually buildings) with:
    asset_id, lon, lat, occupancy_type, value, (optional) deductible, limit

`occupancy_type` must match the `occupancy_type` values used in the
VulnerabilitySet so each asset can be mapped to the correct depth-damage
curve (e.g. "residential", "commercial", "industrial").

Two ways to populate a portfolio:
1. Bring your own CSV (`ExposurePortfolio.from_csv`).
2. Source real building footprints from public datasets - see
   `data_sources.py` for documented, ready-to-run queries against
   Overture Maps and Microsoft Global ML Building Footprints. Those
   functions return a GeoDataFrame of footprints; use
   `ExposurePortfolio.from_footprints` to turn that into a valued
   exposure portfolio by attaching occupancy types and $/m2 replacement
   costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["asset_id", "lon", "lat", "occupancy_type", "value"]


class ExposurePortfolio:
    def __init__(self, df: pd.DataFrame):
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"exposure table missing required columns: {missing}")
        df = df.copy()
        if "deductible" not in df.columns:
            df["deductible"] = 0.0
        if "limit" not in df.columns:
            df["limit"] = np.inf
        # Optional vulnerability-differentiating attributes. Left unset
        # (None), an asset simply falls back to the occupancy-level base
        # vulnerability curve - see VulnerabilitySet.get() for the
        # fallback hierarchy.
        if "construction_type" not in df.columns:
            df["construction_type"] = None
        if "basement" not in df.columns:
            df["basement"] = None
        if "floors" not in df.columns:
            df["floors"] = None
        self.df = df.reset_index(drop=True)

    @classmethod
    def from_csv(cls, path: str) -> "ExposurePortfolio":
        return cls(pd.read_csv(path))

    @classmethod
    def from_footprints(
        cls,
        footprints: pd.DataFrame,
        occupancy_map: dict | None = None,
        default_occupancy: str = "residential",
        cost_per_m2: dict | None = None,
        default_cost_per_m2: float = 1500.0,
    ) -> "ExposurePortfolio":
        """
        Convert raw building footprints (e.g. from Overture Maps or
        Microsoft Building Footprints, after computing centroid lon/lat and
        footprint area) into a valued exposure portfolio.

        Parameters
        ----------
        footprints : DataFrame with columns [id, lon, lat, area_m2, class]
            `class` is the source dataset's building class/subtype, used to
            infer occupancy via `occupancy_map`.
        occupancy_map : dict mapping source class -> occupancy_type
        cost_per_m2 : dict mapping occupancy_type -> replacement cost / m2
        """
        occupancy_map = occupancy_map or {}
        cost_per_m2 = cost_per_m2 or {}

        occ = footprints.get("class", pd.Series(dtype=object)).map(occupancy_map)
        occ = occ.fillna(default_occupancy)

        cost = occ.map(cost_per_m2).fillna(default_cost_per_m2)
        area = footprints.get("area_m2", pd.Series(60.0, index=footprints.index))
        value = area * cost

        df = pd.DataFrame({
            "asset_id": footprints["id"].astype(str),
            "lon": footprints["lon"],
            "lat": footprints["lat"],
            "occupancy_type": occ,
            "value": value,
        })
        return cls(df)

    def __len__(self):
        return len(self.df)

    def total_insured_value(self) -> float:
        return float(self.df["value"].sum())

    def __repr__(self):
        tiv = self.total_insured_value()
        occ_counts = self.df["occupancy_type"].value_counts().to_dict()
        return f"<ExposurePortfolio: {len(self)} assets, TIV={tiv:,.0f}, occupancy={occ_counts}>"
