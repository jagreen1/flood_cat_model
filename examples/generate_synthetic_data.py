"""
generate_synthetic_data.py
===========================
Builds a complete, self-consistent synthetic input set for the flood cat
model so the whole engine can be run end-to-end with no external data:

    1. A stochastic hazard event set: 800 synthetic flood footprints over
       a ~50km x 50km coastal river-plain study area, with annual rates
       calibrated so the area experiences a physically sensible
       frequency-severity relationship (frequent shallow floods, rare
       deep floods).
    2. An exposure portfolio: ~2,000 synthetic buildings scattered over
       the same area (denser near the river), each with an occupancy
       type and replacement value.
    3. A vulnerability set: depth-damage curves for residential,
       commercial and industrial occupancies, styled on the shape of
       published curves (e.g. JRC / Huizinga et al. 2017) without
       reproducing their actual data points.

Run directly: `python generate_synthetic_data.py`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"

RNG = np.random.default_rng(7)

# Study area: a synthetic river plain, roughly 0.45deg x 0.45deg (~50km)
LON_MIN, LAT_MIN = -1.20, 51.20
CELL_SIZE = 0.01          # ~1.1km cells
NX, NY = 45, 45           # grid dimensions
RIVER_LON = LON_MIN + 0.5 * NX * CELL_SIZE  # river runs down the middle


# --------------------------------------------------------------------- #
# 1. Hazard: stochastic flood event set
# --------------------------------------------------------------------- #
def generate_hazard_event_set(n_events: int = 800):
    lons = LON_MIN + (np.arange(NX) + 0.5) * CELL_SIZE
    lats = LAT_MIN + (np.arange(NY) + 0.5) * CELL_SIZE
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # shape (NY, NX)

    # distance from the (wiggly) river centreline, used to shape footprints
    river_wiggle = 0.03 * np.sin((lat_grid - LAT_MIN) * 25)
    dist_from_river = np.abs(lon_grid - (RIVER_LON + river_wiggle))

    depths = np.zeros((n_events, NY, NX), dtype=np.float32)
    records = []

    # Design the frequency-severity relationship: a handful of "magnitude
    # bins" (like a return-period ladder), each populated with several
    # stochastic footprint variants (different peak depth, extent, and
    # location jitter) so within a bin no two events are identical -
    # this is what makes it a genuine *stochastic* event set rather than
    # a lookup table of historical return periods.
    magnitude_bins = [
        # (label, central_RP_years, peak_depth_m, extent_km, n_variants)
        ("nuisance", 2, 0.3, 4, 140),
        ("minor", 5, 0.6, 6, 130),
        ("moderate", 10, 1.0, 9, 120),
        ("significant", 25, 1.6, 13, 110),
        ("major", 50, 2.3, 17, 100),
        ("severe", 100, 3.2, 22, 80),
        ("extreme", 250, 4.3, 28, 70),
        ("catastrophic", 500, 5.5, 34, 30),
        ("worst_case", 1000, 7.0, 40, 20),
    ]

    event_id = 0
    for label, central_rp, peak_depth, extent_km, n_variants in magnitude_bins:
        # convert extent in km to degrees (roughly, at this latitude)
        extent_deg = extent_km / 111.0
        # total rate for this bin = 1/RP; split across variants so that
        # summing the variants' rates reproduces the bin's frequency
        bin_rate = 1.0 / central_rp
        for _ in range(n_variants):
            # jitter the footprint centre along the river and its depth/extent
            centre_lat = RNG.uniform(LAT_MIN + 0.05, LAT_MIN + NY * CELL_SIZE - 0.05)
            centre_lon = RIVER_LON + 0.03 * np.sin((centre_lat - LAT_MIN) * 25)
            variant_peak = peak_depth * RNG.uniform(0.75, 1.25)
            variant_extent = extent_deg * RNG.uniform(0.7, 1.3)

            d_lon = lon_grid - centre_lon
            d_lat = (lat_grid - centre_lat) / 2.2  # elongate along the river
            radial = np.sqrt(d_lon ** 2 + d_lat ** 2)

            footprint = variant_peak * np.exp(-(radial ** 2) / (2 * variant_extent ** 2))
            # depth also decays away from the river channel itself
            river_decay = np.exp(-(dist_from_river ** 2) / (2 * (variant_extent * 0.6) ** 2))
            footprint = footprint * (0.4 + 0.6 * river_decay)
            footprint[footprint < 0.02] = 0.0  # dry threshold

            depths[event_id] = footprint
            records.append({
                "event_id": f"EV{event_id:05d}",
                "magnitude_bin": label,
                "annual_rate": bin_rate / n_variants,
                "central_return_period": central_rp,
            })
            event_id += 1

    event_table = pd.DataFrame(records)
    grid_meta = dict(lon_min=LON_MIN, lat_min=LAT_MIN, cell_size=CELL_SIZE, nx=NX, ny=NY)
    return depths, event_table, grid_meta


# --------------------------------------------------------------------- #
# 2. Exposure: synthetic building portfolio
# --------------------------------------------------------------------- #
def generate_exposure_portfolio(n_assets: int = 2000):
    # Denser development near the river, thinning out with distance
    lats = RNG.uniform(LAT_MIN + 0.02, LAT_MIN + NY * CELL_SIZE - 0.02, n_assets)
    river_wiggle = 0.03 * np.sin((lats - LAT_MIN) * 25)
    river_centre = RIVER_LON + river_wiggle
    offset = RNG.normal(0, 0.10, n_assets)  # most development within ~10km of river
    lons = river_centre + offset
    # clip to study area
    lons = np.clip(lons, LON_MIN + 0.01, LON_MIN + NX * CELL_SIZE - 0.01)

    occupancy = RNG.choice(
        ["residential", "commercial", "industrial"],
        size=n_assets, p=[0.72, 0.20, 0.08],
    )
    base_value = {
        "residential": 250_000,
        "commercial": 900_000,
        "industrial": 1_500_000,
    }
    value = np.array([base_value[o] for o in occupancy]) * RNG.lognormal(0, 0.35, n_assets)

    # Construction type, basement presence, and floor count - the
    # additional attributes the vulnerability curves can now be
    # differentiated on. Distributions are loosely realistic: wood_frame
    # dominates residential, concrete/steel dominate industrial;
    # basements are more common inland (further from the river, i.e.
    # cheaper/older ground) is not modelled here for simplicity - kept
    # independent of location.
    construction_probs = {
        "residential": [0.30, 0.45, 0.15, 0.10],   # masonry, wood_frame, concrete, steel_frame
        "commercial": [0.25, 0.10, 0.35, 0.30],
        "industrial": [0.10, 0.05, 0.35, 0.50],
    }
    construction_types_list = ["masonry", "wood_frame", "concrete", "steel_frame"]
    construction_type = np.array([
        RNG.choice(construction_types_list, p=construction_probs[o]) for o in occupancy
    ])

    basement = RNG.choice([0, 1], size=n_assets, p=[0.55, 0.45])

    floor_probs = {
        "residential": [0.55, 0.35, 0.08, 0.02],   # 1, 2, 3, 4 floors
        "commercial": [0.35, 0.30, 0.20, 0.15],
        "industrial": [0.60, 0.20, 0.12, 0.08],
    }
    floors = np.array([
        RNG.choice([1, 2, 3, 4], p=floor_probs[o]) for o in occupancy
    ])

    df = pd.DataFrame({
        "asset_id": [f"A{i:05d}" for i in range(n_assets)],
        "lon": lons,
        "lat": lats,
        "occupancy_type": occupancy,
        "value": value.round(0),
        "deductible": (value * 0.01).round(0),   # 1% deductible
        "limit": value,                           # full value insured limit
        "construction_type": construction_type,
        "basement": basement,
        "floors": floors,
    })
    return df


# --------------------------------------------------------------------- #
# 3. Vulnerability: depth-damage curves by occupancy
# --------------------------------------------------------------------- #
def generate_vulnerability_curves():
    """
    Depth-damage curve shapes loosely styled on the published literature
    (steep initial damage for residential single-storey stock, more
    gradual for multi-storey commercial/industrial with elevated
    equipment) - illustrative only, not a reproduction of any specific
    published dataset. Replace with real JRC/Hazus curves for production
    use (see engine/data_sources.py).

    Demonstrates the curve hierarchy end to end:
      1. A base curve per occupancy_type (construction/basement/floors
         = "any") - the required fallback for every occupancy.
      2. Construction-type-specific curves (masonry omitted - it's
         treated as representative of the base curve).
      3. A basement=1 curve per occupancy - basements sustain higher
         damage even at shallow depths since they flood essentially as
         soon as any water reaches the building.
      4. Floor-count curves (2, 3+) per occupancy - more floors means
         more of the building's value sits above the flood, so the same
         depth destroys a smaller fraction of total value.
      5. A couple of fully-specified combination curves, to show exact
         multi-attribute matches taking priority over any fallback.
    Everything else (e.g. a masonry building with a basement) resolves
    through VulnerabilitySet's hierarchical fallback - see vulnerability.py.
    """
    depths = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

    base_curves = {
        "residential": [0.00, 0.08, 0.18, 0.28, 0.45, 0.58, 0.68, 0.82, 0.90, 0.95, 0.97],
        "commercial": [0.00, 0.04, 0.10, 0.17, 0.32, 0.44, 0.55, 0.70, 0.80, 0.87, 0.91],
        "industrial": [0.00, 0.03, 0.07, 0.12, 0.24, 0.34, 0.44, 0.60, 0.71, 0.79, 0.85],
    }
    cv = {"residential": 0.25, "commercial": 0.30, "industrial": 0.35}

    rows = []

    def add_curve(occ, ratios, construction_type="any", basement="any", floors_band="any",
                  cv_override=None):
        cv_val = cv[occ] if cv_override is None else cv_override
        for d, r in zip(depths, ratios):
            rows.append({
                "occupancy_type": occ,
                "construction_type": construction_type,
                "basement": basement,
                "floors_band": floors_band,
                "depth_m": d,
                "damage_ratio": round(min(max(r, 0.0), 1.0), 4),
                "cv": cv_val,
            })

    # 1. Base curves - required fallback for every occupancy
    for occ, ratios in base_curves.items():
        add_curve(occ, ratios)

    # 2. Construction-type modifiers (any basement, any floors).
    #    masonry ~= base curve, so no explicit masonry row is needed.
    construction_factor = {"concrete": 0.85, "steel_frame": 0.80, "wood_frame": 1.15}
    for occ, ratios in base_curves.items():
        for constr, factor in construction_factor.items():
            modified = [r * factor for r in ratios]
            add_curve(occ, modified, construction_type=constr)

    # 3. Basement=1 modifier (any construction, any floors): basements
    #    flood almost fully at shallow depth, so damage ratio is inflated
    #    and doesn't start from zero.
    basement_factor = 1.2
    for occ, ratios in base_curves.items():
        modified = [max(r * basement_factor, 0.10 if d > 0 else 0.0) for d, r in zip(depths, ratios)]
        add_curve(occ, modified, basement="1")

    # 4. Floor-count modifiers (any construction, any basement): value is
    #    spread over more floors, so a given depth damages a smaller share.
    floor_factor = {"2": 0.75, "3+": 0.55}
    for occ, ratios in base_curves.items():
        for band, factor in floor_factor.items():
            modified = [r * factor for r in ratios]
            add_curve(occ, modified, floors_band=band)

    # 5. Fully-specified combination curves - exact match beats any fallback
    add_curve(
        "residential",
        [r * 1.15 * 1.2 for r in base_curves["residential"]],
        construction_type="wood_frame", basement="1", floors_band="1",
        cv_override=0.28,
    )
    add_curve(
        "commercial",
        [r * 0.85 * 0.55 for r in base_curves["commercial"]],
        construction_type="concrete", basement="0", floors_band="3+",
        cv_override=0.20,
    )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------- #
def main():
    hazard_dir = DATA_DIR / "hazard"
    exposure_dir = DATA_DIR / "exposure"
    vuln_dir = DATA_DIR / "vulnerability"
    for d in (hazard_dir, exposure_dir, vuln_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic hazard event set...")
    depths, event_table, grid_meta = generate_hazard_event_set()
    np.savez_compressed(
        hazard_dir / "synthetic_events.npz",
        depths=depths,
        event_table=event_table.to_dict(orient="list"),
        **grid_meta,
    )
    event_table.to_csv(hazard_dir / "synthetic_event_table.csv", index=False)
    print(f"  {len(event_table)} events, grid {NY}x{NX}, "
          f"total annual rate={event_table['annual_rate'].sum():.3f}")

    print("Generating synthetic exposure portfolio...")
    exposure_df = generate_exposure_portfolio()
    exposure_df.to_csv(exposure_dir / "synthetic_exposure.csv", index=False)
    print(f"  {len(exposure_df)} assets, TIV={exposure_df['value'].sum():,.0f}")

    print("Generating synthetic vulnerability curves...")
    vuln_df = generate_vulnerability_curves()
    vuln_df.to_csv(vuln_dir / "synthetic_vulnerability.csv", index=False)
    print(f"  {vuln_df['occupancy_type'].nunique()} occupancy curves")

    print("\nDone. Files written to:", DATA_DIR)


if __name__ == "__main__":
    main()
