# Flood Catastrophe Modelling Engine

This repository contains a python-based Flood Catastrophe Modelling Engine that was developed for experimental purposes. It mirrors how typical commerical and open-source CAT models are built, following the Hazard, Vulnerability, Exposure, Financial pipeline. The engine can be run with synthetically generated or user input data. 

A simple modular Python implementation of the standard catastrophe
modelling structure used across insurance/reinsurance:

```
HAZARD  →  VULNERABILITY  →  EXPOSURE  →  FINANCIAL LOSS
```

Given a stochastic event set of flood depths, an exposure portfolio, and
depth-damage vulnerability curves, the engine computes an Event Loss Table,
simulates a Year Loss Table via Monte Carlo, and derives EP curves
and headline risk metrics (AAL, OEP, AEP).


This mirrors how commercial cat models (e.g., RMS, Verisk/AIR, CoreLogic) and
open-source models (e.g., OASIS LMF, CLIMADA) are typically built.

- **Hazard**: physically independent of what's insured. A catalogue of
  possible events, each with a footprint and an annual rate of
  occurrence (not a probability — a Poisson rate).
- **Vulnerability**: physically independent of location. Depth →
  damage-ratio functions per building type.
- **Exposure**: what's actually at risk — where, how much it's worth,
  what type of building it is.
- **Financial**: policy terms (deductible/limit) that convert physical
  damage into money actually paid.

Keeping these four layers strictly separate is what lets you swap in
real hazard data, real exposure data, or real vulnerability curves
independently, without touching the loss calculation logic.

## Project layout

```
engine/
  hazard.py          Stochastic event set (gridded flood depths + rates)
  exposure.py         Exposure portfolio (asset locations, values, occupancy)
  vulnerability.py    Depth-damage curves per occupancy type
  financial.py         Deductible/limit application
  loss_engine.py       Ties it together: Event Loss Table + Monte Carlo Year Loss Table
  metrics.py           AAL, OEP curve, AEP curve, TVaR, return-period losses
  data_sources.py      Helpers/docs for sourcing real public exposure & vulnerability data
examples/
  generate_synthetic_data.py   Builds a complete synthetic input set
  run_example.py                Runs the full pipeline end-to-end, saves plots + tables
data/
  hazard/, exposure/, vulnerability/    Inputs (synthetic, or your own)
  output/                                ELT, YLT, EP curves, metrics, plots
```

## Quick start

```bash
pip install -r requirements.txt
python examples/generate_synthetic_data.py   # builds synthetic hazard/exposure/vulnerability
python examples/run_example.py                # runs the full engine, writes data/output/
```

Outputs written to `data/output/`:
- `event_loss_table.csv` — loss per stochastic event
- `year_loss_table.csv` — 100,000 simulated years of annual loss
- `oep_curve.csv`, `aep_curve.csv` — full exceedance probability curves
- `summary_metrics.csv` — AAL, TVaR, and loss at standard return periods
- `ep_curves.png`, `annual_loss_distribution.png` — plots

## How the engine works

### 1. Hazard: stochastic event set
A `HazardEventSet` is a 3D array of flood depths `(n_events, ny, nx)` over
a regular lat/lon grid, plus a table of `(event_id, annual_rate)`. Each
event's `annual_rate` is a **Poisson rate** — the expected number of
times that specific event occurs per year — not an exceedance
probability. This is the key idea that makes it "stochastic": many
physically plausible events, each with its own frequency, rather than a
handful of fixed "1-in-100-year" scenarios.

The synthetic generator builds ~800 events across 9 magnitude bins
(2yr → 1000yr), with several randomised footprint variants per bin so
that no two "100-year" events look identical — same spirit as a real
model's numerical weather / hydraulic simulation ensemble.

### 2. Vulnerability: depth-damage curves
A `VulnerabilitySet` maps a building's characteristics to a
piecewise-linear depth → damage-ratio curve. Damage ratio (0–1) × asset
value = ground-up loss. Curves optionally carry a coefficient of
variation to sample secondary uncertainty instead of using the
deterministic mean damage.

Curves can be defined at up to four levels of detail:
`occupancy_type` (required) plus optional `construction_type`
(e.g. masonry/wood_frame/concrete/steel_frame), `basement` (0/1), and
`floors_band` (`"1"`, `"2"`, `"3+"`, auto-binned from a raw floor count).
Any attribute can be the wildcard `"any"`. Only an occupancy-level base
curve (`any/any/any`) is required — add more specific curves only where
you actually have better data; `VulnerabilitySet.get()` looks for the
most specific match and falls back progressively (relaxing
construction_type, then basement, then floors_band) down to the base
curve. This means a plain occupancy-only CSV (the original format)
still works unchanged, and you can layer in more granular curves
incrementally.

```csv
occupancy_type,construction_type,basement,floors_band,depth_m,damage_ratio,cv
residential,any,any,any,0.0,0.00,0.25          # required base curve
residential,wood_frame,any,any,0.0,0.00,0.25    # construction-specific
residential,any,1,any,0.0,0.05,0.25             # basement-specific
residential,wood_frame,1,1,0.0,0.00,0.28        # fully-specified combo
```

On the exposure side, an asset just carries plain columns
(`construction_type`, `basement`, `floors`); the engine normalizes and
bins them (`vulnerability.normalize_basement`, `vulnerability.floor_band`)
before doing the lookup, and assets are grouped by their resolved curve
so the curve lookup happens once per distinct combination rather than
once per asset.

### 3. Exposure: the portfolio at risk
An `ExposurePortfolio` is a table of assets: location, occupancy type,
value, and (optionally) deductible/limit. `depths_at_points()` on the
hazard set does a fast nearest-cell lookup to get every event's flood
depth at every asset location in one vectorised operation.

### 4. Financial: policy terms
`apply_policy_terms()` converts ground-up loss into net (insured) loss
per asset per event: `net = min(max(ground_up - deductible, 0), limit)`.

### 5. Loss engine: ELT → YLT
`LossEngine.compute_event_loss_table()` produces one row per event with
gross and net portfolio loss. `simulate_year_loss_table()` then runs an
efficient Monte Carlo simulation (Poisson-thinning, not a
event-by-event/year-by-year loop) to produce `n_years` of simulated
annual experience — the occurrence loss (worst single event) and
aggregate loss (sum of all events) for each simulated year.

### 6. Metrics: AAL, OEP, AEP
- **AAL** — mean of the simulated annual loss distribution.
- **OEP** — P(worst single event loss in a year > L), for per-occurrence
  risk (e.g. cat XL reinsurance).
- **AEP** — P(total annual loss > L), for aggregate covers / overall
  solvency.
- `loss_at_return_period()` interpolates either curve (log-linear in
  exceedance probability) to give the standard "loss at the 1-in-100"
  style figures.

## Bringing your own data

### Your own hazard, exposure, or vulnerability
- Hazard: build a `HazardEventSet` from your own gridded depths (e.g.
  from JBA, Fathom, JRC, or a national flood model) — construct a
  `HazardGrid` + depths array + event table with rates, or extend
  `HazardEventSet` with a `from_geotiff_folder` / `from_netcdf` loader
  for your source format.
- Exposure: `ExposurePortfolio.from_csv()` — any CSV with
  `asset_id, lon, lat, occupancy_type, value` (+ optional
  `deductible, limit`).
- Vulnerability: `VulnerabilitySet.from_csv()` — long-format CSV with
  `occupancy_type, depth_m, damage_ratio` (+ optional `cv`,
  `construction_type`, `basement`, `floors_band` — see above).

### Public datasets
`engine/data_sources.py` documents and provides ready-to-run helpers for:
- **Overture Maps** building footprints (global, open, monthly
  GeoParquet releases, queryable directly via DuckDB by bounding box) —
  `fetch_overture_buildings()`.
- **Microsoft Global ML Building Footprints** (global, satellite-derived
  footprints, per-country downloads) — `microsoft_building_footprints_info()`
  gives the download + processing steps.
- **JRC Global Flood Depth-Damage Functions** (Huizinga, de Moel &
  Szewczyk, 2017 — open, continent/occupancy-level depth-damage curves
  and max-damage-per-m² by country) — see `JRC_VULNERABILITY_INFO`.

These require outbound internet access to the respective providers and
are best run in your own environment; each function/constant documents
exactly what to run and how to feed the result into
`ExposurePortfolio.from_footprints()` or `VulnerabilitySet.from_csv()`.

## Extending the engine
- **Correlation/clustering**: the Monte Carlo step currently treats
  events as independent Poisson processes; add clustering (e.g. multiple
  events per storm system) by grouping event draws.
- **Reinsurance structures**: `financial.py` currently applies simple
  per-asset deductible/limit; add per-occurrence and aggregate
  reinsurance layers on top of the YLT.
- **Real hazard grids**: swap the synthetic `.npz` loader for a
  GeoTIFF/NetCDF loader if you have rasterio/xarray available.
- **Climate change conditioning**: scale event depths or annual rates by
  scenario before running `compute_event_loss_table()`.
