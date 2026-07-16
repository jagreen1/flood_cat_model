"""
hazard.py
=========
The HAZARD layer of the cat model.

Represents a *stochastic event set*: a catalogue of possible flood events,
each with:
    - a unique event_id
    - an annual occurrence rate (expected number of times per year this
      exact event happens - a Poisson rate, NOT a probability)
    - a gridded flood depth footprint (metres of water, 0 = dry)

This is the standard way catastrophe models represent hazard: instead of
"the 1-in-100-year flood", you have thousands of physically plausible
events, each with its own footprint and its own rate of occurrence. Annual
probabilities (OEP/AEP) fall out of the *combination* of many such events,
not from any single event.

Depths are stored as a single 3D numpy array (n_events, ny, nx) over a
regular lat/lon grid, which keeps the engine dependency-light (no GDAL /
rasterio requirement) while still behaving like a real gridded hazard
layer. Swap `HazardEventSet.from_netcdf` / `.from_geotiff_folder` in if you
have real modelled hazard (e.g. JBA, Fathom, JRC, national EA/FEMA grids).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HazardGrid:
    """Regular lat/lon grid definition shared by every event footprint."""
    lon_min: float
    lat_min: float
    cell_size: float  # degrees
    nx: int
    ny: int

    @property
    def lons(self) -> np.ndarray:
        return self.lon_min + (np.arange(self.nx) + 0.5) * self.cell_size

    @property
    def lats(self) -> np.ndarray:
        return self.lat_min + (np.arange(self.ny) + 0.5) * self.cell_size

    def cell_index(self, lon: np.ndarray, lat: np.ndarray):
        """Nearest-cell lookup: vectorised, returns (row, col) index arrays.
        Points outside the grid are flagged with -1."""
        col = np.floor((lon - self.lon_min) / self.cell_size).astype(int)
        row = np.floor((lat - self.lat_min) / self.cell_size).astype(int)
        outside = (col < 0) | (col >= self.nx) | (row < 0) | (row >= self.ny)
        col = np.where(outside, -1, col)
        row = np.where(outside, -1, row)
        return row, col


class HazardEventSet:
    """
    A stochastic event set of flood depths.

    Parameters
    ----------
    grid : HazardGrid
        Spatial grid shared by all events.
    depths : np.ndarray, shape (n_events, ny, nx)
        Flood depth in metres for every event / cell. 0 = no flooding.
    event_table : pd.DataFrame
        Must contain columns: event_id, annual_rate [, return_period, description]
        `annual_rate` is a Poisson rate (events / year), NOT an exceedance
        probability. Independent events can share overlapping return
        periods; that's expected in a real stochastic set.
    """

    def __init__(self, grid: HazardGrid, depths: np.ndarray, event_table: pd.DataFrame):
        assert depths.shape[0] == len(event_table), (
            "depths first axis must match number of events in event_table"
        )
        assert depths.shape[1:] == (grid.ny, grid.nx), "depth grid shape mismatch"
        required_cols = {"event_id", "annual_rate"}
        missing = required_cols - set(event_table.columns)
        if missing:
            raise ValueError(f"event_table missing required columns: {missing}")

        self.grid = grid
        self.depths = depths
        self.event_table = event_table.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_npz(cls, path: str) -> "HazardEventSet":
        """Load a previously saved event set (see `.save`)."""
        npz = np.load(path, allow_pickle=True)
        grid = HazardGrid(
            lon_min=float(npz["lon_min"]),
            lat_min=float(npz["lat_min"]),
            cell_size=float(npz["cell_size"]),
            nx=int(npz["nx"]),
            ny=int(npz["ny"]),
        )
        event_table = pd.DataFrame(npz["event_table"].item())
        return cls(grid=grid, depths=npz["depths"], event_table=event_table)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            depths=self.depths,
            lon_min=self.grid.lon_min,
            lat_min=self.grid.lat_min,
            cell_size=self.grid.cell_size,
            nx=self.grid.nx,
            ny=self.grid.ny,
            event_table=self.event_table.to_dict(orient="list"),
        )

    # ------------------------------------------------------------------ #
    # Core operation: sample depths at exposure locations
    # ------------------------------------------------------------------ #
    def depths_at_points(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        Return depth matrix of shape (n_events, n_points): flood depth (m)
        at each exposure location, for every event in the set.
        Points falling outside the hazard grid get depth = 0 (not flooded /
        no data -- conservative choice, override upstream if you'd rather
        flag these explicitly).
        """
        row, col = self.grid.cell_index(np.asarray(lon), np.asarray(lat))
        n_points = len(lon)
        out = np.zeros((len(self.event_table), n_points), dtype=np.float32)
        valid = row >= 0
        if valid.any():
            out[:, valid] = self.depths[:, row[valid], col[valid]]
        return out

    def __len__(self):
        return len(self.event_table)

    def __repr__(self):
        return (
            f"<HazardEventSet: {len(self)} events, "
            f"grid {self.grid.ny}x{self.grid.nx} @ {self.grid.cell_size}deg, "
            f"total annual rate={self.event_table['annual_rate'].sum():.3f}>"
        )
