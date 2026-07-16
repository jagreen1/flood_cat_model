"""
run_example.py
===============
End-to-end demo of the flood catastrophe modelling engine using the
synthetic data produced by `generate_synthetic_data.py`.

Pipeline:  Hazard -> Vulnerability -> Exposure -> Financial Loss
    1. Load hazard event set, exposure portfolio, vulnerability curves.
    2. Compute the Event Loss Table (ELT): loss per stochastic event.
    3. Monte Carlo simulate a Year Loss Table (YLT) from the ELT.
    4. Derive AAL, Occurrence EP curve, Aggregate EP curve, and a
       standard return-period loss summary table.
    5. Save all outputs (CSVs + plots) to data/output/.

Run: `python run_example.py` (after `python generate_synthetic_data.py`)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from engine.hazard import HazardEventSet
from engine.exposure import ExposurePortfolio
from engine.vulnerability import VulnerabilitySet
from engine.loss_engine import LossEngine
from engine import metrics

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT_DIR = DATA_DIR / "output"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 1. Load inputs
    # ---------------------------------------------------------------- #
    print("Loading hazard, exposure and vulnerability inputs...")
    hazard = HazardEventSet.from_npz(DATA_DIR / "hazard" / "synthetic_events.npz")
    exposure = ExposurePortfolio.from_csv(DATA_DIR / "exposure" / "synthetic_exposure.csv")
    vulnerability = VulnerabilitySet.from_csv(DATA_DIR / "vulnerability" / "synthetic_vulnerability.csv")

    print(f"  {hazard}")
    print(f"  {exposure}")
    print(f"  {vulnerability}")

    # ---------------------------------------------------------------- #
    # 2. Event Loss Table
    # ---------------------------------------------------------------- #
    print("\nComputing Event Loss Table (ELT)...")
    engine = LossEngine(hazard, exposure, vulnerability, seed=42)
    elt = engine.compute_event_loss_table(stochastic_damage=False)
    elt.to_csv(OUT_DIR / "event_loss_table.csv", index=False)
    print(elt.sort_values("gross_loss", ascending=False).head(5).to_string(index=False))

    # ---------------------------------------------------------------- #
    # 3. Year Loss Table (Monte Carlo)
    # ---------------------------------------------------------------- #
    n_years = 100_000
    print(f"\nSimulating {n_years:,} years of experience (Monte Carlo)...")
    ylt = engine.simulate_year_loss_table(n_years=n_years, loss_col="net_loss", seed=123)
    ylt.to_csv(OUT_DIR / "year_loss_table.csv", index=False)

    # ---------------------------------------------------------------- #
    # 4. Metrics: AAL, OEP, AEP
    # ---------------------------------------------------------------- #
    print("\nComputing risk metrics...")
    summary = metrics.summary_metrics(ylt)
    summary_df = pd.DataFrame([summary]).T.rename(columns={0: "value"})
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv")
    print(summary_df.to_string())

    oep = metrics.occurrence_ep_curve(ylt)
    aep = metrics.aggregate_ep_curve(ylt)
    oep.to_csv(OUT_DIR / "oep_curve.csv", index=False)
    aep.to_csv(OUT_DIR / "aep_curve.csv", index=False)

    tiv = exposure.total_insured_value()
    print(f"\nTotal Insured Value: {tiv:,.0f}")
    print(f"AAL: {summary['AAL']:,.0f}  ({summary['AAL'] / tiv * 100:.3f}% of TIV)")

    # ---------------------------------------------------------------- #
    # 5. Plots
    # ---------------------------------------------------------------- #
    print("\nPlotting EP curves...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, curve, title in [
        (axes[0], oep, "Occurrence Exceedance Probability (OEP)"),
        (axes[1], aep, "Aggregate Exceedance Probability (AEP)"),
    ]:
        ax.plot(curve["return_period_years"], curve["loss"] / 1e6, lw=2, color="#2b6cb0")
        ax.set_xscale("log")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Loss (\u0024 millions)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        for rp in (10, 50, 100, 250, 500):
            loss_at_rp = metrics.loss_at_return_period(curve, rp)
            ax.axvline(rp, color="grey", lw=0.5, ls="--", alpha=0.5)

    fig.suptitle("Flood Catastrophe Model - Exceedance Probability Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ep_curves.png", dpi=150)
    print(f"  Saved {OUT_DIR / 'ep_curves.png'}")

    # Annual loss distribution histogram (non-zero years) for context
    fig2, ax2 = plt.subplots(figsize=(7.5, 5))
    nonzero = ylt.loc[ylt["aggregate_loss"] > 0, "aggregate_loss"] / 1e6
    ax2.hist(nonzero, bins=80, color="#c05621", alpha=0.85)
    ax2.axvline(summary["AAL"] / 1e6, color="black", lw=1.5, label=f"AAL = ${summary['AAL']/1e6:.2f}m")
    ax2.set_xlabel("Annual aggregate loss (\u0024 millions)")
    ax2.set_ylabel("Simulated years (count)")
    ax2.set_title("Simulated Annual Aggregate Loss Distribution")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "annual_loss_distribution.png", dpi=150)
    print(f"  Saved {OUT_DIR / 'annual_loss_distribution.png'}")

    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
