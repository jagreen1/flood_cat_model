"""
Flood Catastrophe Modelling Engine
===================================
A minimal, transparent implementation of the standard
Hazard -> Vulnerability -> Exposure -> Financial Loss catastrophe
modelling structure used across the insurance / reinsurance industry.

Quick start
-----------
    from engine.hazard import HazardEventSet
    from engine.exposure import ExposurePortfolio
    from engine.vulnerability import VulnerabilitySet
    from engine.loss_engine import LossEngine
    from engine import metrics

    hazard = HazardEventSet.from_npz("data/hazard/synthetic_events.npz")
    exposure = ExposurePortfolio.from_csv("data/exposure/synthetic_exposure.csv")
    vulnerability = VulnerabilitySet.from_csv("data/vulnerability/synthetic_vulnerability.csv")

    engine = LossEngine(hazard, exposure, vulnerability)
    elt = engine.compute_event_loss_table()
    ylt = engine.simulate_year_loss_table(n_years=100_000)

    aal = metrics.average_annual_loss(ylt)
    oep = metrics.occurrence_ep_curve(ylt)
    aep = metrics.aggregate_ep_curve(ylt)
"""

from . import hazard, exposure, vulnerability, financial, loss_engine, metrics, data_sources

__all__ = [
    "hazard", "exposure", "vulnerability", "financial",
    "loss_engine", "metrics", "data_sources",
]
