"""
financial.py
============
The FINANCIAL LOSS layer of the cat model: converts ground-up physical
damage into the loss actually borne by a (re)insurer or portfolio holder,
by applying per-asset policy terms.

Kept deliberately simple - per-asset deductible and limit, applied
independently per event, which is standard for a first-pass cat model.
Extend this module for layered reinsurance, per-occurrence/aggregate
policy terms, or facultative structures.
"""

from __future__ import annotations

import numpy as np


def apply_policy_terms(ground_up_loss: np.ndarray, deductible: np.ndarray,
                        limit: np.ndarray) -> np.ndarray:
    """
    ground_up_loss : array, shape (n_events, n_assets) - uncapped physical loss
    deductible, limit : arrays, shape (n_assets,)

    Returns the net (insured) loss per event per asset:
        net = min(max(ground_up - deductible, 0), limit)
    """
    gu = np.asarray(ground_up_loss)
    ded = np.asarray(deductible)
    lim = np.asarray(limit)
    net = np.clip(gu - ded, 0.0, None)
    net = np.minimum(net, lim)
    return net
