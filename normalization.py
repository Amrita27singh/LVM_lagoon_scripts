"""
lvm_lagoon.normalization
========================
Normalize and combine multiple LVM exposures into a
single flux-calibrated data cube.

Corresponds to: normalizing_exposures.ipynb
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_exposure(
    flux: np.ndarray,
    ivar: np.ndarray,
    wave: np.ndarray,
    *,
    norm_window: tuple[float, float] = (6540.0, 6580.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a flux array to its median in a reference wavelength window.

    Parameters
    ----------
    flux : ndarray, shape (n_spaxels, n_wave)
    ivar : ndarray, same shape as *flux*
    wave : ndarray, shape (n_wave,)
    norm_window : (float, float)
        Wavelength range (Å) used to compute the normalisation factor.
        Defaults to a continuum window near Hα.

    Returns
    -------
    flux_norm : ndarray  – normalised flux
    ivar_norm : ndarray  – rescaled inverse-variance
    """
    w0, w1 = norm_window
    chan = (wave >= w0) & (wave <= w1)
    if chan.sum() == 0:
        raise ValueError(
            f"norm_window ({w0}, {w1}) Å contains no wavelength channels."
        )

    norm_factor = np.nanmedian(flux[:, chan], axis=1, keepdims=True)  # (n_spaxels, 1)
    # Avoid division by zero for completely empty spaxels
    norm_factor = np.where(norm_factor == 0, 1.0, norm_factor)

    flux_norm = flux / norm_factor
    ivar_norm = ivar * norm_factor ** 2

    logger.debug(
        "Normalization: median factor = %.4f ± %.4f",
        float(np.nanmedian(norm_factor)),
        float(np.nanstd(norm_factor)),
    )
    return flux_norm, ivar_norm


def combine_exposures(
    exposures: Sequence[dict],
    wave: np.ndarray,
    *,
    method: str = "inverse_variance",
    norm_window: tuple[float, float] = (6540.0, 6580.0),
) -> dict:
    """Combine a list of normalized exposures.

    Each element of *exposures* is a dict with at least ``flux`` and
    ``ivar`` arrays (shape n_spaxels × n_wave).

    Parameters
    ----------
    exposures : list of dict
    wave      : 1-D wavelength array (Å)
    method    : ``"inverse_variance"`` (default) or ``"mean"``
    norm_window : passed to :func:`normalize_exposure`

    Returns
    -------
    dict with keys ``flux``, ``ivar``, ``n_exposures``
    """
    if not exposures:
        raise ValueError("exposures list is empty")

    logger.info("Combining %d exposures using method='%s'", len(exposures), method)

    normed = [
        normalize_exposure(exp["flux"], exp["ivar"], wave,
                           norm_window=norm_window)
        for exp in exposures
    ]
    flux_stack = np.stack([f for f, _ in normed], axis=0)   # (n_exp, n_spaxels, n_wave)
    ivar_stack = np.stack([iv for _, iv in normed], axis=0)

    if method == "inverse_variance":
        combined_ivar = np.nansum(ivar_stack, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            combined_flux = np.nansum(flux_stack * ivar_stack, axis=0) / combined_ivar
    elif method == "mean":
        combined_flux = np.nanmean(flux_stack, axis=0)
        combined_ivar = np.nansum(ivar_stack, axis=0)
    else:
        raise ValueError(f"Unknown combination method: '{method}'")

    return dict(flux=combined_flux, ivar=combined_ivar,
                n_exposures=len(exposures))


def sigma_clip_combine(
    flux_stack: np.ndarray,
    ivar_stack: np.ndarray,
    *,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative sigma-clip combination along the first axis (exposures).

    Parameters
    ----------
    flux_stack : ndarray, shape (n_exp, n_spaxels, n_wave)
    ivar_stack : ndarray, same shape
    n_sigma    : clipping threshold

    Returns
    -------
    flux_combined, ivar_combined : ndarray, shape (n_spaxels, n_wave)
    """
    mask = np.zeros_like(flux_stack, dtype=bool)
    for _ in range(5):  # max iterations
        med  = np.nanmedian(np.where(mask, np.nan, flux_stack), axis=0)
        std  = np.nanstd(np.where(mask, np.nan, flux_stack), axis=0)
        mask = np.abs(flux_stack - med[np.newaxis]) > n_sigma * std[np.newaxis]

    flux_stack_clipped = np.where(mask, 0.0, flux_stack)
    ivar_stack_clipped = np.where(mask, 0.0, ivar_stack)

    combined_ivar = np.nansum(ivar_stack_clipped, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined_flux = (
            np.nansum(flux_stack_clipped * ivar_stack_clipped, axis=0)
            / combined_ivar
        )
    return combined_flux, combined_ivar
