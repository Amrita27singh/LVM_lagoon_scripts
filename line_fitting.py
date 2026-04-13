"""
lvm_lagoon.line_fitting
=======================
Gaussian emission-line fitting for LVM IFU spectra.

Corresponds to: fitting_lines.ipynb
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from astropy.table import Table

try:
    from scipy.optimize import curve_fit, OptimizeWarning
    from scipy.ndimage import median_filter
except ImportError as e:
    raise ImportError("scipy is required for line fitting.") from e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Line catalogue
# ---------------------------------------------------------------------------

#: Default set of emission lines used by the pipeline (vacuum wavelengths, Å)
DEFAULT_LINES: dict[str, float] = {
    "Hbeta":     4861.33,
    "OIII_4959": 4958.91,
    "OIII_5007": 5006.84,
    "NII_6548":  6548.05,
    "Halpha":    6562.80,
    "NII_6583":  6583.45,
    "SII_6717":  6716.44,
    "SII_6731":  6730.82,
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LineFitResult:
    """Result of a single Gaussian fit to one emission line."""
    name:      str
    wave0:     float          # rest wavelength (Å)
    amplitude: float          # peak amplitude (flux units)
    center:    float          # fitted line center (Å)
    sigma:     float          # Gaussian sigma (Å)
    continuum: float          # local continuum level
    amp_err:   float = np.nan
    cen_err:   float = np.nan
    sig_err:   float = np.nan
    chi2:      float = np.nan
    flag:      int   = 0      # 0 = good, 1 = fit failed, 2 = S/N < threshold

    @property
    def flux(self) -> float:
        """Integrated line flux = amplitude × sigma × sqrt(2π)."""
        return self.amplitude * self.sigma * np.sqrt(2.0 * np.pi)

    @property
    def flux_err(self) -> float:
        """Propagated flux uncertainty."""
        s, a = self.sigma, self.amplitude
        sig_f = np.sqrt(2 * np.pi) * np.sqrt(
            (s * self.amp_err) ** 2 + (a * self.sig_err) ** 2
        )
        return sig_f

    @property
    def snr(self) -> float:
        return self.flux / self.flux_err if self.flux_err > 0 else 0.0

    @property
    def ew(self) -> float:
        """Rest-frame equivalent width (Å).  Negative = emission."""
        if self.continuum <= 0:
            return np.nan
        return -self.flux / self.continuum


# ---------------------------------------------------------------------------
# Gaussian helpers
# ---------------------------------------------------------------------------

def _gaussian(x, amp, cen, sig, cont):
    return amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) + cont


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_line(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    name: str,
    wave0: float,
    *,
    window: float = 30.0,
    sigma_init: float = 2.0,
    snr_min: float = 3.0,
) -> LineFitResult:
    """Fit a single Gaussian to one emission line in a 1-D spectrum.

    Parameters
    ----------
    wave, flux, ivar : 1-D arrays
    name  : line identifier string
    wave0 : rest-frame wavelength (Å)
    window : ±half-window around *wave0* to include in the fit (Å)
    sigma_init : initial Gaussian sigma (Å)
    snr_min : minimum S/N to consider a detection

    Returns
    -------
    LineFitResult
    """
    # Extract fitting window
    idx = (wave >= wave0 - window) & (wave <= wave0 + window)
    if idx.sum() < 5:
        return LineFitResult(name=name, wave0=wave0, amplitude=0, center=wave0,
                             sigma=sigma_init, continuum=0, flag=1)

    w, f = wave[idx], flux[idx]
    err = np.where(ivar[idx] > 0, 1.0 / np.sqrt(ivar[idx]), np.inf)

    # Continuum estimate from 20 % of window edges
    edge_n = max(2, int(0.2 * len(w)))
    cont0  = float(np.nanmedian(np.concatenate([f[:edge_n], f[-edge_n:]])))
    amp0   = float(np.nanmax(f) - cont0)

    p0 = [amp0, wave0, sigma_init, cont0]
    bounds = (
        [0,         wave0 - 10, 0.5,   -np.inf],
        [np.inf,    wave0 + 10, window, np.inf],
    )

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                _gaussian, w, f, p0=p0, sigma=err, bounds=bounds,
                maxfev=5000, absolute_sigma=True,
            )
        perr = np.sqrt(np.diag(pcov))

        # Goodness of fit
        resid = f - _gaussian(w, *popt)
        chi2  = float(np.sum((resid / err) ** 2) / max(len(w) - 4, 1))

        result = LineFitResult(
            name=name, wave0=wave0,
            amplitude=popt[0], center=popt[1],
            sigma=popt[2],     continuum=popt[3],
            amp_err=perr[0],   cen_err=perr[1],
            sig_err=perr[2],   chi2=chi2,
        )
        if result.snr < snr_min:
            result.flag = 2

    except (RuntimeError, ValueError) as exc:
        logger.debug("Fit failed for line %s: %s", name, exc)
        result = LineFitResult(
            name=name, wave0=wave0, amplitude=0, center=wave0,
            sigma=sigma_init, continuum=cont0, flag=1,
        )

    return result


def fit_all_lines(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    lines: dict[str, float] | None = None,
    **kwargs,
) -> dict[str, LineFitResult]:
    """Fit a catalogue of emission lines in a single spectrum.

    Parameters
    ----------
    wave, flux, ivar : 1-D arrays
    lines : dict {name: rest_wavelength_Å}.  Uses DEFAULT_LINES if None.
    **kwargs : passed to :func:`fit_line`

    Returns
    -------
    dict {line_name: LineFitResult}
    """
    if lines is None:
        lines = DEFAULT_LINES
    return {name: fit_line(wave, flux, ivar, name, w0, **kwargs)
            for name, w0 in lines.items()}


def fit_cube(
    wave: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    lines: dict[str, float] | None = None,
    **kwargs,
) -> Table:
    """Fit emission lines in every spaxel of a data cube.

    Parameters
    ----------
    wave : 1-D array, shape (n_wave,)
    flux : 2-D array, shape (n_spaxels, n_wave)
    ivar : same shape as flux
    mask : boolean, same shape — True = bad
    lines : emission-line dict

    Returns
    -------
    astropy.table.Table with one row per spaxel and columns for each
    line parameter (flux, snr, sigma, ew, flag, …)
    """
    if lines is None:
        lines = DEFAULT_LINES

    n_spaxels = flux.shape[0]
    records = []

    for i in range(n_spaxels):
        if mask[i].all():
            row = {"spaxel": i}
            for name in lines:
                row[f"{name}_flux"]  = np.nan
                row[f"{name}_snr"]   = np.nan
                row[f"{name}_sigma"] = np.nan
                row[f"{name}_ew"]    = np.nan
                row[f"{name}_flag"]  = 1
            records.append(row)
            continue

        fi   = np.where(mask[i], 0.0, flux[i])
        ivri = np.where(mask[i], 0.0, ivar[i])
        fits_i = fit_all_lines(wave, fi, ivri, lines=lines, **kwargs)

        row = {"spaxel": i}
        for name, res in fits_i.items():
            row[f"{name}_flux"]  = res.flux
            row[f"{name}_snr"]   = res.snr
            row[f"{name}_sigma"] = res.sigma
            row[f"{name}_ew"]    = res.ew
            row[f"{name}_flag"]  = res.flag
        records.append(row)

        if (i + 1) % 500 == 0:
            logger.info("  … fitted %d / %d spaxels", i + 1, n_spaxels)

    logger.info("Line fitting complete: %d spaxels", n_spaxels)
    return Table(records)
