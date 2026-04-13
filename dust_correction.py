"""
lvm_lagoon.dust_correction
==========================
Balmer-decrement dust correction for LVM IFU emission-line maps.

Corresponds to: dust_correction.ipynb
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from astropy.table import Table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intrinsic Balmer ratios (Case B, Te~10000 K, ne~100 cm-3)
# ---------------------------------------------------------------------------
BALMER_HA_HB_INTRINSIC = 2.86   # Hα / Hβ


# ---------------------------------------------------------------------------
# Extinction laws
# ---------------------------------------------------------------------------

def ccm89(wave_aa: np.ndarray, Rv: float = 3.1) -> np.ndarray:
    """Cardelli, Clayton & Mathis (1989) extinction law.

    Parameters
    ----------
    wave_aa : wavelengths in Angstrom
    Rv      : ratio of total to selective extinction (default 3.1)

    Returns
    -------
    A(λ) / E(B-V)  as a function of wavelength
    """
    x = 1e4 / wave_aa   # inverse micron
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # Optical/NIR:  1.1 ≤ x ≤ 3.3
    opt = (x >= 1.1) & (x <= 3.3)
    y   = x[opt] - 1.82
    a[opt] = (1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3
              + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6
              + 0.32999 * y**7)
    b[opt] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3
              - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6
              - 2.09002 * y**7)

    # UV:  3.3 ≤ x ≤ 8.0
    uv  = (x >= 3.3) & (x <= 8.0)
    fa  = np.where((x >= 5.9) & uv,
                   -0.04473 * (x - 5.9)**2 - 0.009779 * (x - 5.9)**3, 0.0)
    fb  = np.where((x >= 5.9) & uv,
                    0.2130  * (x - 5.9)**2 + 0.1207   * (x - 5.9)**3, 0.0)
    a[uv] = 1.752 - 0.316 * x[uv] - 0.104 / ((x[uv] - 4.67)**2 + 0.341) + fa[uv]
    b[uv] = -3.090 + 1.825 * x[uv] + 1.206 / ((x[uv] - 4.62)**2 + 0.263) + fb[uv]

    return (a + b / Rv) * Rv   # = A(λ) / E(B-V)  ×  Rv / Rv  → A(λ)/E(B-V)


def odonnell94(wave_aa: np.ndarray, Rv: float = 3.1) -> np.ndarray:
    """O'Donnell (1994) update to CCM89 in the optical."""
    # Use CCM89 as base and replace optical coefficients
    x = 1e4 / wave_aa
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    opt = (x >= 1.1) & (x <= 3.3)
    y = x[opt] - 1.82
    a[opt] = (1 - 0.505 * y - 0.598 * y**2 + 1.191 * y**3 + 0.701 * y**4
              - 1.230 * y**5 - 0.508 * y**6 + 0.874 * y**7)
    b[opt] = (0 + 1.952 * y + 2.908 * y**2 - 4.635 * y**3 - 3.225 * y**4
              + 10.099 * y**5 + 0.659 * y**6 - 7.431 * y**7)
    result = ccm89(wave_aa, Rv)
    result[opt] = (a[opt] + b[opt] / Rv) * Rv
    return result


LAWS: dict[str, Callable] = {
    "ccm89":      ccm89,
    "odonnell94": odonnell94,
}


# ---------------------------------------------------------------------------
# E(B-V) map from Balmer decrement
# ---------------------------------------------------------------------------

def compute_ebv(
    ha_flux: np.ndarray,
    hb_flux: np.ndarray,
    *,
    law: str = "odonnell94",
    Rv: float = 3.1,
    ha_wave: float = 6562.80,
    hb_wave: float = 4861.33,
    ratio_intrinsic: float = BALMER_HA_HB_INTRINSIC,
) -> np.ndarray:
    """Compute E(B-V) from Hα/Hβ flux ratio.

    Parameters
    ----------
    ha_flux, hb_flux : ndarray (n_spaxels,)
    law   : extinction law name, one of ``"ccm89"`` or ``"odonnell94"``
    Rv    : selective-to-total extinction ratio
    ha_wave, hb_wave : wavelengths in Å
    ratio_intrinsic  : theoretical Balmer decrement

    Returns
    -------
    ebv : ndarray (n_spaxels,),  NaN where ratio < intrinsic or Hβ ≤ 0
    """
    ext_fn = LAWS[law]

    wave_pair = np.array([ha_wave, hb_wave])
    A_per_ebv = ext_fn(wave_pair, Rv)          # [A(Hα)/E(B-V), A(Hβ)/E(B-V)]
    delta_A   = A_per_ebv[1] - A_per_ebv[0]   # A(Hβ) - A(Hα) per E(B-V)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_obs = ha_flux / hb_flux

    with np.errstate(invalid="ignore", divide="ignore"):
        ebv = 2.5 / delta_A * np.log10(ratio_obs / ratio_intrinsic)

    ebv = np.where(ratio_obs < ratio_intrinsic, 0.0, ebv)   # floor at 0
    ebv = np.where(hb_flux <= 0, np.nan, ebv)
    logger.info(
        "E(B-V): median=%.3f, max=%.3f (law=%s)",
        float(np.nanmedian(ebv)), float(np.nanmax(np.nan_to_num(ebv))), law,
    )
    return ebv


# ---------------------------------------------------------------------------
# Dereddening
# ---------------------------------------------------------------------------

def deredden_flux(
    flux: np.ndarray,
    ebv:  np.ndarray,
    wavelengths: np.ndarray,
    *,
    law: str = "odonnell94",
    Rv: float = 3.1,
) -> np.ndarray:
    """Correct emission-line fluxes for dust extinction.

    Parameters
    ----------
    flux        : ndarray, shape (n_spaxels, n_lines)
    ebv         : ndarray, shape (n_spaxels,)
    wavelengths : ndarray, shape (n_lines,) – rest-frame Å for each line
    law, Rv     : extinction-law choice

    Returns
    -------
    flux_corr : ndarray, same shape as *flux*
    """
    ext_fn  = LAWS[law]
    A_per_ebv = ext_fn(wavelengths, Rv)          # (n_lines,)
    A         = ebv[:, np.newaxis] * A_per_ebv   # (n_spaxels, n_lines)

    flux_corr = flux * 10.0 ** (0.4 * A)
    flux_corr = np.where(np.isnan(A), flux, flux_corr)   # keep original where E(B-V) is NaN
    return flux_corr


def correct_line_table(
    tbl: Table,
    ebv: np.ndarray,
    line_waves: dict[str, float],
    *,
    law: str = "odonnell94",
    Rv: float = 3.1,
) -> Table:
    """Apply dust correction to an Astropy line-flux Table in-place.

    Adds columns ``{name}_flux_dered`` for each line in *line_waves*.

    Parameters
    ----------
    tbl        : table from :func:`~lvm_lagoon.line_fitting.fit_cube`
    ebv        : 1-D array (n_spaxels,)
    line_waves : dict {colname_prefix: rest_wavelength_Å}

    Returns
    -------
    tbl (modified in-place)
    """
    names = list(line_waves.keys())
    waves = np.array([line_waves[n] for n in names])
    raw   = np.column_stack([tbl[f"{n}_flux"] for n in names]).astype(float)

    corr = deredden_flux(raw, ebv, waves, law=law, Rv=Rv)

    for j, name in enumerate(names):
        tbl[f"{name}_flux_dered"] = corr[:, j]

    tbl["ebv"] = ebv
    logger.info("Dust correction applied to %d lines in %d spaxels",
                len(names), len(tbl))
    return tbl
