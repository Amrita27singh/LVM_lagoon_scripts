"""
lvm_lagoon.dap_output
=====================
Read and reformat LVM-DAP output FITS files into a
uniform data structure used by the rest of the pipeline.

Corresponds to: lvm-dap-gen-out-mod.ipynb
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dap_output(
    fits_path: str | Path,
    *,
    ext_flux: str = "FLUX",
    ext_ivar: str = "IVAR",
    ext_mask: str = "MASK",
    ext_wave: str = "WAVE",
) -> dict:
    """Read a LVM-DAP output FITS file and return a structured dict.

    Parameters
    ----------
    fits_path : str or Path
        Path to the DAP output FITS file.
    ext_flux, ext_ivar, ext_mask, ext_wave : str
        FITS extension names for flux, inverse-variance,
        mask, and wavelength arrays.  Override if your
        DAP version uses different names.

    Returns
    -------
    dict with keys
        ``wave``   – 1-D wavelength array  (Å)
        ``flux``   – 3-D flux cube         (fibre × fibre × wave) or 2-D (spaxel × wave)
        ``ivar``   – inverse-variance array, same shape as flux
        ``mask``   – boolean mask array,    same shape as flux
        ``header`` – primary FITS header
        ``meta``   – dict of useful header keywords
    """
    fits_path = Path(fits_path)
    logger.info("Loading DAP output: %s", fits_path)

    with fits.open(fits_path) as hdul:
        header = hdul[0].header

        wave = hdul[ext_wave].data.astype(float)
        flux = hdul[ext_flux].data.astype(float)
        ivar = hdul[ext_ivar].data.astype(float)
        mask = hdul[ext_mask].data.astype(bool)

    meta = _extract_meta(header)

    data = dict(wave=wave, flux=flux, ivar=ivar, mask=mask,
                header=header, meta=meta)
    logger.info(
        "Loaded cube: %d spaxels, %d wavelength channels",
        flux.shape[0], flux.shape[-1],
    )
    return data


def reformat_dap_table(fits_path: str | Path, ext: str = "DAP_TABLE") -> Table:
    """Read a DAP output table extension into an Astropy Table.

    Parameters
    ----------
    fits_path : str or Path
    ext : str
        FITS extension name of the binary table.

    Returns
    -------
    astropy.table.Table
    """
    fits_path = Path(fits_path)
    logger.info("Reading DAP table extension '%s' from %s", ext, fits_path)
    with fits.open(fits_path) as hdul:
        tbl = Table(hdul[ext].data)
    logger.info("Table has %d rows, %d columns", len(tbl), len(tbl.colnames))
    return tbl


def apply_spaxel_mask(data: dict, snr_threshold: float = 3.0) -> dict:
    """Flag additional spaxels with median S/N below *snr_threshold*.

    Updates ``data['mask']`` in-place and returns the same dict.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nanmedian(data["flux"] * np.sqrt(data["ivar"]), axis=-1)

    low_snr = snr < snr_threshold
    n_flagged = int(low_snr.sum())
    logger.info(
        "Masking %d / %d spaxels with S/N < %.1f",
        n_flagged, low_snr.size, snr_threshold,
    )
    data["mask"][low_snr, :] = True
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_meta(header: fits.Header) -> dict:
    """Pull a standard set of keywords from the FITS header."""
    keys = [
        "OBJECT", "RA", "DEC", "EXPTIME", "MJDOBS",
        "INSTRUME", "TELESCOP", "NAXIS1", "NAXIS2",
    ]
    return {k: header.get(k) for k in keys}
