"""
lvm_lagoon.pipeline
===================
Orchestrate the full resolved-analysis pipeline:

  Step 1  dap_output      – load & mask the LVM-DAP cube
  Step 2  normalization   – normalize & combine exposures
  Step 3  line_fitting    – Gaussian emission-line fits per spaxel
  Step 4  dust_correction – Balmer-decrement E(B-V) & dereddening
  Step 5  pyneb_analysis  – Te, Ne, ionic abundances via PyNeb
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

from lvm_lagoon import dap_output, normalization, line_fitting
from lvm_lagoon import dust_correction, pyneb_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    dap_fits_path: str | Path,
    output_dir: str | Path = "lvm_results",
    *,
    # Step 1 options
    snr_threshold: float = 3.0,
    # Step 2 options
    extra_exposures: list[dict] | None = None,
    norm_window: tuple[float, float] = (6540.0, 6580.0),
    combine_method: str = "inverse_variance",
    # Step 3 options
    lines: dict[str, float] | None = None,
    snr_min_line: float = 3.0,
    # Step 4 options
    ext_law: str = "odonnell94",
    Rv: float = 3.1,
    # Step 5 options
    te_col: str = "Te_NII",
    ne_col: str = "Ne_SII",
    ne_default: float = 100.0,
    te_default: float = 1e4,
    save_intermediate: bool = True,
) -> Table:
    """Run the full LVM Lagoon resolved-analysis pipeline.

    Parameters
    ----------
    dap_fits_path   : path to the LVM-DAP output FITS file
    output_dir      : directory where output files are written
    snr_threshold   : minimum spaxel S/N to keep (Step 1)
    extra_exposures : additional exposure dicts to combine (Step 2)
    norm_window     : wavelength window for normalisation (Step 2)
    combine_method  : ``"inverse_variance"`` or ``"mean"`` (Step 2)
    lines           : emission-line dict; defaults to DEFAULT_LINES (Step 3)
    snr_min_line    : minimum line S/N to flag as detection (Step 3)
    ext_law         : extinction law name (Step 4)
    Rv              : R_V value (Step 4)
    te_col, ne_col  : column names for Te/Ne input to abundance calc (Step 5)
    ne_default      : fallback Ne when diagnostic fails (Step 5)
    te_default      : fallback Te when diagnostic fails (Step 5)
    save_intermediate : write each step's output to *output_dir*

    Returns
    -------
    final_table : astropy.table.Table
        One row per spaxel with all derived quantities.
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LVM Lagoon Pipeline  –  %s", dap_fits_path)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Load & mask DAP output
    # -----------------------------------------------------------------------
    logger.info("[1/5] Loading DAP output …")
    data = dap_output.load_dap_output(dap_fits_path)
    data = dap_output.apply_spaxel_mask(data, snr_threshold=snr_threshold)

    wave = data["wave"]
    flux = data["flux"]
    ivar = data["ivar"]
    mask = data["mask"]
    logger.info("  → %d spaxels, %.1f – %.1f Å",
                flux.shape[0], wave[0], wave[-1])

    # -----------------------------------------------------------------------
    # Step 2: Normalize (and optionally combine extra exposures)
    # -----------------------------------------------------------------------
    logger.info("[2/5] Normalizing …")
    if extra_exposures:
        all_exps = [dict(flux=flux, ivar=ivar)] + extra_exposures
        combined = normalization.combine_exposures(
            all_exps, wave,
            method=combine_method,
            norm_window=norm_window,
        )
        flux = combined["flux"]
        ivar = combined["ivar"]
        logger.info("  → Combined %d exposures", combined["n_exposures"])
    else:
        flux, ivar = normalization.normalize_exposure(
            flux, ivar, wave, norm_window=norm_window
        )
        logger.info("  → Single-exposure normalisation applied")

    if save_intermediate:
        _save_array(output_dir / "step2_flux_norm.npy", flux)

    # -----------------------------------------------------------------------
    # Step 3: Fit emission lines
    # -----------------------------------------------------------------------
    logger.info("[3/5] Fitting emission lines …")
    if lines is None:
        lines = line_fitting.DEFAULT_LINES

    line_tbl = line_fitting.fit_cube(
        wave, flux, ivar, mask,
        lines=lines, snr_min=snr_min_line,
    )
    if save_intermediate:
        line_tbl.write(output_dir / "step3_line_fits.fits",
                       overwrite=True, format="fits")
        logger.info("  → Saved step3_line_fits.fits")

    # -----------------------------------------------------------------------
    # Step 4: Dust correction
    # -----------------------------------------------------------------------
    logger.info("[4/5] Applying dust correction …")
    ha_flux = line_tbl["Halpha_flux"].data.astype(float)
    hb_flux = line_tbl["Hbeta_flux"].data.astype(float)

    ebv = dust_correction.compute_ebv(
        ha_flux, hb_flux, law=ext_law, Rv=Rv,
    )
    line_tbl = dust_correction.correct_line_table(
        line_tbl, ebv, lines, law=ext_law, Rv=Rv,
    )
    if save_intermediate:
        line_tbl.write(output_dir / "step4_dust_corrected.fits",
                       overwrite=True, format="fits")
        logger.info("  → Saved step4_dust_corrected.fits")

    # -----------------------------------------------------------------------
    # Step 5: PyNeb – Te, Ne, abundances
    # -----------------------------------------------------------------------
    logger.info("[5/5] Running PyNeb analysis …")
    final_tbl = pyneb_analysis.compute_te_ne(
        line_tbl, ne_default=ne_default,
    )
    final_tbl = pyneb_analysis.compute_abundances(
        final_tbl,
        te_col=te_col, ne_col=ne_col,
        te_default=te_default, ne_default=ne_default,
    )
    final_tbl = pyneb_analysis.compute_total_oxygen(final_tbl)

    # -----------------------------------------------------------------------
    # Save final output
    # -----------------------------------------------------------------------
    out_path = output_dir / "lvm_lagoon_results.fits"
    final_tbl.write(out_path, overwrite=True, format="fits")
    logger.info("Final table saved to: %s", out_path)
    logger.info("Pipeline complete in %.1f s", time.time() - t0)

    return final_tbl


# ---------------------------------------------------------------------------
# Step-by-step helpers (for interactive/notebook use)
# ---------------------------------------------------------------------------

def step1_load(dap_fits_path, snr_threshold=3.0):
    """Run only Step 1 (DAP loading & masking)."""
    data = dap_output.load_dap_output(dap_fits_path)
    return dap_output.apply_spaxel_mask(data, snr_threshold=snr_threshold)


def step2_normalize(data, norm_window=(6540.0, 6580.0)):
    """Run only Step 2 (normalization) given a data dict from step1."""
    flux, ivar = normalization.normalize_exposure(
        data["flux"], data["ivar"], data["wave"], norm_window=norm_window,
    )
    return dict(**data, flux=flux, ivar=ivar)


def step3_fit_lines(data, lines=None, snr_min=3.0):
    """Run only Step 3 (line fitting) given a data dict from step2."""
    return line_fitting.fit_cube(
        data["wave"], data["flux"], data["ivar"], data["mask"],
        lines=lines or line_fitting.DEFAULT_LINES, snr_min=snr_min,
    )


def step4_dust_correct(line_tbl, lines, ext_law="odonnell94", Rv=3.1):
    """Run only Step 4 (dust correction) given a line table from step3."""
    ha = line_tbl["Halpha_flux"].data.astype(float)
    hb = line_tbl["Hbeta_flux"].data.astype(float)
    ebv = dust_correction.compute_ebv(ha, hb, law=ext_law, Rv=Rv)
    return dust_correction.correct_line_table(line_tbl, ebv, lines,
                                              law=ext_law, Rv=Rv)


def step5_pyneb(line_tbl, te_col="Te_NII", ne_col="Ne_SII",
                te_default=1e4, ne_default=100.0):
    """Run only Step 5 (PyNeb Te/Ne/abundances) given a table from step4."""
    tbl = pyneb_analysis.compute_te_ne(line_tbl, ne_default=ne_default)
    tbl = pyneb_analysis.compute_abundances(
        tbl, te_col=te_col, ne_col=ne_col,
        te_default=te_default, ne_default=ne_default,
    )
    return pyneb_analysis.compute_total_oxygen(tbl)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save_array(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)
    logger.info("  → Saved %s", path.name)
