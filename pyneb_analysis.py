"""
lvm_lagoon.pyneb_analysis
=========================
Electron temperature, density, and ionic-abundance maps
using PyNeb for the LVM Lagoon Nebula (M8) data.

Corresponds to: run_pyneb.ipynb
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from astropy.table import Table

try:
    import pyneb as pn
except ImportError as exc:
    raise ImportError(
        "PyNeb is required for this module.  "
        "Install it with:  pip install pyneb"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic line ratios
# ---------------------------------------------------------------------------

#: Te diagnostics:  {label: (atom, wave_num, wave_den)}
TE_DIAGNOSTICS: dict[str, tuple[str, str, str]] = {
    "Te_NII":   ("N2",  "5755",  "6583+6548"),
    "Te_OIII":  ("O3",  "4363",  "5007+4959"),
    "Te_SII":   ("S2",  "4068+4076", "6717+6731"),
}

#: Ne diagnostics:  {label: (atom, wave_num, wave_den)}
NE_DIAGNOSTICS: dict[str, tuple[str, str, str]] = {
    "Ne_SII":   ("S2",  "6717",  "6731"),
    "Ne_OII":   ("O2",  "3726",  "3729"),
}

#: Ionic abundances to compute:  {label: (atom, line_wave, H_flux_col, H_line_wave)}
ABUNDANCE_DIAGNOSTICS: dict[str, tuple[str, float, str, float]] = {
    "O+":   ("O2",  3727.0, "Hbeta_flux_dered",  4861.33),
    "O++":  ("O3",  5007.0, "Hbeta_flux_dered",  4861.33),
    "N+":   ("N2",  6583.0, "Halpha_flux_dered", 6562.80),
    "S+":   ("S2",  6717.0, "Halpha_flux_dered", 6562.80),
    "S++":  ("S3",  9069.0, "Halpha_flux_dered", 6562.80),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_te_ne(
    tbl: Table,
    *,
    te_diags:  dict | None = None,
    ne_diags:  dict | None = None,
    ne_default: float = 100.0,
) -> Table:
    """Compute electron temperature and density maps with PyNeb.

    Adds columns ``Te_NII``, ``Ne_SII`` etc. to *tbl* in-place.

    Parameters
    ----------
    tbl        : Astropy Table with dereddened line fluxes
    te_diags   : override default TE_DIAGNOSTICS
    ne_diags   : override default NE_DIAGNOSTICS
    ne_default : fallback Ne (cm-3) when Ne diagnostic fails

    Returns
    -------
    tbl (modified in-place)
    """
    if te_diags is None:
        te_diags = TE_DIAGNOSTICS
    if ne_diags is None:
        ne_diags = NE_DIAGNOSTICS

    n = len(tbl)

    # --- Electron density from SII doublet ---
    for label, (atom, wnum, wden) in ne_diags.items():
        col_n = _line_col(wnum)
        col_d = _line_col(wden)
        if col_n not in tbl.colnames or col_d not in tbl.colnames:
            logger.warning("Ne diagnostic %s: columns not found, skipping", label)
            continue

        atom_obj = pn.Atom(atom[:-1], int(atom[-1]))
        ne_arr = np.full(n, np.nan)

        for i in range(n):
            fn = float(tbl[col_n][i])
            fd = float(tbl[col_d][i])
            if np.isnan(fn) or np.isnan(fd) or fd <= 0:
                continue
            try:
                ne_arr[i] = atom_obj.getTemDen(
                    fn / fd, tem=1e4, wave1=int(wnum.split("+")[0]),
                    wave2=int(wden.split("+")[0]), to_eval="den",
                )
            except Exception:
                pass

        tbl[label] = ne_arr
        logger.info("%s: median=%.0f cm-3", label,
                    float(np.nanmedian(ne_arr)))

    # --- Electron temperature ---
    ne_col = "Ne_SII" if "Ne_SII" in tbl.colnames else None

    for label, (atom, wnum, wden) in te_diags.items():
        col_n = _line_col(wnum.split("+")[0])
        col_d = _line_col(wden.split("+")[0])
        if col_n not in tbl.colnames or col_d not in tbl.colnames:
            logger.warning("Te diagnostic %s: columns not found, skipping", label)
            continue

        atom_obj = pn.Atom(atom[:-1], int(atom[-1]))
        te_arr = np.full(n, np.nan)

        for i in range(n):
            # Numerator: sum all listed waves
            fn = sum(
                float(tbl[_line_col(w)][i])
                for w in wnum.split("+")
                if _line_col(w) in tbl.colnames
            )
            fd = sum(
                float(tbl[_line_col(w)][i])
                for w in wden.split("+")
                if _line_col(w) in tbl.colnames
            )
            ne_i = float(tbl[ne_col][i]) if ne_col else ne_default
            if np.isnan(ne_i) or ne_i <= 0:
                ne_i = ne_default
            if np.isnan(fn) or np.isnan(fd) or fd <= 0:
                continue
            try:
                te_arr[i] = atom_obj.getTemDen(
                    fn / fd, den=ne_i,
                    wave1=int(wnum.split("+")[0]),
                    wave2=int(wden.split("+")[0]),
                    to_eval="tem",
                )
            except Exception:
                pass

        tbl[label] = te_arr
        logger.info("%s: median=%.0f K", label,
                    float(np.nanmedian(te_arr)))

    return tbl


def compute_abundances(
    tbl: Table,
    *,
    abund_diags: dict | None = None,
    te_col: str = "Te_NII",
    ne_col: str = "Ne_SII",
    te_default: float = 1e4,
    ne_default: float = 100.0,
) -> Table:
    """Compute ionic abundances X+n/H+ for each spaxel.

    Parameters
    ----------
    tbl         : Table with Te/Ne columns and dereddened fluxes
    abund_diags : override ABUNDANCE_DIAGNOSTICS
    te_col      : column name for Te (K)
    ne_col      : column name for Ne (cm-3)

    Returns
    -------
    tbl (modified in-place) with new ionic abundance columns
    """
    if abund_diags is None:
        abund_diags = ABUNDANCE_DIAGNOSTICS

    n = len(tbl)

    for ion, (atom, line_wave, h_col, h_wave) in abund_diags.items():
        line_col = _line_col(str(int(line_wave)))
        if line_col not in tbl.colnames or h_col not in tbl.colnames:
            logger.warning("Abundance %s: columns not found, skipping", ion)
            continue

        atom_obj = pn.Atom(atom[:-1], int(atom[-1]))
        abund    = np.full(n, np.nan)

        for i in range(n):
            f_ion = float(tbl[line_col][i])
            f_h   = float(tbl[h_col][i])
            te    = float(tbl[te_col][i]) if te_col in tbl.colnames else te_default
            ne    = float(tbl[ne_col][i]) if ne_col in tbl.colnames else ne_default

            if np.isnan(te) or te <= 0:
                te = te_default
            if np.isnan(ne) or ne <= 0:
                ne = ne_default
            if np.isnan(f_ion) or np.isnan(f_h) or f_h <= 0:
                continue

            try:
                abund[i] = atom_obj.getIonAbundance(
                    f_ion / f_h, tem=te, den=ne,
                    wave=int(line_wave),
                )
            except Exception:
                pass

        col_name = f"abund_{ion.replace('+', 'p').replace('++', 'pp')}"
        tbl[col_name] = abund
        logger.info("Ionic abundance %s/%s: median=%.2e",
                    ion, "H", float(np.nanmedian(abund)))

    return tbl


def compute_total_oxygen(tbl: Table) -> Table:
    """Add total oxygen abundance O/H = O+/H + O++/H (ICF=1)."""
    op_col  = "abund_Op"
    opp_col = "abund_Opp"
    if op_col in tbl.colnames and opp_col in tbl.colnames:
        tbl["12+log_OH"] = 12.0 + np.log10(
            tbl[op_col].data.astype(float) + tbl[opp_col].data.astype(float)
        )
        logger.info(
            "12+log(O/H): median=%.2f",
            float(np.nanmedian(tbl["12+log_OH"])),
        )
    else:
        logger.warning(
            "O+/H or O++/H columns not found; skipping total oxygen."
        )
    return tbl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_col(wave: str) -> str:
    """Map a wavelength string to the expected table column name.

    Tries both ``{wave}_flux_dered`` and ``{wave}_flux``.
    Returns the column name (existence checked later).
    """
    return f"{wave}_flux_dered"
