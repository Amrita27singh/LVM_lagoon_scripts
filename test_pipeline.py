"""
tests/test_pipeline.py
======================
Unit tests for the lvm_lagoon pipeline modules.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from astropy.table import Table


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:

    def _make_data(self, n_spaxels=10, n_wave=200):
        rng = np.random.default_rng(42)
        wave = np.linspace(4700, 7000, n_wave)
        flux = rng.uniform(0.5, 2.0, (n_spaxels, n_wave))
        ivar = np.ones_like(flux) * 10
        return wave, flux, ivar

    def test_normalize_shape(self):
        from lvm_lagoon.normalization import normalize_exposure
        wave, flux, ivar = self._make_data()
        fn, ivn = normalize_exposure(flux, ivar, wave)
        assert fn.shape == flux.shape
        assert ivn.shape == ivar.shape

    def test_normalize_median_near_one(self):
        from lvm_lagoon.normalization import normalize_exposure
        wave, flux, ivar = self._make_data()
        fn, _ = normalize_exposure(flux, ivar, wave,
                                   norm_window=(6540., 6580.))
        med_in_window = np.nanmedian(fn[:, (wave >= 6540) & (wave <= 6580)])
        assert abs(med_in_window - 1.0) < 0.15

    def test_combine_two_exposures(self):
        from lvm_lagoon.normalization import combine_exposures
        wave, flux, ivar = self._make_data()
        exp1 = dict(flux=flux,        ivar=ivar)
        exp2 = dict(flux=flux * 1.05, ivar=ivar * 0.9)
        out = combine_exposures([exp1, exp2], wave)
        assert out["flux"].shape == flux.shape
        assert out["n_exposures"] == 2

    def test_bad_norm_window_raises(self):
        from lvm_lagoon.normalization import normalize_exposure
        wave, flux, ivar = self._make_data()
        with pytest.raises(ValueError, match="norm_window"):
            normalize_exposure(flux, ivar, wave, norm_window=(9000., 9100.))


# ---------------------------------------------------------------------------
# Line fitting
# ---------------------------------------------------------------------------

class TestLineFitting:

    def _gaussian_spectrum(self, amplitude=100., center=6562.8, sigma=2.5,
                           cont=5., n_wave=300):
        wave = np.linspace(6480, 6650, n_wave)
        flux = amplitude * np.exp(-0.5 * ((wave - center) / sigma) ** 2) + cont
        ivar = np.ones_like(flux) / (flux * 0.01) ** 2
        return wave, flux, ivar

    def test_single_line_detected(self):
        from lvm_lagoon.line_fitting import fit_line
        wave, flux, ivar = self._gaussian_spectrum()
        res = fit_line(wave, flux, ivar, "Halpha", 6562.8, snr_min=3.0)
        assert res.flag == 0
        assert abs(res.center - 6562.8) < 1.0

    def test_flux_positive(self):
        from lvm_lagoon.line_fitting import fit_line
        wave, flux, ivar = self._gaussian_spectrum(amplitude=50.)
        res = fit_line(wave, flux, ivar, "Halpha", 6562.8)
        assert res.flux > 0

    def test_fit_cube_returns_table(self):
        from lvm_lagoon.line_fitting import fit_cube
        n_sp, n_w = 5, 300
        wave = np.linspace(6480, 6650, n_w)
        flux = np.ones((n_sp, n_w)) * 5.0
        ivar = np.ones_like(flux)
        mask = np.zeros_like(flux, dtype=bool)
        tbl = fit_cube(wave, flux, ivar, mask,
                       lines={"Halpha": 6562.8})
        assert isinstance(tbl, Table)
        assert len(tbl) == n_sp
        assert "Halpha_flux" in tbl.colnames


# ---------------------------------------------------------------------------
# Dust correction
# ---------------------------------------------------------------------------

class TestDustCorrection:

    def test_ebv_zero_for_intrinsic_ratio(self):
        from lvm_lagoon.dust_correction import compute_ebv, BALMER_HA_HB_INTRINSIC
        ha = np.array([BALMER_HA_HB_INTRINSIC])
        hb = np.array([1.0])
        ebv = compute_ebv(ha, hb)
        assert float(ebv[0]) == pytest.approx(0.0, abs=1e-6)

    def test_ebv_positive_for_excess_ratio(self):
        from lvm_lagoon.dust_correction import compute_ebv
        ha = np.array([4.5])
        hb = np.array([1.0])
        ebv = compute_ebv(ha, hb)
        assert float(ebv[0]) > 0.0

    def test_deredden_increases_flux(self):
        from lvm_lagoon.dust_correction import deredden_flux
        flux = np.array([[1.0, 1.0]])
        ebv  = np.array([0.3])
        waves = np.array([4861.33, 6562.80])
        corr = deredden_flux(flux, ebv, waves)
        assert (corr >= flux).all()

    def test_ccm89_law_shape(self):
        from lvm_lagoon.dust_correction import ccm89
        waves = np.linspace(3000, 9000, 100)
        result = ccm89(waves)
        assert result.shape == waves.shape
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# PyNeb analysis (skipped if pyneb not installed)
# ---------------------------------------------------------------------------

pyneb = pytest.importorskip("pyneb")

class TestPynebAnalysis:

    def _make_tbl(self, n=20):
        rng = np.random.default_rng(0)
        tbl = Table()
        tbl["spaxel"]              = np.arange(n)
        tbl["Halpha_flux_dered"]   = rng.uniform(200, 400, n)
        tbl["Hbeta_flux_dered"]    = rng.uniform(70,  120, n)
        tbl["NII_6583_flux_dered"] = rng.uniform(50,  150, n)
        tbl["NII_5755_flux_dered"] = rng.uniform(2,   10,  n)
        tbl["SII_6717_flux_dered"] = rng.uniform(30,  80,  n)
        tbl["SII_6731_flux_dered"] = rng.uniform(25,  70,  n)
        tbl["OIII_5007_flux_dered"]= rng.uniform(100, 300, n)
        tbl["Hbeta_flux"]          = tbl["Hbeta_flux_dered"]
        tbl["Halpha_flux"]         = tbl["Halpha_flux_dered"]
        return tbl

    def test_te_ne_adds_columns(self):
        from lvm_lagoon.pyneb_analysis import compute_te_ne
        tbl = self._make_tbl()
        out = compute_te_ne(tbl)
        # At least Ne_SII should be attempted
        assert "Ne_SII" in out.colnames or "Te_NII" in out.colnames

    def test_total_oxygen_column(self):
        from lvm_lagoon.pyneb_analysis import compute_total_oxygen
        tbl = Table()
        tbl["abund_Op"]  = np.array([1e-4, 2e-4])
        tbl["abund_Opp"] = np.array([3e-4, 4e-4])
        out = compute_total_oxygen(tbl)
        assert "12+log_OH" in out.colnames
        assert np.all(np.isfinite(out["12+log_OH"]))
