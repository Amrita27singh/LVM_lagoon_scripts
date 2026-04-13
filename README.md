# lvm-lagoon

A Python package for **resolved spectral analysis of LVM IFU data** of the Lagoon Nebula (M8).

Converts the five Jupyter notebooks in this repo into a clean, reusable, testable pipeline:

| Step | Notebook | Module |
|------|----------|--------|
| 1 | `lvm-dap-gen-out-mod.ipynb` | `lvm_lagoon.dap_output` |
| 2 | `normalizing_exposures.ipynb` | `lvm_lagoon.normalization` |
| 3 | `fitting_lines.ipynb` | `lvm_lagoon.line_fitting` |
| 4 | `dust_correction.ipynb` | `lvm_lagoon.dust_correction` |
| 5 | `run_pyneb.ipynb` | `lvm_lagoon.pyneb_analysis` |

---

## Installation

### From GitHub (recommended while in development)

```bash
pip install git+https://github.com/Amrita27singh/LVM_lagoon_scripts.git
```

### For development (editable install)

```bash
git clone https://github.com/Amrita27singh/LVM_lagoon_scripts.git
cd LVM_lagoon_scripts
pip install -e ".[dev]"
```

### Dependencies

- `numpy`, `scipy`, `astropy`, `matplotlib`
- [`pyneb`](https://github.com/Morisset/PyNeb_devel) (for Step 5)

---

## Quick Start

### One-line pipeline

```python
from lvm_lagoon import run_pipeline

results = run_pipeline(
    "path/to/lvm_dap_output.fits",
    output_dir="my_results/",
    snr_threshold=3.0,
    ext_law="odonnell94",
)
print(results.colnames)
```

### Command line

```bash
lvm-lagoon path/to/dap_output.fits --output-dir results/ --snr 3.0 --verbose
```

### Step-by-step (notebook style)

```python
from lvm_lagoon.pipeline import (
    step1_load, step2_normalize, step3_fit_lines,
    step4_dust_correct, step5_pyneb,
)
from lvm_lagoon.line_fitting import DEFAULT_LINES

# Step 1 – load
data = step1_load("dap_output.fits", snr_threshold=3.0)

# Step 2 – normalize
data = step2_normalize(data)

# Step 3 – fit emission lines
line_tbl = step3_fit_lines(data, lines=DEFAULT_LINES)

# Step 4 – dust correction
line_tbl = step4_dust_correct(line_tbl, lines=DEFAULT_LINES)

# Step 5 – PyNeb Te/Ne/abundances
final = step5_pyneb(line_tbl)

final.write("results.fits", overwrite=True)
```

---

## Module Reference

### `lvm_lagoon.dap_output`
- `load_dap_output(fits_path)` – Read LVM-DAP FITS into a dict of arrays
- `apply_spaxel_mask(data, snr_threshold)` – Flag low-S/N spaxels

### `lvm_lagoon.normalization`
- `normalize_exposure(flux, ivar, wave)` – Normalize to a continuum window
- `combine_exposures(exposures, wave)` – Inverse-variance combine multiple frames
- `sigma_clip_combine(flux_stack, ivar_stack)` – Iterative sigma-clip combination

### `lvm_lagoon.line_fitting`
- `fit_line(wave, flux, ivar, name, wave0)` – Gaussian fit to one line
- `fit_all_lines(wave, flux, ivar)` – Fit all lines in DEFAULT_LINES
- `fit_cube(wave, flux, ivar, mask)` – Fit all spaxels, returns Astropy Table

Default lines: `Hβ, [OIII]4959/5007, [NII]6548/6583, Hα, [SII]6717/6731`

### `lvm_lagoon.dust_correction`
- `compute_ebv(ha_flux, hb_flux)` – E(B-V) from Balmer decrement
- `deredden_flux(flux, ebv, wavelengths)` – Apply extinction correction
- `correct_line_table(tbl, ebv, line_waves)` – Add `_flux_dered` columns to table
- Supported laws: `"ccm89"` (Cardelli+89), `"odonnell94"` (O'Donnell 94)

### `lvm_lagoon.pyneb_analysis`
- `compute_te_ne(tbl)` – Electron T/N maps via [NII], [OIII], [SII] diagnostics
- `compute_abundances(tbl)` – Ionic abundances O⁺, O⁺⁺, N⁺, S⁺, S⁺⁺
- `compute_total_oxygen(tbl)` – Add `12+log(O/H)` column

---

## Running tests

```bash
pytest tests/ -v
```

---

## Citing

If you use this pipeline, please cite the LVM survey paper and the relevant atomic-data sources (PyNeb, CCM89, O'Donnell 1994).

---

## License

MIT
