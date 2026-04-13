"""
lvm_lagoon
==========
A pipeline for resolved spectral analysis of LVM IFU data.

Pipeline steps
--------------
1. dap_output      – Read & reformat LVM-DAP output cubes
2. normalization   – Normalize/combine multi-exposure spectra
3. line_fitting    – Gaussian emission-line fitting
4. dust_correction – Balmer-decrement dust correction
5. pyneb_analysis  – PyNeb Te/Ne / ionic-abundance maps

Quick start
-----------
>>> from lvm_lagoon.pipeline import run_pipeline
>>> run_pipeline("path/to/dap_output.fits", output_dir="results/")
"""

__version__ = "0.1.0"
__author__  = "Amrita Singh"

from lvm_lagoon.pipeline import run_pipeline  # noqa: F401
