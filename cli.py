"""
lvm_lagoon.cli
==============
Command-line entry point for the LVM Lagoon pipeline.

Usage
-----
    lvm-lagoon path/to/dap_output.fits --output-dir results/ --snr 3.0
"""

from __future__ import annotations

import argparse
import logging
import sys

from lvm_lagoon.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lvm-lagoon",
        description="Run the LVM Lagoon resolved-analysis pipeline.",
    )
    p.add_argument("dap_fits", metavar="DAP_FITS",
                   help="Path to the LVM-DAP output FITS file")
    p.add_argument("--output-dir", "-o", default="lvm_results",
                   help="Output directory (default: lvm_results/)")
    p.add_argument("--snr", type=float, default=3.0,
                   help="Minimum spaxel S/N (default: 3.0)")
    p.add_argument("--ext-law", default="odonnell94",
                   choices=["ccm89", "odonnell94"],
                   help="Dust extinction law (default: odonnell94)")
    p.add_argument("--Rv", type=float, default=3.1,
                   help="R_V value (default: 3.1)")
    p.add_argument("--no-save-intermediate", action="store_true",
                   help="Skip writing intermediate FITS files")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable DEBUG logging")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(
            args.dap_fits,
            output_dir=args.output_dir,
            snr_threshold=args.snr,
            ext_law=args.ext_law,
            Rv=args.Rv,
            save_intermediate=not args.no_save_intermediate,
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
