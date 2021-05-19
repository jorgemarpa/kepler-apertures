import os
import sys
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy import units
import lightkurve as lk
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

sys.path.append("%s/Work/BAERI/ADAP/kepler-apertures/" % os.environ["HOME"])
from kepler_apertures import EXBAMachine
from kepler_apertures.utils import get_bls_periods

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=None,
    help="Which quarter.",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    default=False,
    help="Save models.",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    default=False,
    help="dry run.",
)
args = parser.parse_args()


# @profile
def run_code():

    hdu = fits.open(
        "../data/export/%i/%02i/image_gaiadr3.fits.gz" % (args.quarter, args.channel)
    )
    hdr = hdu[0].header
    img = hdu[1].data
    catalog = Table(hdu[2].data).to_pandas()

    # load LCs
    lc_files = np.sort(
        glob.glob(
            "../data/export/%i/%02i/lc_Gaia_EDR3_*.fits.gz"
            % (args.quarter, args.channel)
        )
    )
    apertures = ["O", 1, 2, 3, 4, 5, 6, 7]

    lcs = []
    for f in tqdm(lc_files, desc="Loading LCs"):
        meta = fits.open(f)[0].header
        lc = lk.KeplerLightCurve.read(f)
        lc.label = meta["LABEL"]
        lcs.append(lc)

    # creating Design matrix
    times = lcs[0].time.jd
    breaks = list(np.where(np.diff(times) > 0.3)[0] + 1)
    # spline DM
    n_knots = int((lc.time[-1] - lc.time[0]).value / 0.5)
    spline_dm = lk.designmatrix.create_spline_matrix(
        times,
        n_knots=n_knots,
        include_intercept=True,
        name="spline_dm",
    )  # .split(breaks)
    spline_dm = lk.DesignMatrix(
        spline_dm.X[:, spline_dm.X.sum(axis=0) != 0], name="spline_dm"
    )
    # offset matrix
    offset_dm = lk.DesignMatrix(np.ones_like(times), name="offset")  # .split(breaks)
    offset_dm.prior_mu = np.ones(1)
    offset_dm.prior_sigma = np.ones(1) * 0.000001

    # DM with CBVs, first 4 only
    cbvs = lk.correctors.download_kepler_cbvs(
        mission="Kepler", quarter=args.quarter, channel=args.channel
    )
    basis = 4
    cbv_dm = cbvs[np.in1d(cbvs.cadenceno.value, lc.cadenceno.value)].to_designmatrix()
    cbv_dm = lk.DesignMatrix(cbv_dm.X[:, :basis], name="cbv").split(breaks)
    cbv_dm.prior_mu = np.ones(basis * (len(breaks) + 1))
    cbv_dm.prior_sigma = np.ones(basis * (len(breaks) + 1)) * 100000

    # collection of DMs
    dmc = lk.DesignMatrixCollection([offset_dm, cbv_dm, spline_dm])

    pmin, pmax, ffrac = 0.5, 30, 10
    duration_in = [0.03, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.33, 0.4, 0.45]
    period_in = 1 / np.linspace(1 / pmin, 1 / pmax, 60000)

    corrected_lcs = []
    for i, lc in tqdm(enumerate(lcs), desc="Regressor Corrector: ", total=len(lcs)):
        lc_cor = lc.copy()
        for ap in apertures:
            lc_aux = lc.copy()
            ext = ap if ap != "O" else ""
            lc_aux.flux = lc["flux%s" % str(ext)]
            lc_aux.flux_err = lc["flux_err%s" % str(ext)]
            rc = lk.RegressionCorrector(lc_aux)
            try:
                clc = rc.correct(dmc, sigma=3)
            except np.linalg.LinAlgError:
                clc = lc.copy()
            lc_cor["flux%s" % str(ext)] = clc.flux
            lc_cor["flux_err%s" % str(ext)] = clc.flux_err

        corrected_lcs.append(lc_cor)
        # if args.save:

    return


if __name__ == "__main__":
    print("Running BLS for Q: %i Ch: %i" % (args.quarter, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    if args.channel in [5, 6, 7, 8]:
        print("Channles with no data, exiting.")
        sys.exit()
    run_code()

    print("Done!")
