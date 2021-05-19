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
        "../data/export/%i/%02i/image_q%i_ch%02i_gaiadr3.fits.gz"
        % (args.quarter, args.channel, args.quarter, args.channel)
    )
    catalog = Table(hdu[2].data).to_pandas()

    # load LCs
    lc_files = np.sort(
        glob.glob(
            "../data/export/%i/%02i/lc_q%i_ch%02i_Gaia_EDR3_*.fits.gz"
            % (args.quarter, args.channel, args.quarter, args.channel)
        )
    )
    apertures = ["O", 1, 2, 3, 4, 5, 6, 7]
    remove_cols = [
        "flux1",
        "flux_err1",
        "flux2",
        "flux_err2",
        "flux3",
        "flux_err3",
        "flux4",
        "flux_err4",
        "flux5",
        "flux_err5",
        "flux6",
        "flux_err6",
        "flux7",
        "flux_err7",
    ]
    remove_keys = [
        "FLFRSAPO",
        "CRWDSAPO",
        "FLFRSAP1",
        "CRWDSAP1",
        "FLFRSAP2",
        "CRWDSAP2",
        "FLFRSAP3",
        "CRWDSAP3",
        "FLFRSAP4",
        "CRWDSAP4",
        "FLFRSAP5",
        "CRWDSAP5",
        "FLFRSAP6",
        "CRWDSAP6",
        "FLFRSAP7",
        "CRWDSAP7",
    ]

    lcs, metricCmpl, metricCrwd, aper_use = [], [], [], []
    for f in tqdm(lc_files, desc="Loading LCs"):
        meta = fits.open(f)[0].header
        cmpl = np.array([meta["FLFRSAP%s" % str(k)] for k in apertures])
        crwd = np.array([meta["CRWDSAP%s" % str(k)] for k in apertures])
        winner = np.where(crwd == np.amax(crwd))[0]
        winner = winner[np.argmax(cmpl[winner])]
        aper_use.append(apertures[winner])
        metricCmpl.append(cmpl[winner])
        metricCrwd.append(crwd[winner])
        lc = lk.KeplerLightCurve.read(f)
        lc.label = meta["LABEL"]
        winner = apertures[winner] if winner != 0 else ""
        lc.flux = lc["flux%s" % str(winner)]
        lc.flux_err = lc["flux_err%s" % str(winner)]
        lc.remove_columns(remove_cols)
        for k in remove_keys:
            lc.meta.pop(k, None)
        lcs.append(lc)

    df = pd.DataFrame(
        np.array([catalog.designation, metricCmpl, metricCrwd, aper_use]).T,
        columns=["designation", "FLFRSAP", "CRWDSAP", "Aperture"],
    )

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
    best_periods, periods_snr = [], []
    power = []
    frequency = []
    duration, depth, transit_time = [], [], []
    for i, lc in tqdm(enumerate(lcs), desc="BLS search: ", total=len(lcs)):
        rc = lk.RegressionCorrector(lc)
        try:
            clc = rc.correct(dmc, sigma=3)
        except np.linalg.LinAlgError:
            clc = lc.copy()
        corrected_lcs.append(clc)

        # clc = clc.remove_outliers(sigma_lower=1e10, sigma_upper=5)
        periodogram = clc.normalize().to_periodogram(
            method="bls",
            period=period_in,
            duration=duration_in,
        )
        best_fit_period = periodogram.period_at_max_power.value
        power_amax = np.argmax(periodogram.power)
        power_snr = periodogram.snr[power_amax].value
        dur = periodogram.duration_at_max_power.value
        dep = periodogram.depth_at_max_power.value
        ttime = periodogram.transit_time_at_max_power.value

        best_periods.append(best_fit_period)
        periods_snr.append(power_snr)
        power.append(periodogram.power)
        frequency.append(periodogram.frequency)
        duration.append(dur)
        depth.append(dep)
        transit_time.append(ttime)

    power = np.array(power)
    frequency = np.array(frequency)

    df["BLS_period"] = np.array(best_periods)
    df["BLS_period_SNR"] = np.array(periods_snr)
    df["BLS_duration"] = np.array(duration)
    df["BLS_depth"] = np.array(depth)
    df["BLS_transit_time"] = np.array(transit_time)

    if args.save:
        to_save = {
            "lc_corrected": corrected_lcs,
            "power": power,
            "table": df,
            "frequency": frequency[0],
        }
        file_name = "%s/bls_results_q%i_ch%02i.pkl" % (
            os.path.dirname(lc.meta["FILENAME"]),
            args.quarter,
            args.channel,
        )
        print(file_name)

        with open(file_name, "wb") as f:
            pickle.dump(to_save, f)

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
