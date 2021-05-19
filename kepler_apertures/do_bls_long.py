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
    "--save",
    dest="save",
    action="store_true",
    default=False,
    help="Save models.",
)
parser.add_argument(
    "--test",
    dest="test",
    action="store_true",
    default=False,
    help="Test code in a small sample.",
)
parser.add_argument(
    "--batch",
    dest="batch",
    type=int,
    default=0,
    help="Run by bathc or all (0)",
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
    default=False,
    help="dry run.",
)
args = parser.parse_args()

quarters = np.arange(5, 18)
channels = np.arange(1, 85)
channels = np.delete(channels, [4, 5, 6, 7])

partitions = np.append(np.arange(0, 9327, 100), 9327)

apertures = ["O", 1, 2, 3, 4, 5, 6, 7]
names_remove = [
    "OUTPUT",
    "MODULE",
    "ROW",
    "COLUMN",
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
    "FILENAME",
]


# @profile
def give_me_full_lc(catalog, name):
    flc = []
    for q in quarters:
        ch = catalog.loc[name, (str(q), "channel")]
        if np.isfinite(ch):
            fname = "../data/export/%i/%02i/lc_q%i_ch%02i_%s.fits.gz" % (
                q,
                int(ch),
                q,
                int(ch),
                name.replace(" ", "_"),
            )
            lc = lk.KeplerLightCurve.read(fname)
            flc.append(lc)
        else:
            continue
    return lk.LightCurveCollection(flc)


spline_offset_dm_dict = {}
cbv_dm_dict = {}


# @profile
def correct_lc(lc, plot=False):
    # creating Design matrix
    times = lc.time.jd
    breaks = list(np.where(np.diff(times) > 0.3)[0] + 1)
    # spline DM
    if lc.quarter in spline_offset_dm_dict.keys():
        spline_dm = spline_offset_dm_dict[lc.quarter]["spline_dm"]
        offset_dm = spline_offset_dm_dict[lc.quarter]["offset_dm"]
    else:
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
        offset_dm = lk.DesignMatrix(
            np.ones_like(times), name="offset"
        )  # .split(breaks)
        offset_dm.prior_mu = np.ones(1)
        offset_dm.prior_sigma = np.ones(1) * 0.000001

        spline_offset_dm_dict[lc.quarter] = {
            "spline_dm": spline_dm,
            "offset_dm": offset_dm,
        }

    # DM with CBVs, first 4 only
    if "%i-%i" % (lc.quarter, lc.channel) in cbv_dm_dict.keys():
        cbv_dm = cbv_dm_dict["%i-%i" % (lc.quarter, lc.channel)]
    else:
        cbvs = lk.correctors.download_kepler_cbvs(
            mission="Kepler", quarter=lc.quarter, channel=lc.channel
        )
        basis = 4
        cbv_dm = cbvs[
            np.in1d(cbvs.cadenceno.value, lc.cadenceno.value)
        ].to_designmatrix()
        cbv_dm = lk.DesignMatrix(cbv_dm.X[:, :basis], name="cbv").split(breaks)
        cbv_dm.prior_mu = np.ones(basis * (len(breaks) + 1))
        cbv_dm.prior_sigma = np.ones(basis * (len(breaks) + 1)) * 100000

        cbv_dm_dict["%i-%i" % (lc.quarter, lc.channel)] = cbv_dm

    # collection of DMs
    dmc = lk.DesignMatrixCollection([offset_dm, cbv_dm, spline_dm])

    rc = lk.RegressionCorrector(lc)
    try:
        clc = rc.correct(dmc)
    except np.linalg.LinAlgError:
        clc = lc.copy()
    if plot:
        rc.diagnose()
        plt.show()
    return clc


# @profile
def stitch_lcs(flc, aperture="O"):
    new_flc, scales = [], []
    for i, lc in enumerate(flc):
        flx_complet = lc.meta["FLFRSAP%s" % (str(aperture))]
        scales.append(flx_complet)
        new_lc = lc.copy()

        for k in names_remove:
            del new_lc[0].meta[k]

        ext = aperture if aperture != "O" else ""
        new_lc.flux = lc["flux%s" % str(ext)]
        new_lc.flux_err = lc["flux_err%s" % str(ext)]
        for ap in range(1, 8):
            if ap == aperture:
                continue
            new_lc.remove_columns(["flux%s" % str(ap), "flux_err%s" % str(ap)])
        new_lc = correct_lc(new_lc, plot=False)
        new_flc.append(
            new_lc.remove_outliers(sigma_lower=1e10, sigma_upper=3) * flx_complet
        )
    return lk.LightCurveCollection(new_flc).stitch()


# @profile
def run_code():

    # load Gaia catalogs
    big_cat_path = "../data/catalogs/EXBA_catalog_all_sources.csv"
    if os.path.isfile(big_cat_path):
        print("Loading big catalog from disk...")
        print(big_cat_path)
        big_df = pd.read_csv(big_cat_path, index_col=0, header=[0, 1])
    else:
        big_cat = []
        for q in tqdm(quarters, desc="Quarters"):
            cat_aux = []
            for ch in channels:
                hdu = fits.open(
                    "../data/export/%i/%02i/image_q%i_ch%02i_gaiadr3.fits.gz"
                    % (q, ch, q, ch)
                )
                hdr = hdu[0].header
                img = hdu[1].data
                cat = Table(hdu[2].data).to_pandas()
                cat["quarter"] = q
                cat["channel"] = ch
                cat["index"] = cat.index
                cat_aux.append(cat)
            big_cat.append(pd.concat(cat_aux, axis=0).set_index("designation"))
        # concatenate catalogs keeping quarters as column key
        big_df = pd.concat(
            big_cat, axis=1, join="outer", keys=[str(q) for q in quarters]
        )
        if args.save:
            big_df.to_csv(big_cat_path)

    pmin, pmax, ffrac = 60, 360, 5000
    duration_in = [0.03, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.33, 0.4, 0.45]
    period_in = 1 / np.linspace(1 / pmin, 1 / pmax, 10000)

    long_lcs = []
    best_periods, periods_snr = [], []
    power, frequency = [], []
    duration, depth, transit_time = [], [], []
    if args.test:
        big_df = big_df.head(5)
    if args.batch != 0:
        big_df = big_df.iloc[partitions[args.batch - 1] : partitions[args.batch]]
    for id, row in tqdm(big_df.iterrows(), desc="BLS search: ", total=len(big_df)):
        # print(id)
        lcc = give_me_full_lc(big_df, id)
        # if True:
        #     continue
        llc = stitch_lcs(lcc, aperture="7")
        long_lcs.append(llc)
        # clc = clc.remove_outliers(sigma_lower=1e10, sigma_upper=5)
        periodogram = llc.to_periodogram(
            method="bls",
            period=period_in,
            duration=duration_in,
            frequency_factor=ffrac,
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

    # sys.exit()
    power = np.array(power)
    frequency = np.array(frequency)

    df = pd.DataFrame(
        np.array(
            [
                big_df.index.values,
                best_periods,
                periods_snr,
                duration,
                depth,
                transit_time,
            ]
        ).T,
        columns=[
            "designation",
            "BLS_period",
            "BLS_period_SNR",
            "BLS_duration",
            "BLS_depth",
            "BLS_transit_time",
        ],
    )

    if args.save:
        to_save = {
            "lc_corrected": long_lcs,
            "power": power,
            "frequency": frequency[0],
        }
        file_name = "../data/export/long_bls/bls_longP_results_%02i.pkl" % (args.batch)
        print(file_name)

        with open(file_name, "wb") as f:
            pickle.dump(to_save, f)

        df.to_csv("../data/export/long_bls/bls_longP_results_%02i.csv" % (args.batch))

    return


if __name__ == "__main__":
    print("Running long BLS")
    print("batch: ", args.batch)
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    run_code()

    print("Done!")
