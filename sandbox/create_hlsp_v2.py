import os, glob, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from tqdm.auto import tqdm
import argparse

import warnings

warnings.filterwarnings("ignore", category=u.UnitsWarning)
warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

parser = argparse.ArgumentParser(description="HLSP maker")
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
args = parser.parse_args()

project_name = "kbonus-apexba"
ap_names = [0, 1, 2, 3, 4, 5, 6, 7]
ap_percut = ["op", 0, 15, 30, 45, 60, 75, 90]

columns = [
    "TIME",
    "CADENCENO",
    "FLUX",
    "FLUX1",
    "FLUX2",
    "FLUX3",
    "FLUX4",
    "FLUX5",
    "FLUX6",
    "FLUX7",
    "FLUX_ERR",
    "FLUX_ERR1",
    "FLUX_ERR2",
    "FLUX_ERR3",
    "FLUX_ERR4",
    "FLUX_ERR5",
    "FLUX_ERR6",
    "FLUX_ERR7",
    "QUALITY",
    "SAP_QUALITY",
]


def run_code():
    q = args.quarter
    ch = args.channel
    in_dir = "../../EXBA_LCFs/data/q%02i/ch%02i" % (q, ch)
    files = np.sort(
        glob.glob("%s/hlsp_kBonus-ApEXBA_kepler_*-q%02i_v1.0_lc.fits.gz" % (in_dir, q))
    )
    for i, f in tqdm(enumerate(files), total=len(files), leave=True):
        hdu = fits.open(f)
        # update header cards
        designation = hdu[0].header["OBJECT"]

        hdu[0].header.set(
            "DOI",
            "10.17909/t9-d5wy-e535",
            "Digital Object Identifier for the HLSP data collection",
            before="ORIGIN",
        )
        hdu[0].header.set("FILTER", "KEPLER", "", after="INSTRUME")
        hdu[0].header.set("TIMESYS", "TDB", "Time scale", after="FILTER")

        new_table = Table(hdu[1].data)

        for col in columns:
            if col.startswith("FLUX"):
                new_table[col].unit = "e-/s"
            if col == "TIME":
                new_table[col] -= 2454833
                new_table[col].unit = "BJD - 2454833"

        hdu[1] = fits.BinTableHDU(new_table[columns], name="LIGHTCURVE")

        # aperture metrics
        for k, ap in enumerate(ap_names):
            hdu[0].header.set(
                "PERCUT%i" % ap,
                np.round(hdu[0].header["PERCUT%i" % ap], decimals=1),
                "Percentile cut to define aperture %i" % ap,
            )
            hdu[0].header.set(
                "FLFRSAP%i" % ap,
                np.round(hdu[0].header["FLFRSAP%i" % ap], decimals=5),
                "Flux completeness metric for aperture %i" % ap,
            )
            hdu[0].header.set(
                "CRWDSAP%i" % ap,
                np.round(hdu[0].header["CRWDSAP%i" % ap], decimals=5),
                "Flux crowding metric for aperture %i" % ap,
            )
            if ap == 0:
                del hdu[0].header["PERCUT%i" % ap]

        # new dir tree
        out_dir = "../../EXBA_LCFs/data_v2/q%02i/ch%02i" % (q, ch)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # new file name
        fname = "hlsp_kbonus-apexba_kepler_kepler_%s-q%02i_kepler_v1.0_lc.fits" % (
            designation.replace(" ", "-").lower(),
            q,
        )
        hdu.writeto("%s/%s" % (out_dir, fname), overwrite=True, checksum=True)
    return


if __name__ == "__main__":
    print("Quarter %i Channel %i" % (args.quarter, args.channel))
    if args.channel in [5, 6, 7, 8]:
        print("Channel with no data!")
        sys.exit()
    run_code()
    print("Done!")
