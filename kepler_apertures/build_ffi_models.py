import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("%s/Work/BAERI/ADAP/kepler-apertures/" % os.environ["HOME"])
from kepler_apertures import KeplerFFI

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=None,
    help="Which quarter.",
)
parser.add_argument(
    "--fits-file",
    dest="fits_file",
    type=str,
    default=5,
    help="Name of the FFI fits file",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
parser.add_argument(
    "--plot",
    dest="plot",
    action="store_true",
    default=False,
    help="Make plots.",
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

mjd_done = [
    55307.51459974,
    55399.03668604,
    55492.78603754,
    55585.55556804,
    55677.42404264,
    55769.45696544,
    55864.77969804,
    55958.40644554,
    56047.49693394,
    56137.50692204,
    56236.81420744,
    56330.54311544,
    56390.47480454,
]


def run_code():
    psf = KeplerFFI(
        ffi_name=args.fits_file,
        channel=args.channel,
        plot=args.plot,
        save=args.save,
        quarter=args.quarter,
    )
    if psf.quarter in mjd_done:
        print("FFI channel epoch already done!")
        sys.exit()
    if args.plot:
        if not os.path.isdir("../data/figures/%s" % (str(args.quarter))):
            os.mkdir("../data/figures/%s" % (str(args.quarter)))

        ax = psf.plot_pixel_masks()
        ax = psf.plot_image(sources=False)

    psf._create_sparse()
    print("Computing PSF edges...")
    radius = psf._get_source_mask(
        upper_radius_limit=5,
        lower_radius_limit=1.1,
        flux_cut_off=50,
        dm_type="rf-quadratic",
    )
    print("Computing PSF model...")
    psf_model = psf._build_psf_model(
        rknots=12,
        phiknots=15,
        flux_cut_off=1,
    )
    print("Mean model sparse shape: ", psf.mean_model.shape)
    print("Doing PSF photometry...")
    psf.fit_model()
    psf.save_catalog()
    return


if __name__ == "__main__":
    if args.quarter is not None:
        print("Running PSF models for Q: %i Ch: %i" % (args.quarter, args.channel))
    else:
        print("Running PSF models for file: %s Ch: %i" % (args.fits_file, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    run_code()

    print("Done!")
