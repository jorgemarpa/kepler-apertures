import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('%s/Work/BAERI/ADAP/kepler-apertures/' % os.environ['HOME'])
from kepler_apertures import KeplerFFI

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=5,
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
    "--dm-type",
    dest="dm_type",
    type=str,
    default="cubic",
    help="Type of basis for desing matrix",
)
parser.add_argument(
    "--plot",
    dest="plot",
    action="store_true",
    default=True,
    help="Make plots.",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    default=True,
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

remove_sat = True
mask_bright = True


def run_code():
    psf = KeplerFFI(
        quarter=args.quarter, channel=args.channel, plot=args.plot, save=args.save
        )
    if args.plot:
        if not os.path.isdir("../data/figures/%s" % (str(args.quarter))):
            os.mkdir("../data/figures/%s" % (str(args.quarter)))

        ax = psf.plot_pixel_masks()
        ax = psf.plot_image(sources=False)

    psf._create_sparse()
    print("Computing PSF edges...")
    radius = psf._find_psf_edge(psf.r, psf.dflux, psf.gf,
                                radius_limit=6.0,
                                cut=200, dm_type="cubic")
    print("Computing PSF model...")
    psf_model = psf._build_psf_model(psf.r, psf.phi, psf.dflux,
                                     psf.gf, radius * 2.,
                                     psf.dx, psf.dy,
                                     rknots=5,
                                     phiknots=15)
    return


if __name__ == "__main__":
    print("Running PSF models for Q: %i Ch: %i" % (args.quarter, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    run_code()

    print("Done!")
