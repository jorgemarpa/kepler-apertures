import os
import sys
import glob
import argparse
import numpy as np
from astropy.io import fits
from astropy import units
import lightkurve as lk

sys.path.append("%s/Work/BAERI/ADAP/kepler-apertures/" % os.environ["HOME"])
from kepler_apertures import KeplerPSF, EXBAMachine

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
    "--gaia-dr",
    dest="gaia_dr",
    type=int,
    default=2,
    help="Gaia DR to be query",
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


def _make_aperture_extension(aperture_mask):
    """Returns an `ImageHDU` object containing the 'APERTURE' extension
    of a light curve file."""
    if aperture_mask is not None:
        hdu = fits.ImageHDU(aperture_mask.astype(np.uint8))
        hdu.header["EXTNAME"] = "APERTURE"
    return hdu


# @profile
def run_code():
    # load PRF models from FFI
    print("Loading PRF model...")
    kpsf = KeplerPSF(quarter=args.quarter, channel=args.channel)

    # create EXBA object
    print("Loading EXBA data...")
    exba = EXBAMachine(
        quarter=args.quarter,
        channel=args.channel,
        magnitude_limit=20,
        gaia_dr=args.gaia_dr,
    )
    print(exba)
    path = "../data/export/%i/%02i" % (args.quarter, args.channel)
    if not os.path.isdir(path):
        os.mkdir(path)
    if args.save:
        exba.image_to_fits(
            path="%s/kplr_EXBA_q%02i_ch%02i_img-cat.fits.gz"
            % (path, args.quarter, args.channel),
            overwrite=True,
        )

    # evaluate the PRF models in the EXBA sources
    print("Evaluating PRF...")
    psf_exba = kpsf.evaluate_PSF(exba.flux, exba.dx, exba.dy, exba.gaia_flux[:, 0])

    # do lcs for optimal aperture
    optimal_ap = []
    optimal_cmpt = []
    optimal_crwd = []
    optimal_perc = []
    print("Computing optimal Aperture...")
    for idx, s in exba.sources.iterrows():

        mask, complet, crowd, optim_p = kpsf.optimize_aperture(
            psf_exba, idx=idx, target_complet=0.5, target_crowd=1.0
        )
        optimal_ap.append(mask)
        optimal_cmpt.append(complet)
        optimal_crwd.append(crowd)
        optimal_perc.append(optim_p)

    optimal_ap = np.array(optimal_ap)
    exba.do_photometry(optimal_ap)
    optim_flux = exba.sap_flux
    optim_flux_err = exba.sap_flux_err
    optim_apmask = exba.aperture_mask_2d

    # do lcs for different apertures
    print("Computing multiple Apertures...")
    fluxs, flux_errs, aperture_masks = [], [], []
    metricCmplt, metricCrowd = [], []
    percentiles = np.array([0, 15, 30, 45, 60, 75, 90])
    for i, p in enumerate(percentiles):

        ap_mask, cmplt, crwd = kpsf.create_aperture_mask(psf_exba, p)
        exba.FLFRCSAP = cmplt
        exba.CROWDSAP = crwd

        exba.do_photometry(ap_mask)
        aperture_masks.append(exba.aperture_mask_2d)
        fluxs.append(exba.sap_flux)
        flux_errs.append(exba.sap_flux_err)
        metricCmplt.append(exba.FLFRCSAP)
        metricCrowd.append(exba.CROWDSAP)

    metricCmplt = np.array(metricCmplt)
    metricCrowd = np.array(metricCrowd)

    all_hdus = []
    for idx, s in exba.sources.iterrows():
        tile = int((s.col - exba.tpfs[0].column) / 9)
        meta = {
            "ORIGIN": "ApertureMACHINE",
            "VERSION": exba.__version__,
            "APERTURE": "Aperture",
            "LABEL": s.designation,
            "OBJECT": s.designation,
            "TARGETID": int(s.designation.split(" ")[-1]),
            "MISSION": "Kepler",
            "TELESCOP": "Kepler",
            "INSTRUME": "Kepler Photometer",
            "OBSMODE": "long cadence",
            "SEASON": exba.tpfs[tile].get_header()["SEASON"],
            "EQUINOX": 2000,
            "RA_OBJ": s.ra,
            "DEC_OBJ": s.dec,
            "PMRA": s.pmra / 1000 if np.isfinite(s.pmra) else None,
            "PMDEC": s.pmdec / 1000 if np.isfinite(s.pmdec) else None,
            "PARALLAX": s.parallax if np.isfinite(s.parallax) else None,
            "GMAG": s.phot_g_mean_mag if np.isfinite(s.phot_g_mean_mag) else None,
            "RPMAG": s.phot_rp_mean_mag if np.isfinite(s.phot_rp_mean_mag) else None,
            "BPMAG": s.phot_bp_mean_mag if np.isfinite(s.phot_bp_mean_mag) else None,
            "CHANNEL": exba.channel,
            "MODULE": exba.hdr["MODULE"],
            "OUTPUT": exba.hdr["OUTPUT"],
            "QUARTER": exba.quarter,
            "CAMPAIGN": "EXBA",
            "TPF_ORGN": exba.tpfs_files[tile].split("/")[-1],
            "KID_ORGN": exba.tpfs[tile].get_header()["OBJECT"],
            "ROW": np.round(s.row, decimals=4),
            "COLUMN": np.round(s.col, decimals=4),
            "FLFRSAPO": np.round(optimal_cmpt[idx], decimals=4),
            "CRWDSAPO": np.round(optimal_crwd[idx], decimals=4),
            "PERCUTO": np.round(optimal_perc[idx], decimals=1),
        }
        optim_hdu = _make_aperture_extension(optim_apmask[idx])
        optim_hdu.header["EXTNAME"] = "APERTURE_OPTIMAL"
        aperture_masks_hdu = [optim_hdu]

        quality_arr = exba.tpfs[tile].quality[
            np.in1d(exba.tpfs[tile].cadenceno, exba.cadences)
        ]
        lc_dct = {
            "cadenceno": exba.cadences.astype(np.int32),
            "time": exba.time.astype(np.float64) * units.d,
            "flux": optim_flux[idx].astype(np.float32)
            * (units.electron / units.second),
            "flux_err": optim_flux_err[idx].astype(np.float32)
            * (units.electron / units.second),
            "quality": quality_arr,
        }
        for ip in range(len(fluxs)):
            meta["FLFRSAP%i" % (ip + 1)] = np.round(metricCmplt[ip][idx], decimals=4)
            meta["CRWDSAP%i" % (ip + 1)] = np.round(metricCrowd[ip][idx], decimals=4)
            meta["PERCUT%i" % (ip + 1)] = percentiles[ip]

            lc_dct["flux%i" % (ip + 1)] = fluxs[ip][idx].astype(np.float32) * (
                units.electron / units.second
            )
            lc_dct["flux_err%i" % (ip + 1)] = flux_errs[ip][idx].astype(np.float32) * (
                units.electron / units.second
            )

            ap_hdu = _make_aperture_extension(aperture_masks[ip][idx])
            ap_hdu.header["EXTNAME"] += "%i" % (ip + 1)
            aperture_masks_hdu.append(ap_hdu)

        idx_best = np.argmax(metricCrowd[:, 0])
        lc = lk.LightCurve(lc_dct, meta=meta)

        del lc_dct["time"], lc_dct["flux_err"]
        hdul = lc.to_fits(**dict(sorted(lc_dct.items())), **lc.meta)
        for i in range(4, 19):
            hdul[1].header["TUNIT%i" % (i)] = "e-/s    "
        hdul.extend(aperture_masks_hdu)

        all_hdus.append(hdul)

    if args.save:
        for i, hdu in enumerate(all_hdus):
            hlsp_name_conve = "hlsp_exba_kepler_%s-q%02i_v%s_lc.fits.gz" % (
                exba.sources.designation[i].replace(" ", "-"),
                args.quarter,
                "1.0",
            )
            kplr_name_conve = "kplr_EXBA_q%02i_ch%02i_%s_llc.fits.gz" % (
                self.quarter,
                self.channel,
                exba.sources.designation[i].replace(" ", "-"),
            )
            hdu.writeto("%s/%s" % (path, kplr_name_conve), overwrite=True)

        # # create lcs from apertures
        # exba.create_lcs(exba.aperture_mask)

        # # apply LK flatten
        # exba.apply_flatten()
        #
        # # apply CBV corrector to all LCs
        # exba.apply_CBV(plot=False)
        #
        # # do BLS search
        # exba.do_bls_search(test_lcs=None, n_boots=0, plot=False)

        # store EXBA object

        # print("Saving EXBA object...")
        # exba.store_data()

    return


if __name__ == "__main__":
    print("Computing LCs for Q: %i Ch: %i" % (args.quarter, args.channel))
    if args.dry_run:
        print("Dry run mode, exiting...")
        sys.exit()
    if args.channel in [5, 6, 7, 8]:
        print("Channles with no data, exiting.")
        sys.exit()
    run_code()

    print("Done!")
