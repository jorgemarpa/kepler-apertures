import os
import glob
import warnings
import datetime
import wget

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from tqdm.auto import tqdm

from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.time import Time
from astropy import units
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

from .utils import get_gaia_sources
from . import PACKAGEDIR, DATAOUTDIR
from .version import __version__


class EXBAMachine(object):
    """
    Class that works with Kepler's EXBA data, to identify observed sources using Gaia
    catalogs, and create light curves from simple aperture photometry.
    """

    def __init__(self, channel=53, quarter=5, magnitude_limit=20, gaia_dr=3):
        """
        Parameters
        ----------
        channel : int
            Channel number of the EXBA image.
        quarter : int
            Quarter number of the EXBA image.
        magnitude_limit : float
            Limiting magnitude in g band used when querying Gaia catalogs,
            default is 20 mag.
        gaia_dr : int
            Gaia data release, dafult is EDR3.

        Attributes
        ----------
        quarter : int
            Channel number of the EXBA image.
        channel : int
            Quarter number of the EXBA image.
        gaia_dr : int
            Gaia data release, dafult is EDR3.
        tpfs : lightkurve.TargetPixelFileCollection
            Collection of 4 TPFs that form the full EXBA mask.
        time : numpy.ndarray
            Data array containing the time values.
        cadences : numpy.ndarray
            Data array containing the cadence numbers.
        row : numpy.ndarray
            Data array containing the valid pixel row numbers. Has shape of [n_pixels].
        column : numpy.ndarray
            Data array containing the valid pixel columns numbers.
            Has shape of [n_pixels].
        flux : numpy.ndarray
            Data array containing the valid image fluxes. Has shape of
            [n_times, n_pixels].
        flux_err : numpy.ndarray
            Data array containing the valid image flux errors. Has shape of
            [n_times, n_pixels].
        ra : numpy.ndarray
            Data array containing the valid RA pixel values. Has shape of [n_pixels].
        dec : numpy.ndarray
            Data array containing the valid Dec pixel values. Has shape of [n_pixels].
        dx : numpy.ndarray
            Distance between pixel and source coordinates, units of pixels. Has shape
            of [n_sources, n_pixels]
        dy : numpy.ndarray
            Distance between pixel and source coordinates, units of pixels. Has shape
            of [n_sources, n_pixels]
        r : numpy.ndarray
            Radial distance between pixel and source coordinates (polar coordinates),
            in units of pixels.
        phi : numpy.ndarray
            Angle between pixel and source coordinates (polar coordinates),
            in units of radians
        n_sources : int
            Number of sources in Gaia catalog observed in the EXBA mask.
        n_rows : int
            Number rows in the EXBA image.
        n_columns : int
            Number columns in the EXBA image.
        aperture_mask : numpy.ndarray
            Data array with the source aperture masks. Has shape of
            [n_sources, n_pixels]
        FLFRCSAP : numpy.array
            Data array with the completeness metric for every source computed from
            the photometric aperture.
        CROWDSAP : numpy.array
            Data array with the contamination metric for every source computed from
            the photometric aperture.
        """
        self.quarter = quarter
        self.channel = channel
        self.gaia_dr = gaia_dr

        # load local TPFs files
        tpfs_paths = np.sort(
            glob.glob(
                "%s/data/fits/exba/q%i/ch%02i/*_lpd-targ.fits.gz"
                % (DATAOUTDIR, quarter, channel)
            )
        )
        if len(tpfs_paths) == 0:
            print("Downloading TPFs for EXBA mask...")
            self.download_exba(channel=channel, quarter=quarter)
            tpfs_paths = np.sort(
                glob.glob(
                    "%s/data/fits/exba/q%i/ch%02i/*_lpd-targ.fits.gz"
                    % (DATAOUTDIR, quarter, channel)
                )
            )

        self.tpfs_files = tpfs_paths

        tpfs = lk.TargetPixelFileCollection(
            [lk.KeplerTargetPixelFile(f) for f in tpfs_paths]
        )
        self.tpfs = tpfs
        self.wcs = tpfs[0].wcs
        print(self.tpfs)
        # check for same channels and quarter
        channels = [tpf.get_header()["CHANNEL"] for tpf in tpfs]
        quarters = [tpf.get_header()["QUARTER"] for tpf in tpfs]
        self.hdr = tpfs[0].get_header()

        if len(set(channels)) != 1 and list(set(channels)) != [channel]:
            raise ValueError(
                "All TPFs must be from the same channel %s"
                % ",".join([str(k) for k in channels])
            )

        if len(set(quarters)) != 1 and list(set(quarters)) != [quarter]:
            raise ValueError(
                "All TPFs must be from the same quarter %s"
                % ",".join([str(k) for k in quarters])
            )

        # stich channel's strips and parse TPFs
        time, cadences, row, col, flux, flux_err, unw = self._parse_TPFs_channel(tpfs)
        self.time, self.cadences, flux, flux_err = self._preprocess(
            time, cadences, flux, flux_err
        )
        self.row_2d, self.column_2d, self.flux_2d, self.flux_err_2d = (
            row.copy(),
            col.copy(),
            flux.copy(),
            flux_err.copy(),
        )
        self.row, self.column, self.flux, self.flux_err, self.unw = (
            row.ravel(),
            col.ravel(),
            flux.reshape(flux.shape[0], np.product(flux.shape[1:])),
            flux_err.reshape(flux_err.shape[0], np.product(flux_err.shape[1:])),
            unw.ravel(),
        )
        self.ra, self.dec = self._convert_to_wcs(tpfs, self.row, self.column)

        # search Gaia sources in the sky
        sources = self._do_query(
            self.ra,
            self.dec,
            epoch=self.time[0],
            magnitude_limit=magnitude_limit,
            load=True,
        )
        sources["col"], sources["row"] = self.wcs.wcs_world2pix(
            sources.ra, sources.dec, 0.0
        )
        sources["col"] += tpfs[0].column
        sources["row"] += tpfs[0].row
        self.sources, self.bad_sources = self._clean_source_list(
            sources, self.ra, self.dec
        )

        self.dx, self.dy = np.asarray(
            [
                np.vstack(
                    [
                        self.column - self.sources["col"][idx],
                        self.row - self.sources["row"][idx],
                    ]
                )
                for idx in range(len(self.sources))
            ]
        ).transpose([1, 0, 2])

        self.r = np.hypot(self.dx, self.dy)
        self.phi = np.arctan2(self.dy, self.dx)

        self.n_sources = self.sources.shape[0]
        self.n_rows = self.flux_2d.shape[1]
        self.n_columns = self.flux_2d.shape[2]

        self.aperture_mask = np.zeros_like(self.dx).astype(bool)
        self.FLFRCSAP = np.zeros(self.sources.shape[0])
        self.CROWDSAP = np.zeros(self.sources.shape[0])
        self.cut = np.zeros(self.sources.shape[0])

    def __repr__(self):
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return "EXBA Patch:\n\t Channel %i, Quarter %s, Gaia DR%i sources %i" % (
            self.channel,
            q_result,
            self.gaia_dr,
            len(self.sources),
        )

    @staticmethod
    def download_exba(channel=1, quarter=5):
        """
        Download EXBA fits file to a dedicated quarter/channel directory
        It uses a exba_tpfs_info.csv to map the quarter/channel to the corresponding
        file names in MAST archive.

        Parameters
        ----------
        channel : int
            Number of channel to be download, valid numbers are bwtween 1 and 84.
        quarter : int
            Number of quarter to be download, valid numbers are bwtween 1 and 17.
        """
        url = "https://archive.stsci.edu/missions/kepler/target_pixel_files/1000"
        map = pd.read_csv("%s/data/exba_tpfs_info.csv" % (PACKAGEDIR), index_col=0)
        file_names = map.query("channel == %i and quarter == %i" % (channel, quarter))

        if not os.path.isdir(
            "%s/data/fits/exba/q%i/ch%02i" % (DATAOUTDIR, quarter, channel)
        ):
            os.makedirs("%s/data/fits/exba/q%i/ch%02i" % (DATAOUTDIR, quarter, channel))

        for i, row in file_names.iterrows():
            name = row["file_name"]
            kid = row["kepler_id"].split(" ")[-1]
            out = "%s/data/fits/exba/q%i/ch%02i/%s" % (
                DATAOUTDIR,
                quarter,
                channel,
                name,
            )
            print("%s/%s/%s" % (url, kid, name))
            wget.download("%s/%s/%s" % (url, kid, name), out=out)

        return

    def _parse_TPFs_channel(self, tpfs):
        """
        Function to parse the TPFs containing the EXBA masks (4 per channel) and
        tile them.

        Parameters
        ----------
        tpfs : list of TPFs or TargetPixelFileCollection
            A list of TPFs that contain the 4 EXBA mask per channel.

        Returns
        -------
        times : numpy.ndarray
            Data array containing the time values.
        cadences : numpy.ndarray
            Data array containing the cadence numbers.
        row : numpy.ndarray
            Data array containing the pixel row numbers.
        col : numpy.ndarray
            Data array containing the pixel column numbers.
        flux : numpy.ndarray
            Data array containing the image flux.
        flux_err : numpy.ndarray
            Data array containing the image flux errors.
        """

        cadences = np.array([tpf.cadenceno for tpf in tpfs])
        # check if all TPFs has same cadences
        if not np.all(cadences[1:, :] - cadences[-1:, :] == 0):
            raise ValueError("All TPFs must have same time basis")

        # make sure tpfs are sorted by colum direction
        tpfs = lk.TargetPixelFileCollection(
            [tpfs[i] for i in np.argsort([tpf.column for tpf in tpfs])]
        )

        # extract times
        times = tpfs[0].time.jd

        # extract row,column mesh grid
        col, row = np.hstack(
            [
                np.mgrid[
                    tpf.column : tpf.column + tpf.shape[2],
                    tpf.row : tpf.row + tpf.shape[1],
                ]
                for tpf in tpfs
            ]
        )

        # extract flux vales
        flux = np.hstack([tpf.flux.transpose(1, 2, 0) for tpf in tpfs]).transpose(
            2, 0, 1
        )
        flux_err = np.hstack(
            [tpf.flux_err.transpose(1, 2, 0) for tpf in tpfs]
        ).transpose(2, 0, 1)

        # bookkeeping of tpf-pixel
        unw = np.hstack(
            [np.ones(tpf.shape[1:], dtype=np.int) * i for i, tpf in enumerate(tpfs)]
        )

        return times, cadences[0], row.T, col.T, flux, flux_err, unw

    def _preprocess(self, times, cadences, flux, flux_err):
        """
        Function to clean pixels with nan values and bad cadences. It Returns the same
        input arrays but cleaned.

        Parameters
        ----------
        times : numpy.ndarray
            Data array with the time values.
        cadences : numpy.ndarray
            Data array with the cadence numbers.
        flux : numpy.ndarray
            Data array with the image flux.
        flux_err : numpy.ndarray
            Data array with the image flux errors.

        Returns
        -------
        times : numpy.ndarray
            Data array with the time values.
        cadences : numpy.ndarray
            Data array with the cadence numbers.
        flux : numpy.ndarray
            Data array with the image flux.
        flux_err : numpy.ndarray
            Data array with the image flux errors.
        """
        # Remove cadences with nan flux
        nan_cadences = np.array([np.isnan(im).sum() == 0 for im in flux])
        times = times[nan_cadences]
        cadences = cadences[nan_cadences]
        flux = flux[nan_cadences]
        flux_err = flux_err[nan_cadences]

        return times, cadences, flux, flux_err

    def _convert_to_wcs(self, tpfs, row, col):
        """
        Function to convert pixel number to RA and Dec values using the WCS solution
        embedded in the TPFs.

        Parameters
        ----------
        tpfs : list of TPFs or TargetPixelFileCollection
            A list of TPFs that contain the EXBA tiles.
        row : numpy.ndarray
            Data aray with the row pixel values to be converted to RA & Dec.
        col : numpy.ndarray
            Data aray with the column pixel values to be converted to RA & Dec.

        Returns
        -------
        ra : numpy.ndarray
            Right Ascension coordinate obtained from the WCS solution.
        dec : numpy.ndarray
            Declination coordinate obtained from the WCS solution.
        """
        ra, dec = self.wcs.wcs_pix2world(
            (col - tpfs[0].column), (row - tpfs[0].row), 0.0
        )

        return ra, dec

    def _do_query(self, ra, dec, epoch=2020, magnitude_limit=20, load=True):
        """
        Calculate ra, dec coordinates and search radius to query Gaia catalog.

        Parameters
        ----------
        ra : numpy.ndarray
            Right ascension coordinate of pixels to do Gaia search
        dec : numpy.ndarray
            Declination coordinate of pixels to do Gaia search
        epoch : float
            Epoch of obervation in Julian Days of ra, dec coordinates,
            will be used to propagate proper motions in Gaia.
        magnitude_limit : int
            Limiting magnitued for query
        load : boolean
            Load or not the saved query. Set to False if want to force to run new
            queries.

        Returns
        -------
        sources : pandas.DataFrame
            Catalog with query result
        """
        columns = [
            "designation",
            "ra",
            "ra_error",
            "dec",
            "dec_error",
            "pmra",
            "pmdec",
            "parallax",
            "parallax_error",
            "phot_g_n_obs",
            "phot_g_mean_flux",
            "phot_g_mean_flux_error",
            "phot_g_mean_mag",
            "phot_bp_n_obs",
            "phot_bp_mean_flux",
            "phot_bp_mean_flux_error",
            "phot_bp_mean_mag",
            "phot_rp_n_obs",
            "phot_rp_mean_flux",
            "phot_rp_mean_flux_error",
            "phot_rp_mean_mag",
        ]
        file_name = "%s/data/catalogs/exba/%i/channel_%02i_gaiadr%s_xmatch.csv" % (
            DATAOUTDIR,
            self.quarter,
            self.channel,
            str(self.gaia_dr),
        )
        if os.path.isfile(file_name) and load:
            print("Loading query from file...")
            print(file_name)
            sources = pd.read_csv(file_name)
            sources = sources.loc[:, columns]

        else:
            # find the max circle per TPF that contain all pixel data to query Gaia
            ra_q = ra.mean()
            dec_q = dec.mean()
            rad_q = np.hypot(ra - ra_q, dec - dec_q).max() + 10 / 3600
            # query Gaia with epoch propagation
            sources = get_gaia_sources(
                tuple([ra_q]),
                tuple([dec_q]),
                tuple([rad_q]),
                magnitude_limit=magnitude_limit,
                epoch=Time(epoch, format="jd").jyear,
                dr=self.gaia_dr,
            )
            sources = sources.loc[:, columns]
            if not os.path.isdir(
                "%s/data/catalogs/exba/%i" % (DATAOUTDIR, self.quarter)
            ):
                os.makedirs("%s/data/catalogs/exba/%i" % (DATAOUTDIR, self.quarter))
            sources.to_csv(file_name)
        return sources

    def _clean_source_list(self, sources, ra, dec):
        """
        Function to clean surces from the catalog removing sources outside the image
        coverage (allowing for sources up to 4" outside the mask), and to remove
        blended sources (within 2").

        Parameters
        ----------
        sources : pandas.DataFrame
            Catalog with sources to be removed
        ra : numpy.ndarray
            Data array with values of RA for every pixel in the image.
        dec : numpy.ndarray
            Data array with values of Dec for every pixel in the image.

        Returns
        -------
        sources : pandas.DataFrame
            Clean catalog
        """
        # find sources on the image
        inside = (
            (sources.row > self.row.min() - 1.0)
            & (sources.row < self.row.max() + 1.0)
            & (sources.col > self.column.min() - 1.0)
            & (sources.col < self.column.max() + 1.0)
        )

        # find well separated sources
        s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
        midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]
        # remove sources closer than 4" = 1 pix
        closest = mdist.arcsec < 2.0
        blocs = np.vstack([midx[closest], np.where(closest)[0]])
        bmags = np.vstack(
            [
                sources.phot_g_mean_mag[midx[closest]],
                sources.phot_g_mean_mag[np.where(closest)[0]],
            ]
        )
        faintest = [blocs[idx][s] for s, idx in enumerate(np.argmax(bmags, axis=0))]
        unresolved = np.in1d(np.arange(len(sources)), faintest)
        del s_coords, midx, mdist, closest, blocs, bmags

        # Keep track of sources that we removed
        sources.loc[:, "clean_flag"] = 0
        sources.loc[~inside, "clean_flag"] += 2 ** 0  # outside TPF
        sources.loc[unresolved, "clean_flag"] += 2 ** 1  # close contaminant

        # combine 2 source masks
        clean = sources.clean_flag == 0
        removed_sources = sources[~clean].reset_index(drop=True)
        sources = sources[clean].reset_index(drop=True)

        return sources, removed_sources

    def do_photometry(self, aperture_mask):
        """
        Function to do aperture photometry on a set of sources. It creates/update class
        attributes that contains the SAP flux, errors, and aperture masks.

        Parameters
        ----------
        aperture_mask : numpy.ndarray
            Boolean mask of shape [n_sources, n_pixels] that has the aperture mask
            to be used to compute photometry for a set of sources.
        """
        sap = np.zeros((self.sources.shape[0], self.flux.shape[0]))
        sap_e = np.zeros((self.sources.shape[0], self.flux.shape[0]))

        for sidx in tqdm(range(len(aperture_mask)), desc="SAP", leave=True):
            sap[sidx, :] = self.flux[:, aperture_mask[sidx]].sum(axis=1)
            sap_e[sidx, :] = (
                np.power(self.flux_err[:, aperture_mask[sidx]].value, 2).sum(axis=1)
                ** 0.5
            )

        self.sap_flux = sap
        self.sap_flux_err = sap_e
        self.aperture_mask = aperture_mask
        self.aperture_mask_2d = aperture_mask.reshape(
            self.n_sources, self.n_rows, self.n_columns
        )

        return

    def create_lcs(self, aperture_mask):
        """
        Funciton to create `lightkurve.LightCurve` with the light curves using aperture
        photometry. It creates a class attribute `self.lcs` that is a
        `lk.LightCurveCollection` with the light curves of all input sources.

        Parameters
        ----------
        aperture_mask : numpy.ndarray
            Boolean mask of shape [n_sources, n_pixels] that has the aperture mask
            to be used to compute photometry for a set of sources.
        """
        self.do_photometry(aperture_mask)
        lcs = []
        for idx, s in self.sources.iterrows():
            tile = int((s.col - self.tpfs[0].column) / 9)
            meta = {
                "ORIGIN": "EXBAMachine",
                # "APERTURE_MASK": self.aperture_mask_2d[idx],
                "VERSION": __version__,
                "LABEL": s.designation,
                "TARGETID": int(s.designation.split(" ")[-1]),
                "MISSION": "Kepler",
                "INSTRUME": "Kepler Photometer",
                "OBSMODE": "long cadence",
                "SEASON": self.tpfs[tile].get_header()["SEASON"],
                "EQUINOX": 2000,
                "RA": s.ra,
                "DEC": s.dec,
                "PMRA": s.pmra / 1000 if np.isfinite(s.pmra) else None,
                "PMDEC": s.pmdec / 1000 if np.isfinite(s.pmdec) else None,
                "PARALLAX": s.parallax if np.isfinite(s.parallax) else None,
                "GMAG": s.phot_g_mean_mag if np.isfinite(s.phot_g_mean_mag) else None,
                "RPMAG": s.phot_rp_mean_mag
                if np.isfinite(s.phot_rp_mean_mag)
                else None,
                "BPMAG": s.phot_bp_mean_mag
                if np.isfinite(s.phot_bp_mean_mag)
                else None,
                "CHANNEL": self.channel,
                "MODULE": self.hdr["MODULE"],
                "OUTPUT": self.hdr["OUTPUT"],
                "QUARTER": self.quarter,
                "CAMPAIGN": "EXBA",
                "ROW": np.round(s.row, decimals=4),
                "COLUMN": np.round(s.col, decimals=4),
                "FLFRCSAP": np.round(self.FLFRCSAP[idx], decimals=6),
                "CROWDSAP": np.round(self.CROWDSAP[idx], decimals=6),
                "PERCENT": self.cut[idx],
            }
            lc = lk.LightCurve(
                time=self.time * units.d,
                flux=self.sap_flux[idx] * (units.electron / units.second),
                flux_err=self.sap_flux_err[idx] * (units.electron / units.second),
                meta=meta,
                # time_format="jd",
                # flux_unit="electron/s",
                cadenceno=self.cadences,
            )
            lcs.append(lc)
        self.lcs = lk.LightCurveCollection(lcs)
        return

    def apply_CBV(self, do_under=False, plot=True):
        """
        Applies CBV corrections to all the light curves in `self.lcs`. It optimizes
        the alpha parameter for each correction, if optimization fails, uses the alpha
        value calculated for previous light curve.
        It creates class attributes to access the CBV-corrected light curves, and
        under/over fitting metrics.

        Parameters
        ----------
        do_under : boolean
            Compute or not the under-fitting metric for the CBV correction.
        plot : boolean
            Plot or not CBVcorrector diagnostic figures.
        """
        if True:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=lk.LightkurveWarning)

        # Select which CBVs to use in the correction
        cbv_type = ["SingleScale"]
        # Select which CBV indices to use
        # Use the first 8 SingleScale and all Spike CBVS
        cbv_indices = [np.arange(1, 9)]

        over_fit_m = []
        under_fit_m = []
        corrected_lcs = []
        alpha = 1e-1
        self.alpha = np.zeros(len(self.lcs))

        # what if I optimize alpha for the first lc, then use that one for the rest?
        for i in tqdm(range(len(self.lcs)), desc="Applying CBVs to LCs", leave=True):
            lc = self.lcs[i][self.lcs[i].flux_err > 0].remove_outliers(
                sigma_upper=5, sigma_lower=1e20
            )
            cbvcor = lk.correctors.CBVCorrector(lc, interpolate_cbvs=False)
            if i % 1 == 0:
                print("Optimizing alpha")
                try:
                    cbvcor.correct(
                        cbv_type=cbv_type,
                        cbv_indices=cbv_indices,
                        alpha_bounds=[1e-2, 1e2],
                        target_over_score=0.9,
                        target_under_score=0.8,
                    )
                    alpha = cbvcor.alpha
                    if plot:
                        cbvcor.diagnose()
                        cbvcor.goodness_metric_scan_plot(
                            cbv_type=cbv_type, cbv_indices=cbv_indices
                        )
                        plt.show()
                except (ValueError, TimeoutError):
                    print(
                        "Alpha optimization failed, using previous value %.4f" % alpha
                    )
            self.alpha[i] = alpha
            cbvcor.correct_gaussian_prior(
                cbv_type=cbv_type, cbv_indices=cbv_indices, alpha=alpha
            )
            over_fit_m.append(cbvcor.over_fitting_metric())
            if do_under:
                under_fit_m.append(cbvcor.under_fitting_metric())
            corrected_lcs.append(cbvcor.corrected_lc)

        self.corrected_lcs = lk.LightCurveCollection(corrected_lcs)
        self.over_fitting_metrics = np.array(over_fit_m)
        if do_under:
            self.under_fitting_metrics = np.array(under_fit_m)
        return

    def image_to_fits(self, path=None, overwrite=False):
        """
        Creates a FITS file that contains the time-average imagege of the EXBA mask
        in a ImageHDU, and the source catalog in a BinTableHDU.

        Parameters
        ----------
        path : string
            Directory path where to save the FITS file.
        overwrite : bool
            Overwrite the output file.

        Returns
        -------
        hdu : ImageHDU
            An Image header unit containing the EXBA flux.
        """
        primary_hdu = fits.PrimaryHDU(data=None, header=self.tpfs[0].get_header())
        phdr = primary_hdu.header
        phdr.set("OBJECT", "EXBA mask", "type of image")
        phdr.set("RA_OBJ", self.ra.mean())
        phdr.set("DEC_OBJ", self.dec.mean())
        phdr.set("ROW_0", self.row.min(), "reference pixel value, origin top left")
        phdr.set("COL_0", self.column.min(), "reference pixel value, origin top left")

        image_hdu = fits.ImageHDU(data=self.flux_2d.mean(axis=0).value)
        image_hdu.header["TTYPE1"] = "FLUX"
        image_hdu.header["TFORM1"] = "E"
        image_hdu.header["TUNIT1"] = "e-/s"
        image_hdu.header["DATE"] = (datetime.datetime.now().strftime("%Y-%m-%d"),)

        table_hdu = fits.BinTableHDU(data=Table.from_pandas(self.sources))
        table_hdu.header["GAIA_DR"] = self.gaia_dr
        hdu = fits.HDUList([primary_hdu, image_hdu, table_hdu])

        if path is not None:
            hdu.writeto(path, overwrite=overwrite, checksum=True)
        else:
            return hdu

    def lcs_to_fits(self, path=None):
        """
        Save all the light curves to fits files...
        """
        hdu_list = []
        for i, lc in enumerate(self.lcs):
            # lc.quality = 0
            # lc.centroid_col = lc.column
            # lc.centroid_row = lc.row
            hdu = lc.to_fits(**lc.meta)
            hdu[1].header["FLFRCSAP"] = lc.FLFRCSAP
            hdu[1].header["CROWDSAP"] = lc.CROWDSAP
            hdu = lk.lightcurve._make_aperture_extension(hdu, self.aperture_mask_2d[i])
            hdu[2].header["FLFRCSAP"] = lc.FLFRCSAP
            hdu[2].header["CROWDSAP"] = lc.CROWDSAP

            del hdu[0].header["FLFRCSAP"], hdu[0].header["CROWDSAP"]

            if path is not None:
                name = "%s/lc_%s.fits" % (path, lc.label.replace(" ", "_"))
                hdu.writeto(name, overwrite=overwrite, checksum=True)
            hdu_list.append(hdu)

        return hdu_list

    def plot_image(self, frame=0, sources=True, ax=None):
        """
        Function to plot the full EXBA image and the Gaia Sources.

        Parameters
        ----------
        frame : int
            Frame number. The default is 0, i.e. the first frame.
        sources : boolean
            Whether to overplot or not the source catalog
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5, 7))
        ax = plt.subplot(projection=self.wcs)
        ax.set_title("EXBA mask Quarter %i Channel %i" % (self.quarter, self.channel))
        pc = ax.pcolormesh(
            self.column_2d,
            self.row_2d,
            self.flux_2d[frame],
            shading="auto",
            cmap="viridis",
            norm=colors.SymLogNorm(linthresh=100, vmin=0, vmax=1000, base=10),
        )
        if sources:
            ax.scatter(
                self.sources.col,
                self.sources.row,
                s=20,
                facecolors="none",
                marker="o",
                edgecolors="r",
                linewidth=1.5,
                label="Gaia Sources",
            )
        ax.set_xlabel("R.A. [hh:mm:ss]", fontsize=12)
        ax.set_ylabel("Dec [deg]", fontsize=12)
        cbar = fig.colorbar(pc)
        cbar.set_label(label=r"Flux ($e^{-}s^{-1}$)", size=12)
        ax.set_aspect("equal", adjustable="box")

        return ax

    def plot_stamp(self, source_idx=0, aperture_mask=False, ax=None):
        """
        Creates a figure with the "stamp" image of a given source and its aperture
        mask.

        Parameters
        ----------
        source_idx : int
            Index of the source in `self.sources` catalog to be plotted.
        aperture_mask : boolean
            Plot or not the aperutre mask.
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned.

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """

        if isinstance(source_idx, str):
            idx = np.where(self.sources.designation == source_idx)[0][0]
        else:
            idx = source_idx
        if ax is None:
            fig, ax = plt.subplots(1)
        pc = ax.pcolor(
            self.flux_2d[0],
            shading="auto",
            norm=colors.SymLogNorm(linthresh=50, vmin=3, vmax=5000, base=10),
        )
        ax.scatter(
            self.sources.col - self.column.min() + 0.5,
            self.sources.row - self.row.min() + 0.5,
            s=20,
            facecolors="y",
            marker="o",
            edgecolors="k",
        )
        ax.scatter(
            self.sources.col.iloc[idx] - self.column.min() + 0.5,
            self.sources.row.iloc[idx] - self.row.min() + 0.5,
            s=25,
            facecolors="r",
            marker="o",
            edgecolors="r",
        )
        ax.set_xlabel("Pixels")
        ax.set_ylabel("Pixels")
        plt.colorbar(pc, label=r"Flux ($e^{-}s^{-1}$)", ax=ax)
        ax.set_aspect("equal", adjustable="box")

        if aperture_mask:
            for i in range(self.n_rows):
                for j in range(self.n_columns):
                    if self.aperture_mask_2d[idx, i, j]:
                        rect = patches.Rectangle(
                            xy=(j, i),
                            width=1,
                            height=1,
                            color="red",
                            fill=False,
                            hatch="",
                            lw=1.5,
                        )
                        ax.add_patch(rect)
            zoom = np.argwhere(self.aperture_mask_2d[idx] == True)
            ax.set_ylim(
                np.maximum(0, zoom[0, 0] - 5),
                np.minimum(zoom[-1, 0] + 5, self.n_rows),
            )
            ax.set_xlim(
                np.maximum(0, zoom[0, -1] - 5),
                np.minimum(zoom[-1, -1] + 5, self.n_columns),
            )

            ax.set_title(
                "FLFRCSAP    %.2f\nCROWDSAP %.2f"
                % (self.FLFRCSAP[idx], self.CROWDSAP[idx]),
                bbox=dict(facecolor="white", alpha=1),
            )

        return ax

    def plot_lightcurve(self, source_idx=0, ax=None):
        """
        Creates a figure with the light curve of a given source.
        mask.

        Parameters
        ----------
        source_idx : int
            Index of the source in `self.sources` catalog to be plotted.
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned.

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(9, 3))

        if isinstance(source_idx, str):
            s = np.where(self.sources.designation == source_idx)[0][0]
        else:
            s = source_idx

        ax.set_title(
            "Channel %i  Quarter %i  Source %s (%i)"
            % (self.channel, self.quarter, self.lcs[s].label, s)
        )
        if hasattr(self, "flatten_lcs"):
            self.lcs[s].normalize().plot(label="raw", ax=ax, c="k", alpha=0.4)
            self.flatten_lcs[s].plot(label="flatten", ax=ax, c="k", offset=-0.02)
            if hasattr(self, "corrected_lcs"):
                self.corrected_lcs[s].normalize().plot(
                    label="CBV", ax=ax, c="tab:blue", offset=+0.04
                )
        else:
            self.lcs[s].plot(label="raw", ax=ax, c="k", alpha=0.4)
            if hasattr(self, "corrected_lcs"):
                self.corrected_lcs[s].plot(
                    label="CBV", ax=ax, c="tab:blue", offset=-0.02
                )

        return ax
