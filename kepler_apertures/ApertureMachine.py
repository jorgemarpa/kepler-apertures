import os
import glob
import warnings
import datetime

import numpy as np
import pandas as pd
import pickle
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

from .utils import get_gaia_sources, get_bls_periods


class EXBAMachine(object):
    def __init__(self, channel=53, quarter=5, magnitude_limit=20, gaia_dr=2):

        self.quarter = quarter
        self.channel = channel
        self.gaia_dr = gaia_dr
        self.__version__ = "0.1.0dev"

        # load local TPFs files
        tpfs_paths = np.sort(
            glob.glob(
                "../data/fits/exba/%s/%s/*_lpd-targ.fits.gz"
                % (str(quarter), str(channel))
            )
        )
        if len(tpfs_paths) == 0:
            raise FileNotFoundError("No FITS file for this channel/quarter.")
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
        self.row_2d, self.col_2d, self.flux_2d, self.flux_err_2d = (
            row.copy(),
            col.copy(),
            flux.copy(),
            flux_err.copy(),
        )
        self.row, self.col, self.flux, self.flux_err, self.unw = (
            row.ravel(),
            col.ravel(),
            flux.reshape(flux.shape[0], np.product(flux.shape[1:])),
            flux_err.reshape(flux_err.shape[0], np.product(flux_err.shape[1:])),
            unw.ravel(),
        )
        self.ra, self.dec = self._convert_to_wcs(tpfs, self.row, self.col)

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

        self.dx, self.dy, self.gaia_flux = np.asarray(
            [
                np.vstack(
                    [
                        self.col - self.sources["col"][idx],
                        self.row - self.sources["row"][idx],
                        np.zeros(len(self.col)) + self.sources.phot_g_mean_flux[idx],
                    ]
                )
                for idx in range(len(self.sources))
            ]
        ).transpose([1, 0, 2])

        self.r = np.hypot(self.dx, self.dy)
        self.phi = np.arctan2(self.dy, self.dx)

        self.N_sources = self.sources.shape[0]
        self.N_row = self.flux_2d.shape[1]
        self.N_col = self.flux_2d.shape[2]

        self.aperture_mask = np.zeros_like(self.dx).astype(bool)
        self.FLFRCSAP = np.zeros(self.sources.shape[0])
        self.CROWDSAP = np.zeros(self.sources.shape[0])
        self.cut = np.zeros(self.sources.shape[0])

    def __repr__(self):
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return "EXBA Patch:\n\t Channel %i, Quarter %s, Gaia DR %i sources %i" % (
            self.channel,
            q_result,
            self.gaia_dr,
            len(self.sources),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["tpfs"]
        return state

    # def __setstate__(self, state):

    def _parse_TPFs_channel(self, tpfs):
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
        Clean pixels with nan values and bad cadences.
        """
        # Remove cadences with nan flux
        nan_cadences = np.array([np.isnan(im).sum() == 0 for im in flux])
        times = times[nan_cadences]
        cadences = cadences[nan_cadences]
        flux = flux[nan_cadences]
        flux_err = flux_err[nan_cadences]

        return times, cadences, flux, flux_err

    def _convert_to_wcs(self, tpfs, row, col):
        ra, dec = self.wcs.wcs_pix2world(
            (col - tpfs[0].column), (row - tpfs[0].row), 0.0
        )

        return ra, dec

    def _do_query(self, ra, dec, epoch=2020, magnitude_limit=20, load=True):
        """
        Calculate ra, dec coordinates and search radius to query Gaia catalog

        Parameters
        ----------
        ra : numpy.ndarray
            Right ascension coordinate of pixels to do Gaia search
        ra : numpy.ndarray
            Declination coordinate of pixels to do Gaia search
        epoch : float
            Epoch of obervation in Julian Days of ra, dec coordinates,
            will be used to propagate proper motions in Gaia.

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
        file_name = "../data/catalogs/exba/%i/channel_%02i_gaiadr%s_xmatch.csv" % (
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
            sources.to_csv(file_name)
        return sources

    def _clean_source_list(self, sources, ra, dec):
        # find sources on the image
        inside = (
            (sources.row > self.row.min() - 1.0)
            & (sources.row < self.row.max() + 1.0)
            & (sources.col > self.col.min() - 1.0)
            & (sources.col < self.col.max() + 1.0)
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

        sap = np.zeros((self.sources.shape[0], self.flux.shape[0]))
        sap_e = np.zeros((self.sources.shape[0], self.flux.shape[0]))

        for sidx in tqdm(range(len(aperture_mask)), desc="Simple SAP flux", leave=True):
            sap[sidx, :] = self.flux[:, aperture_mask[sidx]].sum(axis=1)
            sap_e[sidx, :] = (
                np.power(self.flux_err[:, aperture_mask[sidx]].value, 2).sum(axis=1)
                ** 0.5
            )

        self.sap_flux = sap
        self.sap_flux_err = sap_e
        self.aperture_mask = aperture_mask
        self.aperture_mask_2d = aperture_mask.reshape(
            self.N_sources, self.N_row, self.N_col
        )

        return

    def create_lcs(self, aperture_mask):
        self.do_photometry(aperture_mask)
        lcs = []
        for idx, s in self.sources.iterrows():
            meta = {
                "ORIGIN": "ApertureMACHINE",
                # "APERTURE_MASK": self.aperture_mask_2d[idx],
                "LABEL": s.designation,
                "TARGETID": int(s.designation.split(" ")[-1]),
                "MISSION": "Kepler",
                "EQUINOX": 2000,
                "RA": s.ra,
                "DEC": s.dec,
                "PMRA": s.pmra / 1000,
                "PMDEC": s.pmdec / 1000,
                "PARALLAX": s.parallax,
                "GMAG": s.phot_g_mean_mag,
                "RPMAG": s.phot_rp_mean_mag,
                "BPMAG": s.phot_bp_mean_mag,
                "CHANNEL": self.channel,
                "MODULE": self.hdr["MODULE"],
                "OUTPUT": self.hdr["OUTPUT"],
                "QUARTER": self.quarter,
                "CAMPAIGN": "EXBA",
                "ROW": s.row,
                "COLUMN": s.col,
                "FLFRCSAP": self.FLFRCSAP[idx],
                "CROWDSAP": self.CROWDSAP[idx],
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

    def apply_flatten(self):
        self.flatten_lcs = lk.LightCurveCollection([lc.flatten() for lc in self.lcs])
        return

    def apply_CBV(self, do_under=False, ignore_warnings=True, plot=True):
        if ignore_warnings:
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
            if i % 10 == 0:
                print("Optimizing alpha")
                try:
                    cbvcor.correct(
                        cbv_type=cbv_type,
                        cbv_indices=cbv_indices,
                        alpha_bounds=[1e-2, 1e2],
                        target_over_score=0.9,
                        target_under_score=0.8,
                        verbose=False,
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

    def do_bls_search(self, test_lcs=None, n_boots=100, plot=False):

        if test_lcs:
            search_here = list(test_lcs) if len(test_lcs) == 1 else test_lcs
        else:
            if hasattr(self, "corrected_lcs"):
                search_here = self.corrected_lcs
            elif hasattr(self, "flatten_lcs"):
                print("No CBV correction applied, using flatten light curves.")
                search_here = self.flatten_lcs
            else:
                raise AttributeError(
                    "No CBV corrected or flatten light curves were computed,"
                    + " run `apply_CBV()` or `flatten()` first"
                )

        period_best, period_fap, periods_snr = get_bls_periods(
            search_here, plot=plot, n_boots=n_boots
        )

        if test_lcs:
            return period_best, period_fap, periods_snr
        else:
            self.period_df = pd.DataFrame(
                np.array([period_best, period_fap, periods_snr]).T,
                columns=["period_best", "period_fap", "period_snr"],
                index=self.sources.designation.values,
            )
        return

    def image_to_fits(self, path=None, overwrite=False):

        primary_hdu = fits.PrimaryHDU(data=None, header=self.tpfs[0].get_header())
        primary_hdu.header["RA_OBJ"] = self.ra.mean()
        primary_hdu.header["DEC_OBJ"] = self.dec.mean()
        primary_hdu.header["ROW_0"] = self.row.min()
        primary_hdu.header["COL_0"] = self.col.min()

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
        """Save all the light curves to fits files..."""
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

    def store_data(self):
        out_path = os.path.dirname(self.tpfs_files[0])
        with open("%s/exba_object.pkl" % (out_path), "wb") as f:
            pickle.dump(self, f)

        return

    def plot_image(self, sources=True, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5, 7))
        ax = plt.subplot(projection=self.wcs)
        ax.set_title("EXBA | Q: %i | Ch: %i" % (self.quarter, self.channel))
        pc = ax.pcolormesh(
            self.col_2d,
            self.row_2d,
            self.flux_2d[0],
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
        # plt.show()

        return ax

    def plot_stamp(self, source_idx=0, aperture_mask=False, ax=None):

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
            self.sources.col - self.col.min() + 0.5,
            self.sources.row - self.row.min() + 0.5,
            s=20,
            facecolors="y",
            marker="o",
            edgecolors="k",
        )
        ax.scatter(
            self.sources.col.iloc[idx] - self.col.min() + 0.5,
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
            for i in range(self.N_row):
                for j in range(self.N_col):
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
                np.minimum(zoom[-1, 0] + 5, self.N_row),
            )
            ax.set_xlim(
                np.maximum(0, zoom[0, -1] - 5),
                np.minimum(zoom[-1, -1] + 5, self.N_col),
            )

            ax.set_title(
                "FLFRCSAP    %.2f\nCROWDSAP %.2f"
                % (self.FLFRCSAP[idx], self.CROWDSAP[idx]),
                bbox=dict(facecolor="white", alpha=1),
            )

        return ax

    def plot_lightcurve(self, source_idx=0, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(9, 3))

        if isinstance(source_idx, str):
            s = np.where(self.sources.designation == source_idx)[0][0]
        else:
            s = source_idx

        ax.set_title(
            "Ch: %i | Q: %i | Source %s (%i)"
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

    def plot_lightcurves_stamps(self, which="all", step=1):

        s_list = self.sources.index.values
        if which != "all":
            if isinstance(which[0], str):
                s_list = s_list[np.in1d(self.sources.designation, which)]
            else:
                s_list = which
        for s in s_list[::step]:
            fig, ax = plt.subplots(
                1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [4, 1]}
            )
            self.plot_lightcurve(source_idx=s, ax=ax[0])
            self.plot_stamp(source_idx=s, ax=ax[1], aperture_mask=True)

            plt.show()

        return


class EXBALightCurveCollection(object):
    def __init__(self, lcs, metadata, periods=None):
        """
        lcs      : dictionary like
            Dictionary with data, first leveel is quarter
        metadata : dictionary like
            Dictionary with data, first leveel is quarter
        periods  : dictionary like
            Dictionary with data, first leveel is quarter
        """

        # check if each element of exba_quarters are EXBA objects
        # if not all([isinstance(exba, EXBA) for exba in EXBAs]):
        #     raise AssertionError("All elements of the list must be EXBA objects")

        self.quarter = np.unique(list(lcs.keys()))
        self.channel = np.unique([lc.channel for lc in lcs[self.quarter[0]]])

        # check that gaia sources are in all quarters
        gids = [df.designation.tolist() for q, df in metadata.items()]
        unique_gids = np.unique([item for sublist in gids for item in sublist])

        # create matris with index position to link sources across quarters
        # this asume that sources aren't in the same position in the DF, sources
        # can disapear (fall out the ccd), not all sources show up in all quarters.
        pm = np.empty((len(unique_gids), len(self.quarter)), dtype=np.int) * np.nan
        for k, id in enumerate(unique_gids):
            for i, q in enumerate(self.quarter):
                pos = np.where(id == metadata[q].designation.values)[0]
                if len(pos) == 0:
                    continue
                else:
                    pm[k, i] = pos
        # rearange sources as list of lk Collection containing all quarters per source
        sources = []
        for i, gid in enumerate(unique_gids):
            aux = [
                lcs[self.quarter[q]][int(pos)]
                for q, pos in enumerate(pm[i])
                if np.isfinite(pos)
            ]
            sources.append(lk.LightCurveCollection(aux))
        self.lcs = sources
        self.metadata = (
            pd.concat(metadata, axis=0, join="outer")
            .drop_duplicates(["designation"], ignore_index=True)
            .drop(["Unnamed: 0", "col", "row"], axis=1)
            .sort_values("designation")
            .reset_index(drop=True)
        )
        self.periods = (
            pd.concat(periods, axis=0, join="outer")
            .drop_duplicates(["designation"], ignore_index=True)
            .drop(["Unnamed: 0", "col", "row"], axis=1)
            .sort_values("designation")
            .reset_index(drop=True)
        )
        print(self)

    def __repr__(self):
        ch_result = ",".join([str(k) for k in list([self.channel])])
        q_result = ",".join([str(k) for k in list([self.quarter])])
        return (
            "Light Curves from: \n\tChannels %s \n\tQuarters %s \n\tGaia sources %i"
            % (
                ch_result,
                q_result,
                len(self.lcs),
            )
        )

    def to_fits(self):
        """Save all the light curves to fits files..."""
        raise NotImplementedError

    def stitch_quarters(self):

        # lk.LightCurveCollection.stitch() normalize by default all lcs before stitching
        if hasattr(self, "source_lcs"):
            self.stitched_lcs = lk.LightCurveCollection(
                [lc.stitch() for lc in self.source_lcs]
            )

        return

    def do_bls_long(self, source_idx=0, plot=True, n_boots=50):
        if isinstance(source_idx, str):
            s = np.where(self.metadata.designation == source_idx)[0][0]
        else:
            s = source_idx
        print(self.lcs[s])
        periods, faps, snrs = get_bls_periods(self.lcs[s], plot=plot, n_boots=n_boots)
        return periods, faps, snrs

    def do_bls_search_all(self, plot=False, n_boots=100, fap_tresh=0.1, save=True):

        self.metadata["has_planet"] = False
        self.metadata["N_periodic_quarters"] = 0
        self.metadata["Period"] = None
        self.metadata["Period_snr"] = None
        self.metadata["Period_fap"] = None

        for lc_long in tqdm(self.lcs, desc="Gaia sources"):
            periods, faps, snrs = get_bls_periods(lc_long, plot=plot, n_boots=n_boots)
            # check for significant periodicity detection in at least one quarter
            if np.isfinite(faps).all():
                p_mask = faps < fap_tresh
            else:
                p_mask = snrs > 50
            if p_mask.sum() > 0:
                # check that periods are similar, within a tolerance
                # this assumes that there's one only one period
                # have to fix this to make it work for multiple true periods detected
                # or if BLS detected ane of the armonics, not necesary yet, when need it
                # use np.round(periods) and np.unique() to check for same periods and
                # harmonics within tolerance.
                good_periods = periods[p_mask]
                good_faps = faps[p_mask]
                good_snrs = snrs[p_mask]
                all_close = (
                    np.array(
                        [np.isclose(p, good_periods, atol=0.1) for p in good_periods]
                    ).sum(axis=0)
                    > 1
                )
                if all_close.sum() > 1:
                    idx = np.where(self.metadata.designation == lc_long[0].label)[0]
                    self.metadata["has_planet"].iloc[idx] = True
                    self.metadata["N_periodic_quarters"].iloc[idx] = all_close.sum()
                    self.metadata["Period"].iloc[idx] = good_periods[all_close].mean()
                    self.metadata["Period_fap"].iloc[idx] = good_faps[all_close].mean()
                    self.metadata["Period_snr"].iloc[idx] = good_snrs[all_close].mean()
                # check if periods are harmonics
            # break
        if save:
            cols = [
                "designation",
                "source_id",
                "ref_epoch",
                "ra",
                "ra_error",
                "dec",
                "dec_error",
                "has_planet",
                "N_periodic_quarters",
                "Period",
                "Period_snr",
                "Period_fap",
            ]
            if len(self.channel) == 1:
                ch_str = "%i" % (self.channel[0])
            else:
                ch_str = "%i-%i" % (self.channel[0], self.channel[-1])
            outname = "BLS_results_ch%s.csv" % (ch_str)
            print("Saving data to --> %s/data/bls_results/%s" % (main_path, outname))
            self.metadata.loc[:, cols].to_csv(
                "%s/data/bls_results/%s" % (main_path, outname)
            )

    @staticmethod
    def from_stored_lcs(quarters, channels, lc_type="cbv"):

        # load gaia catalogs and lcs
        metadata, lcs, periods, nodata = {}, {}, {}, []
        for q in tqdm(quarters, desc="Quarters", leave=True):
            metadata_, lcs_, periods_ = [], [], []
            for ch in channels:
                obj_path = "../data/fits/exba/%i/%i/exba_object.pkl" % (q, ch)
                if not os.path.isfile(obj_path):
                    print(
                        "WARNING: quarter %i channel %i have no storaged files"
                        % (q, ch)
                    )
                    nodata.append([q, ch])
                    continue
                exba = pickle.load(open(obj_path, "rb"))
                #     metadata_.append(exba.sources)
                #     lcs_.extend(exba.corrected_lcs)
                periods_.append(exba.period_df)
            #     del obj_path
            # metadata[q] = pd.concat(metadata_, axis=0)
            # lcs[q] = lcs_
            periods[q] = pd.concat(periods_, axis=0)

        # return EXBALightCurveCollection(lcs, metadata, periods=periods)
        return periods
