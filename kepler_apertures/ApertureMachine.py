import os
import glob
import warnings

import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patches
from tqdm import tqdm
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.time import Time
from astropy import units
import lightkurve as lk

from .utils import get_gaia_sources


class EXBAsources(object):
    def __init__(self, channel=53, quarter=5, magnitude_limit=25):

        self.quarter = quarter
        self.channel = channel

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
            self.ra, self.dec, epoch=self.time[0], magnitude_limit=20
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
        return "EXBA Patch:\n\t Channel %i, Quarter %s, Gaia sources %i" % (
            self.channel,
            q_result,
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
        file_name = "../data/catalogs/exba/%i/channel_%02i_gaiadr2_xmatch.csv" % (
            self.quarter,
            self.channel,
        )
        if os.path.isfile(file_name) and load:
            print("Loading query from file...")
            print(file_name)
            sources = pd.read_csv(file_name)
            sources.drop(columns=["Unnamed: 0"], inplace=True)

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
                gaia="dr2",
            )
            columns = [
                "solution_id",
                "designation",
                "source_id",
                "ref_epoch",
                "ra",
                "ra_error",
                "dec",
                "dec_error",
                "parallax",
                "parallax_error",
                "parallax_over_error",
                "duplicated_source",
                "phot_g_n_obs",
                "phot_g_mean_flux",
                "phot_g_mean_flux_error",
                "phot_g_mean_flux_over_error",
                "phot_g_mean_mag",
                "phot_bp_n_obs",
                "phot_bp_mean_flux",
                "phot_bp_mean_flux_error",
                "phot_bp_mean_flux_over_error",
                "phot_bp_mean_mag",
                "phot_rp_n_obs",
                "phot_rp_mean_flux",
                "phot_rp_mean_flux_error",
                "phot_rp_mean_flux_over_error",
                "phot_rp_mean_mag",
                "phot_bp_rp_excess_factor",
                "bp_rp",
                "bp_g",
                "g_rp",
                "radial_velocity",
                "radial_velocity_error",
                "rv_nb_transits",
                "rv_template_teff",
                "rv_template_logg",
                "rv_template_fe_h",
                "phot_variable_flag",
                "l",
                "b",
                "ecl_lon",
                "ecl_lat",
                "teff_val",
                "teff_percentile_lower",
                "teff_percentile_upper",
                "a_g_val",
                "a_g_percentile_lower",
                "a_g_percentile_upper",
                "e_bp_min_rp_val",
                "e_bp_min_rp_percentile_lower",
                "e_bp_min_rp_percentile_upper",
                "radius_val",
                "radius_percentile_lower",
                "radius_percentile_upper",
                "lum_val",
                "lum_percentile_lower",
                "lum_percentile_upper",
                "ra_gaia",
                "dec_gaia",
            ]
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

        for tdx in tqdm(range(len(self.flux)), desc="Simple SAP flux", leave=False):
            sap[:, tdx] = [self.flux[tdx][mask].value.sum() for mask in aperture_mask]
            sap_e[:, tdx] = [
                np.power(self.flux_err[tdx][mask].value, 2).sum() ** 0.5
                for mask in aperture_mask
            ]

        self.sap_lcs = lk.LightCurveCollection(
            [
                lk.KeplerLightCurve(
                    time=self.time,
                    cadenceno=self.cadences,
                    flux=sap[i],
                    flux_err=sap_e[i],
                    time_format="bkjd",
                    flux_unit="electron/s",
                    targetid=self.sources.designation[i],
                    label=self.sources.designation[i],
                    mission="Kepler",
                    quarter=int(self.quarter),
                    channel=int(self.channel),
                    ra=self.sources.ra[i],
                    dec=self.sources.dec[i],
                )
                for i in range(len(sap))
            ]
        )
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
            norm=colors.SymLogNorm(linthresh=100, vmin=0, vmax=2000, base=10),
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
        ax.set_xlabel("R.A. [hh:mm:ss]")
        ax.set_ylabel("Dec [deg]")
        fig.colorbar(pc, label=r"Flux ($e^{-}s^{-1}$)")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

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
