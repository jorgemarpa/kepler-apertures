import os
import sys
import glob
import warnings
import wget

import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.stats import sigma_clip, SigmaClip
from astropy.time import Time
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground, BkgZoomInterpolator

from .utils import get_gaia_sources, make_A_edges, solve_linear_model, _make_A_polar

r_min, r_max = 20, 1044
c_min, c_max = 12, 1112
remove_sat = True
mask_bright = True

quarter_ffi = {
    0: [
        "kplr2009114174833_ffi-cal.fits",
        "kplr2009114204835_ffi-cal.fits",
        "kplr2009115002613_ffi-cal.fits",
        "kplr2009115053616_ffi-cal.fits",
        "kplr2009115080620_ffi-cal.fits",
        "kplr2009115131122_ffi-cal.fits",
        "kplr2009115173611_ffi-cal.fits",
        "kplr2009116035924_ffi-cal.fits",
    ],
    1: [],
    2: ["kplr2009231194831_ffi-cal.fits"],
    3: ["kplr2009292020429_ffi-cal.fits", "kplr2009322233047_ffi-cal.fits"],
    4: [
        "kplr2010019225502_ffi-cal.fits",
        "kplr2010020005046_ffi-cal.fits",
        "kplr2010049182302_ffi-cal.fits",
    ],
    5: ["kplr2010111125026_ffi-cal.fits", "kplr2010140101631_ffi-cal.fits"],
    6: ["kplr2010203012215_ffi-cal.fits", "kplr2010234192745_ffi-cal.fits"],
    7: ["kplr2010296192119_ffi-cal.fits", "kplr2010326181728_ffi-cal.fits"],
    8: ["kplr2011024134926_ffi-cal.fits", "kplr2011053174401_ffi-cal.fits"],
    9: ["kplr2011116104002_ffi-cal.fits", "kplr2011145152723_ffi-cal.fits"],
    10: ["kplr2011208112727_ffi-cal.fits", "kplr2011240181752_ffi-cal.fits"],
    11: ["kplr2011303191211_ffi-cal.fits", "kplr2011334181008_ffi-cal.fits"],
    12: ["kplr2012032101442_ffi-cal.fits", "kplr2012060123308_ffi-cal.fits"],
    13: ["kplr2012121122500_ffi-cal.fits", "kplr2012151105138_ffi-cal.fits"],
    14: ["kplr2012211123923_ffi-cal.fits", "kplr2012242195726_ffi-cal.fits"],
    15: ["kplr2012310200152_ffi-cal.fits", "kplr2012341215621_ffi-cal.fits"],
    16: ["kplr2013038133130_ffi-cal.fits", "kplr2013065115251_ffi-cal.fits"],
    17: ["kplr2013098115308_ffi-cal.fits"],
}


class KeplerFFI(object):
    def __init__(
        self,
        ffi_name: str = "",
        channel: int = 1,
        quarter: int = None,
        plot: bool = True,
        save: bool = True,
    ):

        self.channel = channel
        self.plot = plot
        self.save = save
        self.show = False

        if quarter is not None and quarter in np.arange(18):
            fname = "../data/fits/ffi/%s" % (quarter_ffi[quarter][0])
        elif len(ffi_name) == 17 and ffi_name[:4] == "kplr":
            fname = "../data/fits/ffi/%s_ffi-cal.fits" % (ffi_name)
        else:
            raise ValueError("Invalid quarter or FFI fits file name")

        if not os.path.isfile(fname):
            print("Downloading FFI fits files")
            fits_name = fname.split("/")[-1]
            print(fits_name)
            self.download_ffi(fits_name)

        self.hdr = fits.open(fname)[channel].header
        self.img = fits.open(fname)[channel].data
        self.wcs = WCS(self.hdr)
        self.quarter = quarter if quarter is not None else self.hdr["MJDSTART"]

        row_2d, col_2d = np.mgrid[: self.img.shape[0], : self.img.shape[1]]
        row, col = row_2d.ravel(), col_2d.ravel()
        ra, dec = self.wcs.all_pix2world(np.vstack([col, row]).T, 0).T
        ra_2d, dec_2d = ra.reshape(self.img.shape), dec.reshape(self.img.shape)

        # get coordinates of the center for query
        loc = (self.img.shape[0] // 2, self.img.shape[1] // 2)
        ra_q, dec_q = self.wcs.all_pix2world(np.atleast_2d(loc), 0).T
        rad = [np.hypot(ra - ra.mean(), dec - dec.mean()).max()]

        time = Time(self.hdr["TSTART"] + 2454833, format="jd")
        print(
            "Will query with this (ra, dec, radius, epoch): ",
            ra_q,
            dec_q,
            rad,
            time.jyear,
        )
        if ra_q[0] > 360 or np.abs(dec_q[0]) > 90 or rad[0] > 5:
            raise ValueError(
                "Query values are out of bound, please check WCS solution."
            )
        sources = self._do_query(ra_q, dec_q, rad, time.jyear)
        sources["col"], sources["row"] = self.wcs.all_world2pix(
            sources.loc[:, ["ra", "dec"]].values, 0.5
        ).T

        # correct col,row columns for gaia sources
        sources.row -= r_min
        sources.col -= c_min

        # clean out-of-ccd and blended sources
        clean_sources = self._clean_source_list(sources)
        del sources

        # remove useless Pixels
        self.col_2d = col_2d[r_min:r_max, c_min:c_max] - c_min
        self.row_2d = row_2d[r_min:r_max, c_min:c_max] - r_min
        self.ra_2d = ra_2d[r_min:r_max, c_min:c_max]
        self.dec_2d = dec_2d[r_min:r_max, c_min:c_max]
        flux_2d = self.img[r_min:r_max, c_min:c_max]

        # background substraction
        self.flux_2d = flux_2d - self._model_bkg(flux_2d, mask=None)

        # ravel arrays
        col = self.col_2d.ravel()
        row = self.row_2d.ravel()
        ra = self.ra_2d.ravel()
        dec = self.dec_2d.ravel()
        flux = self.flux_2d.ravel()

        if remove_sat:
            non_sat_mask = ~self._saturated_pixels_mask(
                flux, col, row, saturation_limit=1.5e5
            )
            print("Saturated pixels %i" % (np.sum(~non_sat_mask)))
            self.non_sat_mask = non_sat_mask

            col = col[non_sat_mask]
            row = row[non_sat_mask]
            ra = ra[non_sat_mask]
            dec = dec[non_sat_mask]
            flux = flux[non_sat_mask]

        if mask_bright:
            bright_mask = ~self._mask_bright_sources(
                flux, col, row, clean_sources, mag_limit=10
            )
            print("Bright pixels %i" % (np.sum(~bright_mask)))
            self.bright_mask = bright_mask

            col = col[bright_mask]
            row = row[bright_mask]
            ra = ra[bright_mask]
            dec = dec[bright_mask]
            flux = flux[bright_mask]

        clean_sources = clean_sources[
            (clean_sources.phot_g_mean_flux > 1e3)
            & (clean_sources.phot_g_mean_flux < 1e6)
        ].reset_index(drop=True)

        print("Total Gaia sources %i" % (clean_sources.shape[0]))

        self.sources = clean_sources
        self.flux = flux
        self.flux_err = np.sqrt(np.abs(self.flux))
        self.col = col
        self.row = row
        self.nsurces = clean_sources.shape[0]
        self.npixels = self.flux.shape[0]

        self.rmin = 0.25
        self.rmax = 3.0

        # self._create_sparse()

        # # compute PSF edge model
        # print("Computing PSF edges...")
        # radius = self._find_psf_edge(
        #     self.r, self.dflux, self.gf, radius_limit=6.0, cut=200, dm_type="cubic"
        # )
        #
        # # compute PSF model
        # print("Computing PSF model...")
        # self.psf_data = self._build_psf_model(
        #     self.r, self.phi, self.dflux, self.gf, radius * 2, self.dx, self.dy
        #     rknots=4, phiknots=12
        # )

    @staticmethod
    def download_ffi(fits_name):
        """
        Download FFI fits file to a dedicated quarter directory
        Parameters
        ----------
        fits_name : string
            Name of FFI fits file
        """
        url = "https://archive.stsci.edu/missions/kepler/ffi"
        if fits_name == "":
            raise ValueError("Invalid fits file name")

        if not os.path.isdir("../data/fits/ffi"):
            os.mkdir("../data/fits/ffi")

        out = "../data/fits/ffi/%s" % (fits_name)
        wget.download("%s/%s" % (url, fits_name), out=out)

        return

    def _do_query(self, ra_q, dec_q, rad, epoch):
        file_name = "../data/catalogs/ffi/%s/channel_%i_gaia_xmatch.csv" % (
            str(self.quarter),
            self.channel,
        )
        if os.path.isfile(file_name):
            print("Loading query from file...")
            print(file_name)
            sources = pd.read_csv(file_name)
        else:
            try:
                sources = get_gaia_sources(
                    tuple(ra_q),
                    tuple(dec_q),
                    tuple(rad),
                    magnitude_limit=18,
                    epoch=epoch,
                    dr=3,
                )

            except (TimeoutError, ConnectionResetError):
                print("TimeoutError... \nExiting.")
                sys.exit()
            print("Saving query to file...")
            print(file_name)
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
            sources = sources.loc[:, columns]

            if not os.path.isdir("../data/catalogs/ffi/%s" % str(self.quarter)):
                os.mkdir("../data/catalogs/ffi/%s" % str(self.quarter))
            sources.to_csv(file_name)
        return sources

    def _clean_source_list(self, sources):

        print("Cleaning sources table...")

        # find sources inside the image with 10 pix of inward tolerance
        inside = (
            (sources.row > 10)
            & (sources.row < 1014)
            & (sources.col > 10)
            & (sources.col < 1090)
        )
        sources = sources[inside].reset_index(drop=True)

        # find well separated sources
        s_coords = SkyCoord(sources.ra, sources.dec, unit=("deg"))
        midx, mdist = match_coordinates_3d(s_coords, s_coords, nthneighbor=2)[:2]
        # remove sources closer than 4" = 1 pix
        closest = mdist.arcsec < 8.0
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

        sources = sources[~unresolved].reset_index(drop=True)

        return sources

    def _model_bkg(self, data, mask=None):
        """
        BkgZoomInterpolator:
        This class generates full-sized background and background RMS images
        from lower-resolution mesh images using the `~scipy.ndimage.zoom`
        (spline) interpolator.
        """
        model = Background2D(
            data,
            mask=mask,
            box_size=(64, 50),
            filter_size=15,
            exclude_percentile=20,
            sigma_clip=SigmaClip(sigma=3.0, maxiters=5),
            bkg_estimator=MedianBackground(),
            interpolator=BkgZoomInterpolator(order=3),
        )

        return model.background

    def _saturated_pixels_mask(self, flux, column, row, saturation_limit=1.5e5):
        """Finds and removes saturated pixels, including bleed columns."""
        # Which pixels are saturated
        # saturated = np.nanpercentile(flux, 99, axis=0)
        saturated = np.where((flux > saturation_limit).astype(float))[0]

        # Find bad pixels, including allowence for a bleed column.
        bad_pixels = np.vstack(
            [
                np.hstack([column[saturated] + idx for idx in np.arange(-3, 3)]),
                np.hstack([row[saturated] for idx in np.arange(-3, 3)]),
            ]
        ).T
        # Find unique row/column combinations
        bad_pixels = bad_pixels[
            np.unique(["".join(s) for s in bad_pixels.astype(str)], return_index=True)[
                1
            ]
        ]
        # Build a mask of saturated pixels
        m = np.zeros(len(column), bool)
        for p in bad_pixels:
            m |= (column == p[0]) & (row == p[1])
        return m

    def _mask_bright_sources(self, flux, column, row, sources, mag_limit=10):
        """Finds and removes halos produced by bright stars (<10 mag)"""
        bright_mask = sources["phot_g_mean_mag"] <= mag_limit
        mask_radius = 30  # Pixels

        mask = [
            np.hypot(column - s.col, row - s.row) < mask_radius
            for _, s in sources[bright_mask].iterrows()
        ]
        mask = np.array(mask).sum(axis=0) > 0

        return mask

    def _create_sparse(self):
        # create dx, dy, gf, r, phi, vectors
        # gaia estimate flux values per pixel to be used as flux priors
        dx, dy, sparse_mask = [], [], []
        for i in tqdm(range(len(self.sources)), desc="Gaia sources"):
            dx_aux = self.col - self.sources["col"].iloc[i]
            dy_aux = self.row - self.sources["row"].iloc[i]
            near_mask = sparse.csr_matrix((np.abs(dx_aux) <= 7) & (np.abs(dy_aux) <= 7))

            dx.append(near_mask.multiply(dx_aux))
            dy.append(near_mask.multiply(dy_aux))
            sparse_mask.append(near_mask)

        del dx_aux, dy_aux, near_mask
        dx = sparse.vstack(dx, "csr")
        dy = sparse.vstack(dy, "csr")
        sparse_mask = sparse.vstack(sparse_mask, "csr")
        sparse_mask.eliminate_zeros()

        self.gf = self.sources["phot_g_mean_flux"].values
        self.dflux = sparse_mask.multiply(self.flux).tocsr()
        self.dflux_err = np.sqrt(np.abs(self.dflux))

        # eliminate leaked zero flux values in the sparse_mask
        self.sparse_mask = self.dflux.astype(bool)
        self.dx = self.sparse_mask.multiply(dx).tocsr()
        self.dy = self.sparse_mask.multiply(dy).tocsr()
        del dx, dy, sparse_mask

        # convertion to polar coordinates
        print("to polar coordinates...")
        nnz_inds = self.sparse_mask.nonzero()
        r_vals = np.hypot(self.dx.data, self.dy.data)
        phi_vals = np.arctan2(self.dy.data, self.dx.data)
        self.r = sparse.csr_matrix(
            (r_vals, (nnz_inds[0], nnz_inds[1])),
            shape=self.sparse_mask.shape,
            dtype=float,
        )
        self.phi = sparse.csr_matrix(
            (phi_vals, (nnz_inds[0], nnz_inds[1])),
            shape=self.sparse_mask.shape,
            dtype=float,
        )
        del r_vals, phi_vals, nnz_inds

        return

    def _get_source_mask(
        self,
        upper_radius_limit=7,
        lower_radius_limit=1.1,
        flux_cut_off=300,
        dm_type="rf-quadratic",
    ):
        r = self.r
        mean_flux = self.dflux
        gf = self.gf

        nonz_idx = r.nonzero()
        rad_mask = r.data < upper_radius_limit
        temp_mask = sparse.csr_matrix(
            (r.data[rad_mask], (nonz_idx[0][rad_mask], nonz_idx[1][rad_mask])),
            shape=r.shape,
        ).astype(bool)
        temp_mask = temp_mask.multiply(temp_mask.sum(axis=0) == 1).tocsr()
        temp_mask.eliminate_zeros()

        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log10(temp_mask.astype(float).multiply(mean_flux).data)
        k = np.isfinite(f)
        f_mask = f[k]
        r_mask = temp_mask.astype(float).multiply(r).data[k]
        gf_mask = temp_mask.astype(float).multiply(gf[:, None]).data[k]
        k = np.isfinite(f_mask)

        A = make_A_edges(r_mask, np.log10(gf_mask), type=dm_type)

        for count in [0, 1, 2]:
            sigma_w_inv = A[k].T.dot(A[k])
            B = A[k].T.dot(f_mask[k])
            w = np.linalg.solve(sigma_w_inv, B)
            res = np.ma.masked_array(f_mask, ~k) - A.dot(w)
            k &= ~sigma_clip(res, sigma=3).mask

        test_f = np.linspace(
            np.log10(gf_mask.min()),
            np.log10(gf_mask.max()),
            100,
        )
        test_r = np.arange(lower_radius_limit, upper_radius_limit, 0.125)
        test_r2, test_f2 = np.meshgrid(test_r, test_f)

        test_A = make_A_edges(test_r2.ravel(), test_f2.ravel(), type=dm_type)
        test_val = test_A.dot(w).reshape(test_r2.shape)

        # find radius where flux > cut
        lr = np.zeros(len(test_f)) * np.nan
        for idx in range(len(test_f)):
            loc = np.where(10 ** test_val[idx] < flux_cut_off)[0]
            if len(loc) > 0:
                lr[idx] = test_r[loc[0]]

        ok = np.isfinite(lr)
        polifit_results = np.polyfit(test_f[ok], lr[ok], 2)
        source_radius_limit = np.polyval(polifit_results, np.log10(gf))
        source_radius_limit[
            source_radius_limit > upper_radius_limit
        ] = upper_radius_limit
        source_radius_limit[
            source_radius_limit < lower_radius_limit
        ] = lower_radius_limit

        self.radius = source_radius_limit + 0.5
        # self.source_mask = sparse.csr_matrix(self.r.value < self.radius[:, None])

        if self.save:
            to_save = dict(w=w, polifit_results=polifit_results)
            output = "../data/models/%s/channel_%02i_psf_edge_model_%s.pkl" % (
                str(self.quarter),
                self.channel,
                dm_type,
            )
            if not os.path.isdir("../data/models/%s" % str(self.quarter)):
                os.mkdir("../data/models/%s" % str(self.quarter))
            with open(output, "wb") as file:
                pickle.dump(to_save, file)

        if self.plot:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

            ax[0].scatter(r_mask, f_mask, s=0.4, c="k", alpha=0.5, label="Data")
            ax[0].scatter(
                r_mask[k],
                f_mask[k],
                s=0.4,
                c="g",
                alpha=0.5,
                label="Data clipped",
                rasterized=True,
            )
            ax[0].scatter(
                r_mask[k], A[k].dot(w), c="r", s=0.4, alpha=0.7, label="Model"
            )
            ax[0].set(
                xlabel=("Radius from Source [pix]"), ylabel=("log$_{10}$ Kepler Flux")
            )
            ax[0].legend(frameon=True, loc="upper right")

            im = ax[1].pcolormesh(
                test_f2,
                test_r2,
                10 ** test_val,
                vmin=0,
                vmax=500,
                cmap="viridis",
                shading="auto",
                rasterized=True,
            )
            line = np.polyval(np.polyfit(test_f[ok], lr[ok], 2), test_f)
            line[line > upper_radius_limit] = upper_radius_limit
            line[line < lower_radius_limit] = lower_radius_limit
            ax[1].plot(test_f, line, color="r", label="Best Fit PSF Edge")
            ax[1].legend(frameon=True, loc="upper left")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(r"PSF Flux [$e^-s^{-1}$]")

            ax[1].set(
                ylabel=("Radius from Source [pix]"),
                xlabel=("log$_{10}$ Source Flux"),
            )
            if self.save:
                fig_name = "../data/figures/%s/channel_%02i_psf_edge_model_%s.png" % (
                    str(self.quarter),
                    self.channel,
                    dm_type,
                )
                if not os.path.isdir("../data/figures/%s" % (str(self.quarter))):
                    os.mkdir("../data/figures/%s" % (str(self.quarter)))

                plt.savefig(fig_name, format="png", bbox_inches="tight")
                plt.close()
            elif self.show:
                plt.show()

        return self.radius

    # @profile
    def _build_psf_model(self, rknots=10, phiknots=12, flux_cut_off=1):

        r = self.r
        phi = self.phi
        mean_flux = self.dflux
        mean_flux_err = self.dflux_err
        flux_estimates = self.gf
        warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # remove pixels outside the radius limit and contaminated pixels
        source_mask = []
        for s in range(r.shape[0]):
            nonz_idx = r[s].nonzero()
            rad_mask = r[s].data < self.radius[s]
            aux = sparse.csr_matrix(
                (r[s].data[rad_mask], (nonz_idx[0][rad_mask], nonz_idx[1][rad_mask])),
                shape=r[s].shape,
            ).astype(bool)
            source_mask.append(aux)
        source_mask = sparse.vstack(source_mask, "csr")
        print("# Contaminated pixels: ", (source_mask.sum(axis=0) > 1).sum())
        source_mask = source_mask.multiply(source_mask.sum(axis=0) == 1).tocsr()
        source_mask.eliminate_zeros()

        # mean flux values using uncontaminated mask and normalized by flux estimations
        mean_f = np.log10(
            source_mask.astype(float)
            .multiply(mean_flux)
            .multiply(1 / flux_estimates[:, None])
            .data
        )
        mean_f_err = np.abs(
            source_mask.astype(float)
            .multiply(mean_flux_err)
            .multiply(1 / flux_estimates[:, None])
            .data
        )
        phi_b = source_mask.multiply(phi).data
        r_b = source_mask.multiply(r).data

        # build a design matrix A with b-splines basis in radius and angle axis.
        A = _make_A_polar(
            phi_b.ravel(),
            r_b.ravel(),
            cut_r=6,
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=rknots,
            n_phi_knots=phiknots,
        )
        prior_sigma = np.ones(A.shape[1]) * 100
        prior_mu = np.zeros(A.shape[1]) - 10
        nan_mask = np.isfinite(mean_f.ravel())

        # we solve for A * psf_w = mean_f
        for count in [0, 1, 2]:
            psf_w = solve_linear_model(
                A,
                mean_f.ravel(),
                k=nan_mask,
                prior_mu=prior_mu,
                prior_sigma=prior_sigma,
                errors=False,
            )
            res = np.ma.masked_array(mean_f.ravel(), ~nan_mask) - A.dot(psf_w)
            nan_mask &= ~sigma_clip(res, sigma=3).mask

        # We evaluate our DM and build PSF models per source
        mean_model = sparse.csr_matrix(r.shape)
        m = 10 ** A.dot(psf_w)
        mean_model[source_mask] = m
        mean_model.eliminate_zeros()
        # mean_model = mean_model.multiply(1 / mean_model.sum(axis=1))

        #  re-estimate source flux (from CH updates)
        prior_mu = flux_estimates
        prior_sigma = np.ones(mean_model.shape[0]) * 10 * flux_estimates

        X = mean_model.copy().T

        ws, werrs = solve_linear_model(
            X,
            mean_f.ravel(),
            y_err=mean_f_err.ravel(),
            k=None,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            errors=True,
        )

        # Rebuild source mask
        ok = np.abs(ws - flux_estimates) / werrs > 3
        ok &= ((ws / flux_estimates) < 10) & ((flux_estimates / ws) < 10)
        ok &= ws > 10
        ok &= werrs > 0

        flux_estimates[ok] = ws[ok]

        source_mask = (
            mean_model.multiply(mean_model.T.dot(flux_estimates)).tocsr() > flux_cut_off
        )
        source_mask = source_mask.multiply(source_mask.sum(axis=0) == 1).tocsr()
        source_mask.eliminate_zeros()
        self.rmax = np.percentile(source_mask.multiply(r).data, 99.9)
        Ap = _make_A_polar(
            source_mask.multiply(phi).data,
            source_mask.multiply(r).data,
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=rknots,
            n_phi_knots=phiknots,
        )

        # And create a `mean_model` that has the psf model for all pixels with fluxes
        mean_model = sparse.csr_matrix(r.shape)
        m = 10 ** Ap.dot(psf_w)
        m[~np.isfinite(m)] = 0
        mean_model[source_mask] = m
        mean_model.eliminate_zeros()
        self.mean_model = mean_model
        self.source_mask = source_mask

        mean_f = np.log10(
            source_mask.astype(float)
            .multiply(mean_flux)
            .multiply(1 / flux_estimates[:, None])
            .data
        )
        self.psf_data = dict(
            psf_w=psf_w,
            A=Ap,
            x_data=source_mask.multiply(self.dx).data,
            y_data=source_mask.multiply(self.dy).data,
            f_data=mean_f,
            f_model=np.log10(m),
            clip_mask=nan_mask,
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=rknots,
            n_phi_knots=phiknots,
        )

        print("Total number of pixels: ", mean_f.shape)

        if self.save:
            output = "../data/models/%s/channel_%02i_psf_model.pkl" % (
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir("../data/models/%s" % str(self.quarter)):
                os.mkdir("../data/models/%s" % str(self.quarter))
            with open(output, "wb") as file:
                pickle.dump(self.psf_data, file)

        if self.plot:
            # Plotting r,phi,meanflux used to build PSF model
            ylim = source_mask.multiply(r).data.max() * 1.1
            vmin = -3
            vmax = -0.5
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            ax[0, 0].set_title("Mean flux")
            cax = ax[0, 0].scatter(
                source_mask.multiply(phi).data,
                source_mask.multiply(r).data,
                c=mean_f,
                marker=".",
                s=2,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax[0, 0].set_ylim(0, ylim)
            fig.colorbar(cax, ax=ax[0, 0])
            ax[0, 0].set_ylabel(r"$r$ [pixels]")
            ax[0, 0].set_xlabel(r"$\phi$ [rad]")

            ax[0, 1].set_title("Average PSF Model")
            cax = cax = ax[0, 1].scatter(
                source_mask.multiply(phi).data,
                source_mask.multiply(r).data,
                c=np.log10(m),
                marker=".",
                s=2,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax[0, 1].set_ylim(0, ylim)
            fig.colorbar(cax, ax=ax[0, 1])
            ax[0, 1].set_xlabel(r"$\phi$ [rad]")

            cax = ax[1, 0].scatter(
                source_mask.multiply(self.dx).data,
                source_mask.multiply(self.dy).data,
                c=mean_f,
                marker=".",
                s=2,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            fig.colorbar(cax, ax=ax[1, 0])
            ax[1, 0].set_ylabel("dy")
            ax[1, 0].set_xlabel("dx")

            cax = ax[1, 1].scatter(
                source_mask.multiply(self.dx).data,
                source_mask.multiply(self.dy).data,
                c=np.log10(m),
                marker=".",
                s=2,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            fig.colorbar(cax, ax=ax[1, 1])
            ax[1, 1].set_xlabel("dx")

            if self.save:
                fig_name = "../data/figures/%s/channel_%02i_psf_model.png" % (
                    str(self.quarter),
                    self.channel,
                )
                if not os.path.isdir("../data/figures/%s" % str(self.quarter)):
                    os.mkdir("../data/figures/%s" % str(self.quarter))
                plt.savefig(fig_name, format="png", bbox_inches="tight")
                plt.close()
            elif self.show:
                plt.show()

        return

    # @profile
    def fit_model(self):
        prior_mu = self.gf
        prior_sigma = np.ones(self.mean_model.shape[0]) * 5 * np.abs(self.gf) ** 0.5

        X = self.mean_model.copy()
        X = X.T
        f = self.flux
        fe = self.flux_err

        self.ws, self.werrs = solve_linear_model(
            X, f, y_err=fe, prior_mu=prior_mu, prior_sigma=prior_sigma, errors=True
        )
        self.model_flux = X.dot(self.ws)

        nodata = np.asarray(self.source_mask.sum(axis=1))[:, 0] == 0
        # These sources are poorly estimated
        nodata |= (self.mean_model.max(axis=1) > 1).toarray()[:, 0]
        self.ws[nodata] *= np.nan
        self.werrs[nodata] *= np.nan

        return

    def save_catalog(self):
        df = pd.DataFrame(
            [
                self.sources.designation,
                self.sources.ra,
                self.sources.dec,
                self.sources.col,
                self.sources.row,
                self.ws,
                self.werrs,
            ],
            index=["Gaia_source_id", "RA", "DEC", "Column", "Row", "Flux", "Flux_err"],
        ).T

        if not os.path.isdir("../data/catalogs/ffi/source_catalog/"):
            os.mkdir("../data/catalogs/ffi/source_catalog/")
        df.to_csv(
            "../data/catalogs/ffi/source_catalog/channel_%s_source_catalog_mjd_%s.csv"
            % (self.channel, str(self.hdr["MJDSTART"]))
        )

    def plot_image(self, ax=None, sources=False):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 10))
        ax = plt.subplot(projection=self.wcs)
        im = ax.imshow(
            self.flux_2d,
            cmap=plt.cm.viridis,
            origin="lower",
            norm=colors.SymLogNorm(linthresh=200, vmin=0, vmax=2000, base=10),
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label=r"Flux ($e^{-}s^{-1}$)", fraction=0.042)

        ax.set_title("FFI Ch %i" % (self.channel))
        ax.set_xlabel("R.A. [hh:mm]")
        ax.set_ylabel("Decl. [deg]")
        ax.grid(color="white", ls="solid")
        ax.set_aspect("equal", adjustable="box")

        if sources:
            ax.scatter(
                self.sources.col,
                self.sources.row,
                facecolors="none",
                edgecolors="r",
                linewidths=0.5,
                alpha=0.9,
            )

        if self.save:
            fig_name = "../data/figures/%s/channel_%02i_ffi_image.png" % (
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir("../data/figures/%s" % str(self.quarter)):
                os.mkdir("../data/figures/%s" % str(self.quarter))
            plt.savefig(fig_name, format="png", bbox_inches="tight")

        return ax

    def plot_pixel_masks(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.scatter(
            self.col_2d.ravel()[self.non_sat_mask][~self.bright_mask],
            self.row_2d.ravel()[self.non_sat_mask][~self.bright_mask],
            c="r",
            marker=".",
            label="bright",
        )
        ax.scatter(
            self.col_2d.ravel()[~self.non_sat_mask],
            self.row_2d.ravel()[~self.non_sat_mask],
            c="y",
            marker=".",
            label="saturated",
        )
        ax.legend(loc="best")

        ax.set_xlabel("Column Pixel Number")
        ax.set_ylabel("Row Pixel Number")
        ax.set_title("Pixel Mask")

        if self.save:
            fig_name = "../data/figures/%s/channel_%02i_ffi_pixel_mask.png" % (
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir("../data/figures/%s" % str(self.quarter)):
                os.mkdir("../data/figures/%s" % str(self.quarter))
            plt.savefig(fig_name, format="png", bbox_inches="tight")

        return ax
