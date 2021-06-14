"""
Defines the KeplerFFI class that uses FFIs to model the PRF shape for a given
channel and quarter.
"""
import os
import sys
import warnings
import wget

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import sparse
from tqdm.auto import tqdm
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, match_coordinates_3d
from astropy.stats import sigma_clip, SigmaClip
from astropy.time import Time
from astropy.wcs import WCS
from photutils import Background2D, MedianBackground, BkgZoomInterpolator

from . import PACKAGEDIR, DATAOUTDIR
from .utils import get_gaia_sources, make_A_edges, solve_linear_model, _make_A_polar

r_min, r_max = 20, 1044
c_min, c_max = 12, 1112
remove_sat = True
mask_bright = True

# dictionary with FFI file names and in-quarter mapping
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
    """
    Class for loading Kepler's FFI files and compute PRF models out of them following
    the method discussed in Hedges et al. 2021 and Martinez-Palomera et al. 2021.
    """

    def __init__(
        self,
        ffi_name: str = "",
        channel: int = 1,
        quarter: int = None,
        plot: bool = True,
        save: bool = True,
    ):
        """
        Initialization of the KeplerFFI class
        Parameters
        ----------
        ffi_name : string
            Name of the FFI file used to model the PRF profile. Either ffi_name or
            quarter can be provided.
        channel : int
            Channel number of the FFI to be used to model the PRF. Valid values are
            between 1 and 84.
        quarter : int
            Number of the quarter that will be used to model the PRF.
            Either ffi_name or quarter can be provided, if quarter is provided,
            then all FFI files observed during the time window of that quarter will
            be used to model the PRF by averaging the images.
            Valid values are between 1 and 17.
        plot : boolean
            Whether to clreate diagnostic plots or not.
        save : boolean
            Whether to save the models or not.

        Attributes
        ----------
        channel : int
            Number of the quarter that will be used to model the PRF.
        quarter : int
            Channel number of the FFI to be used to model the PRF.
        plot : bool
            Boolean to create diagnostic plots.
        save : bool
            Boolean to save models and figures.
        hdr : dict
            Header dictionary of the FFI file.
        img : numpy.ndarray
            Original FFI flux image in electros / sec.
        wcs : astropy.WCS
            Object with the WCS solution of the image.
        col_2d : numpy.ndarray
            Data array with the pixel column number in 2D
        row_2d : numpy.ndarray
            Data array with the pixel row number in 2D
        ra_2d : numpy.ndarray
            Data array with the pixel Right Ascension value in 2D, in degs
        dec_2d : numpy.ndarray
            Data array with the pixel Declination value in 2D, in degs
        flux_2d : numpy.ndarray
            Data array with the flux value of each pixel with substracted background
            in 2D, in electros / sec.
        sources : pandas.DataFrame
            Catalog with Gaia sources observed in the image, after cleaning.
        flux : numpy.ndarray
            Data array with the flux value of each pixel with substracted background
            in 1D after removing saturated & bright pixels. in electros / sec.
        flux_err : numpy.ndarray
            Data array with the flux error value of each pixel with substracted
            background in 1D after removing saturated & bright pixels. in electros / sec.
        col : numpy.ndarray
            Data array with the pixel column number in 1D after removing saturated &
            bright pixels
        row : numpy.ndarray
            Data array with the pixel row number in 1D after removing saturated &
            bright pixels
        nsurces : int
            Total number of sources observed in the image after cleaning.
        npixels : int
            Total number of pixels in the image
        gf : numpy.ndarray
            Data array with the Gaia flux value for every source.
        dflux : scipy.sparse.csr_matrix
            Sparse matrix with pixel flux value within r < 7 pixels of the source
            coordinates. Has shape [nsources , npixels]
        dx : scipy.sparse.csr_matrix
            Sparse matrix with distance between the pixel within r < 7 pixels and the
            source location, in pixel units. Has shape [nsources , npixels]
        dy : scipy.sparse.csr_matrix
            Sparse matrix with distance between the pixel within r < 7 pixels and the
            source location, in pixel units. Has shape [nsources , npixels]
        r : scipy.sparse.csr_matrix
            Sparse matrix with radial distance between pixels within r < 7 and the
            source location, in polar coordinates. Has shape [nsources , npixels]
        phi : scipy.sparse.csr_matrix
            Sparse matrix with angle value of pixels within r < 7 and the
            source location, in polar coordinates. Has shape [nsources , npixels]
        """

        self.channel = channel
        self.plot = plot
        self.save = save
        self.show = False

        if quarter is not None and quarter in np.arange(18):
            fname = "%s/data/fits/ffi/%s" % (DATAOUTDIR, quarter_ffi[quarter][0])
        elif len(ffi_name) == 17 and ffi_name[:4] == "kplr":
            fname = "%s/data/fits/ffi/%s_ffi-cal.fits" % (DATAOUTDIR, ffi_name)
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
        if ra_q[0] > 360 or np.abs(dec_q[0]) > 90 or rad[0] > 5:
            raise ValueError(
                "Query values are out of bound, please check WCS solution."
            )

        # remove border Pixels
        self.col_2d = col_2d[r_min:r_max, c_min:c_max] - c_min
        self.row_2d = row_2d[r_min:r_max, c_min:c_max] - r_min
        self.ra_2d = ra_2d[r_min:r_max, c_min:c_max]
        self.dec_2d = dec_2d[r_min:r_max, c_min:c_max]
        flux_2d = self.img[r_min:r_max, c_min:c_max]

        sources = self._do_big_query(self.ra_2d, self.dec_2d, time.jyear)
        sources["col"], sources["row"] = self.wcs.all_world2pix(
            sources.loc[:, ["ra", "dec"]].values, 0.5
        ).T

        # correct col,row columns for gaia sources
        sources.row -= r_min
        sources.col -= c_min

        # clean out-of-ccd and blended sources
        clean_sources = self._clean_source_list(sources)
        del sources

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
            print("Saturated pixels %i: " % (np.sum(~non_sat_mask)))
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
            print("Bright pixels %i: " % (np.sum(~bright_mask)))
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

        print("Total Gaia sources %i: " % (clean_sources.shape[0]))

        self.sources = clean_sources
        self.flux = flux
        self.flux_err = np.sqrt(np.abs(self.flux))
        self.col = col
        self.row = row
        self.nsurces = clean_sources.shape[0]
        self.npixels = self.flux.shape[0]

        self.rmin = 0.25
        self.rmax = 3.0

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

        if not os.path.isdir("%s/data/fits/ffi" % (DATAOUTDIR)):
            os.makedirs("%s/data/fits/ffi" % (DATAOUTDIR))

        out = "%s/data/fits/ffi/%s" % (DATAOUTDIR, fits_name)
        wget.download("%s/%s" % (url, fits_name), out=out)

        return

    def _do_big_query(self, ra, dec, epoch):
        """
        Query Gaia catalogs (EDR3 default) to obtain sources observed in the FFI.
        If query finishs ok, result will be saved for future use in the following
        directory:
            ../data/catalogs/ffi/<quarter#>/channel_<channel#>_gaia_xmatch.csv

        It does nx*ny small queries to avoid TimeoutError that might happen when doing
        large (rad > 0.7 deg) queries to Gaia archive. The ouput file has unique
        objects.

        Parameters
        ----------
        ra : list
            Value of the Right Ascension coordinate used for the query, in deg.
        dec : list
            Value of the Declination coordinate used for the query, in deg.
        epoch : float
            Year of the observation (Julian year) used for proper motion correction.

        Returns
        -------
        sources : pandas.DataFrame
            Clean catalog
        """
        file_name = "%s/data/catalogs/ffi/%s/channel_%i_gaia_xmatch.csv" % (
            DATAOUTDIR,
            str(self.quarter),
            self.channel,
        )
        if os.path.isfile(file_name):
            print("Loading query from file...")
            print(file_name)
            sources = pd.read_csv(file_name).drop("Unnamed: 0", axis=1)
        else:
            # number of cells in the grid to divide the image
            nx = 4
            ny = 4
            stepx = int(self.ra_2d.shape[1] / nx)
            stepy = int(self.ra_2d.shape[0] / ny)
            sources = []
            for x in range(1, nx + 1):
                for y in range(1, ny + 1):
                    ra_cell = self.ra_2d[
                        (y - 1) * stepy : y * stepy, (x - 1) * stepx : x * stepx
                    ]
                    dec_cell = self.dec_2d[
                        (y - 1) * stepy : y * stepy, (x - 1) * stepx : x * stepx
                    ]

                    ra_q = np.mean(ra_cell)
                    dec_q = np.mean(dec_cell)
                    rad_q = np.hypot(
                        ra_cell - ra_cell.mean(), dec_cell - dec_cell.mean()
                    ).max()
                    print(
                        "Will do small queries query with this "
                        + "(ra, dec, radius, epoch): ",
                        ra_q,
                        dec_q,
                        rad_q,
                        epoch,
                    )
                    result = get_gaia_sources(
                        tuple([ra_q]),
                        tuple([dec_q]),
                        tuple([rad_q]),
                        magnitude_limit=18,
                        epoch=epoch,
                        dr=3,
                    )
                    sources.append(result)
            sources = pd.concat(sources, axis=0).drop_duplicates(subset=["designation"])
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

            if not os.path.isdir(
                "%s/data/catalogs/ffi/%s" % (DATAOUTDIR, str(self.quarter))
            ):
                os.makedirs("%s/data/catalogs/ffi/%s" % (DATAOUTDIR, str(self.quarter)))
            sources.to_csv(file_name)
        return sources

    def _clean_source_list(self, sources):
        """
        Function to clean surces from the catalog removing sources near the borders,
        with 10 pixel tolerance, and to remove blended sources (within 8")

        Parameters
        ----------
        sources : pandas.DataFrame
            Catalog with sources to be removed

        Returns
        -------
        sources : pandas.DataFrame
            Clean catalog
        """

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
        # remove sources closer than 8" = 2 pix
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

        Parameters
        ----------
        data : numpy.ndarray
            Data arra with the pixel flux values.
        mask : numpy.ndarray
            Boolean array to mask pixels with sources.

        Returns
        -------
        background : numpy.ndarray
            Data array with background model
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
        """
        Finds and removes saturated pixels, including bleed columns.

        Parameters
        ----------
        flux : numpu.ndarray
            Data array with pixel flux value
        column : numpy.ndarray
            Data array with pixel column value
        row : numpy.ndarray
            Data array with pixel row value
        saturation_limit : foat
            Saturation limit at which pixels are removed.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
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
        """
        Finds and removes halos produced by bright stars (<10 mag)

        Parameters
        ----------
        flux : numpu.ndarray
            Data array with pixel flux value
        column : numpy.ndarray
            Data array with pixel column value
        row : numpy.ndarray
            Data array with pixel row value
        sources : pandas.DataFrame
            Catalog wih observed sources in the image
        mag_limit : foat
            Magnitude limit at which bright sources are identified.

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with rejected pixels
        """
        bright_mask = sources["phot_g_mean_mag"] <= mag_limit
        mask_radius = 30  # Pixels

        mask = [
            np.hypot(column - s.col, row - s.row) < mask_radius
            for _, s in sources[bright_mask].iterrows()
        ]
        mask = np.array(mask).sum(axis=0) > 0

        return mask

    def _create_sparse(self):
        """
        Function to create sparse matrces (scipy.sparse.csr_matrix) for variables:
        dx, dy, dflux, dfluxerr, r, and phy
        This is extremelly necessary for FFI due to the large number of sources (~10k)
        and pixels (~1.1M). The sparse matrices contain the pixel data around the
        sources up to 7 pixels distance from the object location.
        """
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
        plot=False,
    ):
        """
        Find the pixel mask that identifies pixels with contributions from ANY NUMBER
        of Sources.
        Fits a simple polynomial model to the log of the pixel flux values, in radial
        dimension and source flux, to find the optimum circular apertures for every
        source.

        Parameters
        ----------
        upper_radius_limit : float
            The radius limit at which we assume there is no flux from a source of any
            brightness (arcsec).
        lower_radius_limit : float
            The radius limit at which we assume there is flux from a source of any
            brightness (arcsec).
        flux_cut_off : float
            The flux at which we assume a source is too faint to model
        dm_type : string
            Type of design matrix to be used for modeling. Default is `rf-quadratic`,
            which is quadratic in both radius and flux.
        plot : bool
            Whether to show diagnostic plot. Default is False.
        """
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

        # remove pixels outside the radius limit
        source_mask = []
        for s in range(self.r.shape[0]):
            nonz_idx = self.r[s].nonzero()
            rad_mask = self.r[s].data < self.radius[s]
            aux = sparse.csr_matrix(
                (
                    self.r[s].data[rad_mask],
                    (nonz_idx[0][rad_mask], nonz_idx[1][rad_mask]),
                ),
                shape=self.r[s].shape,
            ).astype(bool)
            source_mask.append(aux)
        source_mask = sparse.vstack(source_mask, "csr")
        self.source_mask = source_mask

        if plot:
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
            if not self.show:
                fig_name = "%s/data/figures/%s/channel_%02i_psf_edge_model_%s.png" % (
                    DATAOUTDIR,
                    str(self.quarter),
                    self.channel,
                    dm_type,
                )
                if not os.path.isdir("%s/data/figures/%i" % (DATAOUTDIR, self.quarter)):
                    os.makedirs("%s/data/figures/%i" % (DATAOUTDIR, self.quarter))

                plt.savefig(fig_name, format="png", bbox_inches="tight")
                plt.close()
                return

            plt.show()

        return

    def _get_uncontaminated_source_mask(self):
        """
        creates a mask of shape nsources x npixels where targets are not contaminated.
        This mask is used to select pixels to build the PSF model.
        """

        warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self.uncontaminated_source_mask = self.source_mask.multiply(
            self.source_mask.sum(axis=0) == 1
        ).tocsr()
        self.uncontaminated_source_mask.eliminate_zeros()

    # @profile
    def _build_prf_shape(self, n_r_knots=10, n_phi_knots=12, cut_r=1.5, flux_cut_off=1):
        """
        Builds a sparse model matrix of shape nsources x npixels to be used when
        fitting each source pixels to estimate its PSF photometry

        Parameters
        ----------
        n_r_knots : int
            Number of radial knots in the spline model.
        n_phi_knots : int
            Number of azimuthal knots in the spline model.
        cut_r : int
            Distance at which the spline Design matrix has only dependency in the
            radial axis.
        flux_cut_off: float
            The flux in COUNTS at which to stop evaluating the model.
        """

        flux_estimates = self.gf
        self.n_r_knots = n_r_knots
        self.n_phi_knots = n_phi_knots
        self.cut_r = cut_r

        # mean flux values using uncontaminated mask and normalized by flux estimations
        mean_f = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.dflux)
            .multiply(1 / flux_estimates[:, None])
            .data
        )
        mean_f_err = np.abs(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.dflux_err)
            .multiply(1 / flux_estimates[:, None])
            .data
        )
        phi_b = self.uncontaminated_source_mask.multiply(self.phi).data
        r_b = self.uncontaminated_source_mask.multiply(self.r).data

        # build a design matrix A with b-splines basis in radius and angle axis.
        try:
            A = _make_A_polar(
                phi_b.ravel(),
                r_b.ravel(),
                cut_r=self.cut_r,
                rmin=self.rmin,
                rmax=self.rmax,
                n_r_knots=self.n_r_knots,
                n_phi_knots=self.n_phi_knots,
            )
        except ValueError:
            A = _make_A_polar(
                phi_b.ravel(),
                r_b.ravel(),
                cut_r=self.cut_r,
                rmin=self.rmin,
                rmax=np.percentile(r_b.ravel(), 98),
                n_r_knots=self.n_r_knots,
                n_phi_knots=self.n_phi_knots,
            )
            self.rmax = np.percentile(r_b.ravel(), 98)
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

        self.psf_w = psf_w

        # We evaluate our DM and build PSF models per source
        self._get_mean_model()
        # mean_model = mean_model.multiply(1 / mean_model.sum(axis=1))

        #  re-estimate source flux (from CH updates)
        prior_mu = flux_estimates
        prior_sigma = np.ones(self.mean_model.shape[0]) * 10 * flux_estimates

        X = self.mean_model.copy().T

        fmean = self.uncontaminated_source_mask.astype(float).multiply(self.dflux).data
        femean = (
            self.uncontaminated_source_mask.astype(float).multiply(self.dflux_err).data
        )

        ws, werrs = solve_linear_model(
            X,
            fmean,
            y_err=femean,
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

        self.source_mask = (
            self.mean_model.multiply(self.mean_model.T.dot(flux_estimates)).tocsr()
            > flux_cut_off
        )
        # rebuild uncontaminated_source_mask
        self._get_uncontaminated_source_mask()

        # set new rmax for spline basis
        self.rmax = np.minimum(
            self.rmax,
            np.percentile(self.uncontaminated_source_mask.multiply(self.r).data, 99.9),
        )
        self._get_mean_model()

        self.mean_flux = np.log10(
            self.uncontaminated_source_mask.astype(float)
            .multiply(self.dflux)
            .multiply(1 / flux_estimates[:, None])
            .data
        )

        print(
            "Total number of pixels data used for model fitting: ", self.mean_flux.shape
        )

        if self.save:
            self.save_model()

        if self.plot:
            self.plot_prf_shape()

        return

    def _get_mean_model(self):
        """
        Convenience function to make the scene PRF model
        """
        Ap = _make_A_polar(
            self.uncontaminated_source_mask.multiply(self.phi).data,
            self.uncontaminated_source_mask.multiply(self.r).data,
            rmin=self.rmin,
            rmax=self.rmax,
            cut_r=self.cut_r,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )

        # And create a `mean_model` that has the psf model for all pixels with fluxes
        mean_model = sparse.csr_matrix(self.r.shape)
        m = 10 ** Ap.dot(self.psf_w)
        m[~np.isfinite(m)] = 0
        mean_model[self.uncontaminated_source_mask] = m
        mean_model.eliminate_zeros()
        self.mean_model = mean_model
        self.design_matrix = Ap

        return

    def build_prf_model(self, n_r_knots=5, n_phi_knots=15):
        """
        Function that creates a PRF shape using the sources. In combines all other
        helping functions that build the source mask, remove contaminated pixels,
        estimate PRF edges, and create the final PRF model.

        For details see:
            `self._create_sparse()`
            `self._get_source_mask()`
            `self._get_uncontaminated_source_mask()`
            `self._build_prf_shape()`

        Default parameters where used for Martinez-Palomera et al. 2021.

        Parameters
        ----------
        n_r_knots : int
            Number of radial knots in the spline model.
        n_phi_knots : int
            Number of azimuthal knots in the spline model.
        """
        psf._create_sparse()
        psf._get_source_mask(
            upper_radius_limit=5,
            lower_radius_limit=1.1,
            flux_cut_off=50,
            dm_type="rf-quadratic",
        )
        psf._get_uncontaminated_source_mask()
        psf._build_prf_shape(
            n_r_knots=n_r_knots, n_phi_knots=n_phi_knots, flux_cut_off=1
        )

    def save_model(self, path=None):
        """
        Function to save the PRF model weights, number of knots for r and phy, and
        rmin and rmax to re-build the Design Matrix.
        The file is a csv table, that contain a multi-index column table. Rows are each
        channel, and columns are:
            ["n_r_knots", "n_phi_knots", "rmin", "rmax", ...prf_ws...]
        This file can be loaded as:
            pd.read_csv(fname, index_col=0, header=[0, 1])

        Note: models with different number of knots lead to different number of weights,
        and ins necessary to create separete files to preserve the esctructure.

        Parameters
        ----------
        path : string
            Path of the file
        """
        if path is None:
            fname = "%s/data/ffi_prf_models_v0.1.1.csv" % (PACKAGEDIR)
        else:
            fname = path

        arr_to_save = np.array(
            [self.n_r_knots, self.n_phi_knots, self.rmin, self.rmax]
            + self.psf_w.tolist()
        )

        if not os.path.isfile(fname):
            df_dict = {
                self.quarter: pd.DataFrame(
                    np.atleast_2d(arr_to_save),
                    index=[self.channel],
                    columns=["n_r_knots", "n_phi_knots", "rmin", "rmax"]
                    + ["w%02i" % i for i in range(1, 1 + len(self.psf_w))],
                )
            }
            df = pd.concat(df_dict, axis=1, keys=df_dict.keys())
            df.to_csv(fname)

        else:
            df = pd.read_csv(fname, index_col=0, header=[0, 1])

            if str(self.quarter) in df.columns.levels[0]:
                if self.channel in df.index:
                    if (
                        int(df.loc[self.channel, (str(self.quarter), "n_r_knots")])
                        != self.n_r_knots
                        or int(df.loc[self.channel, (str(self.quarter), "n_phi_knots")])
                        != self.n_phi_knots
                    ):
                        raise ValueError(
                            "Number of knots for r or phi in the file does not"
                            + "matches the number used in the current model. "
                            + "Create a new file for current model."
                        )
                df.loc[self.channel, str(self.quarter)] = arr_to_save
            else:
                df_dict = {
                    self.quarter: pd.DataFrame(
                        np.atleast_2d(arr_to_save),
                        index=[self.channel],
                        columns=["n_r_knots", "n_phi_knots", "rmin", "rmax"]
                        + ["b%02i" % i for i in range(1, 1 + len(self.psf_w))],
                    )
                }
                df_new = pd.concat(df_dict, axis=1, keys=df_dict.keys())
                df = pd.concat([df, df_new], axis=1)

            # df.to_csv(fname)

        return

    def save_model_retro(self, path=None):
        """
        Function to save the PRF model as a pickle file.
        Depricated.

        Parameters
        ----------
        path : string
            Path of the file
        """
        model_data = dict(
            psf_w=self.psf_w,
            A=self.design_matrix,
            x_data=self.uncontaminated_source_mask.multiply(self.dx).data,
            y_data=self.uncontaminated_source_mask.multiply(self.dy).data,
            f_data=self.mean_flux,
            f_model=np.log10(self.mean_model.data),
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )

        if self.save:
            if path is None:
                output = "%s/data/models/%i/channel_%02i_psf_model.pkl" % (
                    DATAOUTDIR,
                    self.quarter,
                    self.channel,
                )
                if not os.path.isdir("%s/data/models/%i" % (DATAOUTDIR, self.quarter)):
                    os.makedirs("%s/data/models/%i" % (DATAOUTDIR, self.quarter))
            else:
                output = path
            with open(output, "wb") as file:
                pickle.dump(model_data, file)
        return

    def plot_prf_shape(self):
        """
        Function to plot the PRF model in Cartesian and Polar coordinates
        """
        ylim = self.uncontaminated_source_mask.multiply(self.r).data.max() * 1.1
        vmin = -3
        vmax = -0.5

        phy = self.uncontaminated_source_mask.multiply(self.phi).data
        r = self.uncontaminated_source_mask.multiply(self.r).data
        x = self.uncontaminated_source_mask.multiply(self.dx).data
        y = self.uncontaminated_source_mask.multiply(self.dy).data

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].set_title("Mean flux")
        cax = ax[0, 0].scatter(
            phy,
            r,
            c=self.mean_flux,
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
            phy,
            r,
            c=np.log10(self.mean_model.data),
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
            x,
            y,
            c=self.mean_flux,
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
            x,
            y,
            c=np.log10(self.mean_model.data),
            marker=".",
            s=2,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        fig.colorbar(cax, ax=ax[1, 1])
        ax[1, 1].set_xlabel("dx")

        if not self.show:
            fig_name = "%s/data/figures/%s/channel_%02i_psf_model.png" % (
                DATAOUTDIR,
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir(
                "%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter))
            ):
                os.makedirs("%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter)))
            plt.savefig(fig_name, format="png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # @profile
    def fit_model(self):
        """
        Function to fit the PRF model and do LFD photometry for the sources observed in
        the FFIs.
        """
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
        """
        Function to save the Photometry Catalog of FFI sources
        """
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

        if not os.path.isdir("%s/data/catalogs/ffi/source_catalog/" % (DATAOUTDIR)):
            os.makedirs("%s/data/catalogs/ffi/source_catalog/" % (DATAOUTDIR))
        df.to_csv(
            "%s/data/catalogs/ffi/source_catalog/channel_%s_source_catalog_mjd_%s.csv"
            % (DATAOUTDIR, self.channel, str(self.hdr["MJDSTART"]))
        )

    def plot_image(self, ax=None, sources=False):
        """
        Function to plot the Full Frame Image and the Gaia Sources

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned
        sources : boolean
            Whether to overplot or not the source catalog

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
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
            fig_name = "%s/data/figures/%s/channel_%02i_ffi_image.png" % (
                DATAOUTDIR,
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir(
                "%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter))
            ):
                os.makedirs("%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter)))
            plt.savefig(fig_name, format="png", bbox_inches="tight")

        return ax

    def plot_pixel_masks(self, ax=None):
        """
        Function to plot the mask used to reject saturated and bright pixels

        Parameters
        ----------
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
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
            fig_name = "%s/data/figures/%s/channel_%02i_ffi_pixel_mask.png" % (
                DATAOUTDIR,
                str(self.quarter),
                self.channel,
            )
            if not os.path.isdir(
                "%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter))
            ):
                os.makedirs("%s/data/figures/%s" % (DATAOUTDIR, str(self.quarter)))
            plt.savefig(fig_name, format="png", bbox_inches="tight")

        return ax
