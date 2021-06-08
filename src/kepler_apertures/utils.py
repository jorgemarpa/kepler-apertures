"""
Collection of utility functions
"""

import numpy as np
import functools

from scipy import sparse
from patsy import dmatrix
from tqdm import tqdm
import pyia
import matplotlib.pyplot as plt

from astropy import units
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares


@functools.lru_cache()
def get_gaia_sources(ras, decs, rads, magnitude_limit=18, epoch=2020, dr=2):
    """
    Will find gaia sources using a TAP query, accounting for proper motions.

    Inputs have be hashable, e.g. tuples

    Parameters
    ----------
    ras : tuple
        Tuple with right ascension coordinates to be queried
        shape nsources
    decs : tuple
        Tuple with declination coordinates to be queried
        shape nsources
    rads : tuple
        Tuple with radius query
        shape nsources
    magnitude_limit : int
        Limiting magnitued for query
    epoch : float
        Year of the observation (Julian year) used for proper motion correction.
    dr : int
        Gaia Data Release to be used, DR2 or EDR3.

    Returns
    -------
    Pandas DatFrame with number of result sources (rows) and Gaia columns

    """
    if not hasattr(ras, "__iter__"):
        ras = [ras]
    if not hasattr(decs, "__iter__"):
        decs = [decs]
    if not hasattr(rads, "__iter__"):
        rads = [rads]

    wheres = [
        f"""1=CONTAINS(
                  POINT('ICRS',ra,dec),
                  CIRCLE('ICRS',{ra},{dec},{rad}))"""
        for ra, dec, rad in zip(ras, decs, rads)
    ]

    where = """\n\tOR """.join(wheres)
    if dr == 2:
        # CH: We don't need a lot of these columns we could greatly reduce it
        gd = pyia.GaiaData.from_query(
            f"""SELECT solution_id, designation, source_id, random_index, ref_epoch,
            coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error, parallax,
            parallax_error, parallax_over_error, pmra, pmra_error, pmdec, pmdec_error,
            ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, dec_parallax_corr,
            dec_pmra_corr, dec_pmdec_corr, parallax_pmra_corr, parallax_pmdec_corr,
            pmra_pmdec_corr, astrometric_n_obs_al, astrometric_n_obs_ac,
            astrometric_n_good_obs_al, astrometric_n_bad_obs_al, astrometric_gof_al,
            astrometric_chi2_al, astrometric_excess_noise, astrometric_excess_noise_sig,
            astrometric_params_solved, astrometric_primary_flag, astrometric_weight_al,
            astrometric_pseudo_colour, astrometric_pseudo_colour_error,
            mean_varpi_factor_al, astrometric_matched_observations,
            visibility_periods_used, astrometric_sigma5d_max, frame_rotator_object_type,
            matched_observations, duplicated_source, phot_g_n_obs, phot_g_mean_flux,
            phot_g_mean_flux_error, phot_g_mean_flux_over_error, phot_g_mean_mag,
            phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error,
            phot_bp_mean_flux_over_error, phot_bp_mean_mag, phot_rp_n_obs,
            phot_rp_mean_flux, phot_rp_mean_flux_error, phot_rp_mean_flux_over_error,
            phot_rp_mean_mag, phot_bp_rp_excess_factor, phot_proc_mode, bp_rp, bp_g, g_rp,
            radial_velocity, radial_velocity_error, rv_nb_transits, rv_template_teff,
            rv_template_logg, rv_template_fe_h, phot_variable_flag, l, b, ecl_lon, ecl_lat,
            priam_flags, teff_val, teff_percentile_lower, teff_percentile_upper, a_g_val,
            a_g_percentile_lower, a_g_percentile_upper, e_bp_min_rp_val,
            e_bp_min_rp_percentile_lower, e_bp_min_rp_percentile_upper, flame_flags,
            radius_val, radius_percentile_lower, radius_percentile_upper, lum_val,
            lum_percentile_lower, lum_percentile_upper, datalink_url, epoch_photometry_url,
            ra as ra_gaia, dec as dec_gaia FROM (
     SELECT *,
     EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
     FROM gaiadr2.gaia_source
     WHERE {where}
    )  AS subquery
    WHERE phot_g_mean_mag<={magnitude_limit}

    """
        )
    elif dr == 3:
        gd = pyia.GaiaData.from_query(
            f"""SELECT designation,
                    coord1(prop) AS ra, ra_error, coord2(prop) AS dec, dec_error,
                    parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error,
                    dr2_radial_velocity, dr2_radial_velocity_error,
                    ruwe, phot_g_n_obs, phot_g_mean_flux,
                    phot_g_mean_flux_error, phot_g_mean_mag,
                    phot_bp_n_obs, phot_bp_mean_flux, phot_bp_mean_flux_error,
                    phot_bp_mean_mag, phot_rp_n_obs,
                    phot_rp_mean_flux, phot_rp_mean_flux_error,
                    phot_rp_mean_mag FROM (
             SELECT *,
             EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, 0, ref_epoch, {epoch}) AS prop
             FROM gaiaedr3.gaia_source
             WHERE {where}
            )  AS subquery
            WHERE phot_g_mean_mag<={magnitude_limit}
            """
        )
    else:
        raise ValueError("Please pass a valid data release")
    return gd.data.to_pandas()


def make_A_edges(r, f, type="quadratic"):
    """
    Creates a design matrix to estimate the PSF edge (in pixels) as a function of the
    flux.

    Parameters
    ----------
    r : numpy ndarray
        Array with radii values
    f : numpy ndarray
        Array with flux values
    type: string
        Type of basis for the design matrix, default is quadratic in both
        radius and flux
    Returns
    -------
    A : numpy ndarray
        A design matrix
    """
    if type == "linear":
        A = np.vstack([r ** 0, r, f]).T
    elif type == "r-quadratic":
        A = np.vstack([r ** 0, r, r ** 2, f]).T
    elif type == "cubic":
        A = np.vstack([r ** 0, r, r ** 2, r ** 3, f]).T
    elif type == "exp":
        A = np.vstack([r ** 0, np.exp(-r), f]).T
    elif type == "inverse":
        A = np.vstack([r ** 0, 1 / r, f]).T
    elif type == "rf-quadratic":
        A = np.vstack(
            [
                r ** 0,
                r,
                r ** 2,
                r ** 0 * f,
                r * f,
                r ** 2 * f,
                r ** 0 * f ** 2,
                r * f ** 2,
                r ** 2 * f ** 2,
            ]
        ).T
    else:
        raise ValueError("Wrong desing matrix basis type")
    return A


def solve_linear_model(
    A, y, y_err=None, prior_mu=None, prior_sigma=None, k=None, errors=False
):
    """
    Solves a linear model with design matrix A and observations y:
        Aw = y
    return the solutions w for the system assuming Gaussian priors.
    Alternatively the observation errors, priors, and a boolean mask for the
    observations (row axis) can be provided.

    Adapted from Luger, Foreman-Mackey & Hogg, 2017
    (https://ui.adsabs.harvard.edu/abs/2017RNAAS...1....7L/abstract)

    Parameters
    ----------
    A : numpy ndarray or scipy sparce csr matrix
        Desging matrix with solution basis
        shape n_observations x n_basis
    y : numpy ndarray
        Observations
        shape n_observations
    y_err : numpy ndarray, optional
        Observation errors
        shape n_observations
    prior_mu : float, optional
        Mean of Gaussian prior values for the weights (w)
    prior_sigma : float, optional
        Standard deviation of Gaussian prior values for the weights (w)
    k : boolean, numpy ndarray, optional
        Mask that sets the observations to be used to solve the system
        shape n_observations

    Returns
    -------
    w : numpy ndarray
        Array with the estimations for the weights
        shape n_basis
    werrs : numpy ndarray
        Array with the error estimations for the weights, returned if y_err is
        provided
        shape n_basis
    """
    if k is None:
        k = np.ones(len(y), dtype=bool)

    if y_err is not None:
        sigma_w_inv = A[k].T.dot(A[k].multiply(1 / y_err[k, None] ** 2))
        B = A[k].T.dot((y[k] / y_err[k] ** 2))
    else:
        sigma_w_inv = A[k].T.dot(A[k])
        B = A[k].T.dot(y[k])

    if prior_mu is not None and prior_sigma is not None:
        sigma_w_inv += np.diag(1 / prior_sigma ** 2)
        B += prior_mu / prior_sigma ** 2

    if type(sigma_w_inv) == sparse.csr_matrix:
        sigma_w_inv = sigma_w_inv.toarray()

    if type(sigma_w_inv) == sparse.csc_matrix:
        sigma_w_inv = sigma_w_inv.toarray()

    if type(sigma_w_inv) == np.matrix:
        sigma_w_inv = np.asarray(sigma_w_inv)

    w = np.linalg.solve(sigma_w_inv, B)
    if errors is True:
        w_err = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5
        return w, w_err
    return w


def _make_A_polar(phi, r, cut_r=1.5, rmin=1, rmax=5, n_r_knots=12, n_phi_knots=15):
    """
    Makes a spline design matrix in polar coordinates
    Parameters
    ----------
    phi : numpy ndarray
    r : numpy ndarray
    cut_r : int
    rmin : float
        Minimum radius value for the array of knots
    rmax : float
        Maximum radius value for the array of knots
    n_r_knots : int
        Number of knots to used for the radius axis
    n_phi_knots : int
        Number of knots to used for the angle axis
    Returns
    -------
    x1 : sparse matrix
        Design matrix in polar coordinates using spline as base functions
    """
    # create the spline bases for radius and angle
    phi_spline = sparse.csr_matrix(wrapped_spline(phi, order=3, nknots=n_phi_knots).T)
    r_knots = np.linspace(rmin ** 0.5, rmax ** 0.5, n_r_knots) ** 2
    cut_r_int = np.where(r_knots <= cut_r)[0].max()
    r_spline = sparse.csr_matrix(
        np.asarray(
            dmatrix(
                "bs(x, knots=knots, degree=3, include_intercept=True)",
                {"x": list(r), "knots": r_knots},
            )
        )
    )
    # build full desing matrix
    X = sparse.hstack(
        [phi_spline.multiply(r_spline[:, idx]) for idx in range(r_spline.shape[1])],
        format="csr",
    )
    # find and remove the angle dependency for all basis for radius < 6
    cut = np.arange(0, phi_spline.shape[1] * cut_r_int)
    a = list(set(np.arange(X.shape[1])) - set(cut))
    X1 = sparse.hstack(
        [X[:, a], r_spline[:, 1:cut_r_int], sparse.csr_matrix(np.ones(X.shape[0])).T],
        format="csr",
    )
    return X1


def wrapped_spline(input_vector, order=2, nknots=10):
    """
    Creates a vector of folded-spline basis according to the input data. This is meant
    to be used to build the basis vectors for periodic data, like the angle in polar
    coordinates.

    Parameters
    ----------
    input_vector : numpy.ndarray
        Input data to create basis, angle values MUST BE BETWEEN -PI and PI.
    order : int
        Order of the spline basis
    nknots : int
         Number of knots for the splines

    Returns
    -------
    folded_basis : numpy.ndarray
        Array of folded-spline basis
    """

    if not ((input_vector > -np.pi) & (input_vector < np.pi)).all():
        raise ValueError("Must be between -pi and pi")
    x = np.copy(input_vector)
    x1 = np.hstack([x, x + np.pi * 2])
    nt = (nknots * 2) + 1

    t = np.linspace(-np.pi, 3 * np.pi, nt)
    dt = np.median(np.diff(t))
    # Zeroth order basis
    basis = np.asarray(
        [
            ((x1 >= t[idx]) & (x1 < t[idx + 1])).astype(float)
            for idx in range(len(t) - 1)
        ]
    )
    # Higher order basis
    for order in np.arange(1, 4):
        basis_1 = []
        for idx in range(len(t) - 1):
            a = ((x1 - t[idx]) / (dt * order)) * basis[idx]

            if ((idx + order + 1)) < (nt - 1):
                b = (-(x1 - t[(idx + order + 1)]) / (dt * order)) * basis[
                    (idx + 1) % (nt - 1)
                ]
            else:
                b = np.zeros(len(x1))
            basis_1.append(a + b)
        basis = np.vstack(basis_1)

    folded_basis = np.copy(basis)[: nt // 2, : len(x)]
    for idx in np.arange(-order, 0):
        folded_basis[idx, :] += np.copy(basis)[nt // 2 + idx, len(x) :]
    return folded_basis
