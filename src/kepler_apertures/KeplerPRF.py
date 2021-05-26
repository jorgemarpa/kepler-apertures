"""
Defines the object class that uses a Kepler PRF model to compute apertures and its
metrics
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches
from astropy.io import fits

from . import PACKAGEDIR, DATAOUTDIR
from .utils import _make_A_polar

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)


class KeplerPRF(object):
    """
    Class to load PRF models computed from FFI, to create photometric apertures
    """

    def __init__(
        self,
        prf_ws: np.array,
        n_r_knots: int = 5,
        n_phi_knots: int = 15,
        rmin: float = 0.25,
        rmax: float = 5,
    ):
        """
        A KeplerPRF object is build by providing the hyperparameters of the spline
        model, and the weights of each basis spline. The hyperparameters allow to
        reconstruct the same basis splines while the weights are used at evaluation of
        the model in new data.

        Parameters
        __________
        prf_ws : numpy.ndarray
            Weights corresponding to each basis of the design matrix.
        rmin : float
            The minimum radius for the PRF model to be fit.
        rmax : float
            The maximum radius for the PRF model to be fit.
        n_r_knots : int
            Number of radial knots in the spline model.
        n_phi_knots : int
            Number of azimuthal knots in the spline model.

        Attributes
        ----------
        prf_w : numpy.ndarray
            Weights corresponding to each basis of the design matrix.
        rmin : float
            The minimum radius for the PRF model to be fit.
        rmax : float
            The maximum radius for the PRF model to be fit.
        n_r_knots : int
            Number of radial knots in the spline model.
        n_phi_knots : int
            Number of azimuthal knots in the spline model.
        """

        self.prf_ws = prf_ws
        self.rmin = rmin
        self.rmax = rmax
        self.n_r_knots = n_r_knots
        self.n_phi_knots = n_phi_knots

    @staticmethod
    def load_from_file(
        quarter: int = 5,
        channel: int = 1,
    ):
        """
        Loads a PRF model build from Kepler's FFI for a given quarter and channel.

        Note: the file with the PRF models is csv file with a multiindex pandas
        DataFrame, the FITS version is in development.

        Parameters
        ----------
        channel : int
            Channel number of the FFI to be used to model the PRF. Valid values are
            between 1 and 84.
        quarter : int
            Number of the quarter that will be used to model the PRF.
            Valid values are between 1 and 17.

        Returns
        -------
        KeplerPRF : KeplerPRF
            An object with the PRF model ready to be evaluated in new data.
        """
        # load PSF model
        fname = "%s/data/ffi_prf_models_v0.1.0.csv" % (PACKAGEDIR)
        if not os.path.isfile(fname):
            raise FileNotFoundError("No PSF files: ", fname)

        try:
            tab = pd.read_csv(fname, index_col=0, header=[0, 1])
            n_r_knots = int(tab.loc[channel, (str(quarter), "n_r_knots")])
            n_phi_knots = int(tab.loc[channel, (str(quarter), "n_phi_knots")])
            rmin = int(tab.loc[channel, (str(quarter), "rmin")])
            rmax = int(tab.loc[channel, (str(quarter), "rmax")])
            prf_ws = tab.loc[channel, str(quarter)].iloc[4:].values

        except KeyError:
            raise IOError(
                "Quarter %i and channel %i has no PRF model data" % (quarter, channel)
            )

        return KeplerPRF(prf_ws, n_r_knots, n_phi_knots, rmin, rmax)

    def evaluate_PSF(self, dx, dy):
        """
        Function to evaluate the PRF model in a grid of data. THe function returns
        a the prediction of the model as normalized flux. The model is evaluated in
        pixels up to r < 7 from the location of the source.

        Parameters
        ----------
        dx : numpy.ndarray
            Distance between pixels (row direction) and source coordinates.
        dx : numpy.ndarray
            Distance between pixels (column direction) and source coordinates.

        Returns
        -------
        source_model: scipy.sparse.csr_matrix
            Normalized fluxvalues of the PRF model evaluation in the dx, dy grid
        """
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        source_mask = r <= np.floor(self.rmax)

        phi[phi >= np.pi] = np.pi - 1e-6

        try:
            dm = _make_A_polar(
                phi[source_mask].ravel(),
                r[source_mask].ravel(),
                rmin=self.rmin,
                rmax=self.rmax,
                n_r_knots=self.n_r_knots,
                n_phi_knots=self.n_phi_knots,
            )
        except ValueError:
            dm = _make_A_polar(
                phi[source_mask].ravel(),
                r[source_mask].ravel(),
                rmin=np.percentile(r[source_mask].ravel(), 1),
                rmax=np.percentile(r[source_mask].ravel(), 99),
                n_r_knots=self.n_r_knots,
                n_phi_knots=self.n_phi_knots,
            )

        source_model = sparse.csr_matrix(r.shape)
        m = 10 ** dm.dot(self.prf_ws)
        source_model[source_mask] = m
        source_model.eliminate_zeros()
        # psf_models = source_model.multiply(1 / source_model.sum(axis=1)).tocsr()

        return source_model

    def diagnose_metrics(self, psf_models, idx=0, ax=None, plot=True):
        """
        Function to evaluate the flux metrics for a single source as a function of
        the parameter that controls the aperture size.
        The flux metrics are computed by taking into account the PSF models of
        neighbor sources.

        This function is meant to be used only to generate the diagnostic or as a
        helping function of `optimize_aperture()` to precalculate the values of the
        metrics and find the optimal aperture in case of isolated sources, where the
        optimal is the full aperture.

        Parameters
        ----------
        psf_models : scipy.sparse.csr_matrix
            Sparse matrix with the PSF models of all sources in the field. It has shape
            of [n_sources, n_pixels]
        idx : int
            Index of the source for which the metrcs will be computed. Has to be a
            number between 0 and psf_models.shape[0].
        ax : matplotlib.axes
            Axis to be used to plot the figure
        plot : boolean
            Plot the metrics values.

        Returns
        -------
        ax : matplotlib.axes
            Figure axes
        """
        compl, crowd, cut = [], [], []
        for p in range(0, 101, 1):
            cut.append(p)
            mask = (
                psf_models[idx] >= np.percentile(psf_models[idx].data, p)
            ).toarray()[0]
            crowd.append(self.compute_CROWDSAP(psf_models, mask, idx))
            compl.append(self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask))
        self.compl = np.array(compl)
        self.crowd = np.array(crowd)
        self.cut = np.array(cut)

        if plot:
            if ax is None:
                fig, ax = plt.subplots(1)
            ax.plot(self.cut, self.compl, label=r"FLFRCSAP")
            ax.plot(self.cut, self.crowd, label=r"CROWDSAP")
            ax.set_xlabel("Percentile")
            ax.set_ylabel("Metric")
            ax.legend()

            return ax

    def create_aperture_mask(self, psf_models, percentile=0, idx=None):
        """
        Function to create the aperture mask of a given source for a given aperture
        size. This function can compute aperutre mask for one or all sources available
        in the psf_models

        Parameters
        ----------
        psf_models : scipy.sparse.csr_matrix
            Sparse matrix with the PSF models of all sources in the field. It has shape
            of [n_sources, n_pixels]
        percentile : float
            Percentile value that defines the isophote from the distribution of values
            in the psf model of the source
        idx : int
            Index of the source for which the metrcs will be computed. Has to be a
            number between 0 and psf_models.shape[0]. If None, then it computes the
            apertures for all sources in psf_models.

        Returns
        -------
        mask : numpy.ndarray
            Boolean array with the aperture mask.
        completeness : numpy.ndarray
            Flux metric indicating flux completeness for the selected aperture.
        crowdeness : numpy.ndarray
            Flux metric indicating flux contamination for the selected aperture.
        """
        if idx is not None:
            mask = (
                psf_models[idx] >= np.percentile(psf_models[idx].data, percentile)
            ).toarray()[0]

            # recompute metrics for optimal mask
            complet = self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask)
            crowd = self.compute_CROWDSAP(psf_models, mask, idx)

            return mask, complet, crowd
        else:
            masks, completeness, crowdeness = [], [], []
            for idx in range(psf_models.shape[0]):
                mask = (
                    psf_models[idx] >= np.percentile(psf_models[idx].data, percentile)
                ).toarray()[0]
                masks.append(mask)
                completeness.append(
                    self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask)
                )
                crowdeness.append(self.compute_CROWDSAP(psf_models, mask, idx))

            return np.array(masks), np.array(completeness), np.array(crowdeness)

    def optimize_aperture(
        self, psf_models, idx=0, target_complet=0.9, target_crowd=0.9, max_iter=100
    ):
        """
        Function to optimize the aperture mask for a given source. There are two
        special cases:
            * Isolated sources, the optimal aperture is the full aperture.
            * If optimizing for one single metric.
        For these last two case, no actual optimization if performed, and we use the
        results from `diagnose_metrics()`.

        The optimization is done using scipy Brent's algorithm and it uses a custom
        loss function that uses a Leaky ReLU term to achive the target value for
        both metrics.

        Parameters
        ----------
        psf_models : scipy.sparse.csr_matrix
            Sparse matrix with the PSF models of all sources in the field. It has shape
            of [n_sources, n_pixels]
        idx : int
            Index of the source for which the metrcs will be computed. Has to be a
            number between 0 and psf_models.shape[0]. If None, then it computes the
            apertures for all sources in psf_models.
        target_complet : float
            Value of the target completeness metric.
        target_crowd : float
            Value of the target crowdeness metric.
        max_iter : int
            Numer of maximum iterations to be performed by the optimizer.

        Returns
        -------
        mask : numpy.ndarray
            Boolean array with the aperture mask.
        completeness : float
            Flux metric indicating flux completeness for the selected aperture.
        crowdeness : float
            Flux metric indicating flux contamination for the selected aperture.
        optimal_percentile : float
            Percentile of the normalized flux distribution that defines the isophote.
        """
        # Do special cases when optimizing for only one metric
        self.diagnose_metrics(psf_models, idx=idx, plot=False)
        if target_complet < 0 and target_crowd > 0:
            optim_p = self.cut[np.argmax(self.crowd)]
        elif target_crowd < 0 and target_complet > 0:
            optim_p = self.cut[np.argmax(self.compl)]
        # for isolated sources, only need to optimize for completeness, in case of
        # asking for 2 metrics
        elif target_complet > 0 and target_crowd > 0 and all(self.crowd > 0.99):
            optim_p = self.cut[np.argmax(self.compl)]
        else:
            optim_params = {
                "percentile_bounds": [5, 95],
                "target_complet": target_complet,
                "target_crowd": target_crowd,
                "max_iter": max_iter,
                "psf_models": psf_models,
                "idx": idx,
            }
            minimize_result = minimize_scalar(
                self._goodness_metric_obj_fun,
                method="Bounded",
                bounds=[5, 95],
                options={"maxiter": max_iter, "disp": False},
                args=(optim_params),
            )
            optim_p = minimize_result.x

        mask = (
            psf_models[idx] >= np.percentile(psf_models[idx].data, optim_p)
        ).toarray()[0]

        # recompute metrics for optimal mask
        complet = self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask)
        crowd = self.compute_CROWDSAP(psf_models, mask, idx)
        return mask, complet, crowd, optim_p

    def _goodness_metric_obj_fun(self, percentile, optim_params):
        """
        The objective function to minimize with scipy.optimize.minimize_scalar called
        during optimization of the photometric aperture.

        Parameters
        ----------
        percentile : int
            Percentile of the normalized flux distribution that defines the isophote.
        optim_params : dictionary
            Dictionary with the variables needed for evaluate the metric:
                psf_models
                idx
                target_complet
                target_crowd

        Returns
        -------
        penalty : int
            Value of the objective function to be used for optiization.
        """
        psf_models = optim_params["psf_models"]
        idx = optim_params["idx"]
        # Find the value where to cut
        cut = np.percentile(psf_models[idx].data, int(percentile))
        # create "isophot" mask with current cut
        mask = (psf_models[idx] > cut).toarray()[0]

        # Do not compute and ignore if target score < 0
        if optim_params["target_complet"] > 0:
            completMetric = self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask)
        else:
            completMetric = 1.0

        # Do not compute and ignore if target score < 0
        if optim_params["target_crowd"] > 0:
            crowdMetric = self.compute_CROWDSAP(psf_models, mask, idx)
        else:
            crowdMetric = 1.0

        # Once we hit the target we want to ease-back on increasing the metric
        # However, we don't want to ease-back to zero pressure, that will
        # unconstrain the penalty term and cause the optmizer to run wild.
        # So, use a "Leaky ReLU"
        # metric' = threshold + (metric - threshold) * leakFactor
        leakFactor = 0.01
        if (
            optim_params["target_complet"] > 0
            and completMetric >= optim_params["target_complet"]
        ):
            completMetric = optim_params["target_complet"] + 0.001 * (
                completMetric - optim_params["target_complet"]
            )

        if (
            optim_params["target_crowd"] > 0
            and crowdMetric >= optim_params["target_crowd"]
        ):
            crowdMetric = optim_params["target_crowd"] + 0.1 * (
                crowdMetric - optim_params["target_crowd"]
            )

        penalty = -(completMetric + 10 * crowdMetric)

        return penalty

    # def plot_mean_PSF(self, ax=None):
    #     """
    #     Function to plot the PRF model as created from the FFI. This is only for
    #     illustration purposes.
    #
    #     Parameters
    #     ----------
    #     ax : matplotlib.axes
    #         Matlotlib axis can be provided, if not one will be created and returned
    #
    #     Returns
    #     -------
    #     ax : matplotlib.axes
    #         Matlotlib axis with the figure
    #     """
    #     if not hasattr(self, "x_data"):
    #         raise AttributeError("Class doesn't have attributes to plot PSF model")
    #
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    #     vmin = -0.5
    #     vmax = -3
    #     cax = ax[0].scatter(
    #         self.x_data,
    #         self.y_data,
    #         c=self.f_data,
    #         marker=".",
    #         s=2,
    #         vmin=vmin,
    #         vmax=vmax,
    #     )
    #     fig.colorbar(cax, ax=ax[0])
    #     ax[0].set_title("Data mean flux")
    #     ax[0].set_ylabel("dy")
    #     ax[0].set_xlabel("dx")
    #
    #     cax = ax[1].scatter(
    #         self.x_data,
    #         self.y_data,
    #         c=self.f_model,
    #         marker=".",
    #         s=2,
    #         vmin=vmin,
    #         vmax=vmax,
    #     )
    #     fig.colorbar(cax, ax=ax[1])
    #     ax[1].set_title("Average PSF Model")
    #     ax[1].set_xlabel("dx")
    #
    #     return ax

    def plot_aperture(self, flux, mask=None, ax=None, log=False):
        """
        Function to plot the photometric aperture for a given source.

        Parameters
        ----------
        flux : numpy.ndarray
            Data array with the flux image.
        mask : numpy.ndarray
            Boolean array with the aperture mask
        log : boolean
            Plot the image in log or linear scale.
        ax : matplotlib.axes
            Matlotlib axis can be provided, if not one will be created and returned

        Returns
        -------
        ax : matplotlib.axes
            Matlotlib axis with the figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5, 5))

        pc = ax.pcolor(
            flux,
            shading="auto",
            norm=colors.LogNorm() if log else None,
        )
        plt.colorbar(pc, label="", fraction=0.038, ax=ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("")
        if mask is not None:
            for i in range(flux.shape[0]):
                for j in range(flux.shape[1]):
                    if mask[i, j]:
                        rect = patches.Rectangle(
                            xy=(j, i),
                            width=1,
                            height=1,
                            color="red",
                            fill=False,
                            hatch="",
                        )
                        ax.add_patch(rect)
            zoom = np.argwhere(mask == True)
            ax.set_ylim(
                np.maximum(0, zoom[0, 0] - 3),
                np.minimum(zoom[-1, 0] + 3, flux.shape[0]),
            )
            ax.set_xlim(
                np.maximum(0, zoom[0, -1] - 3),
                np.minimum(zoom[-1, -1] + 3, flux.shape[1]),
            )
        else:
            ax.set_xlim(np.argmax(flux))
            ax.set_ylim()

        return ax

    @staticmethod
    def compute_FLFRCSAP(psf_model, mask):
        """
        Compute fraction of target flux enclosed in the optimal aperture to total flux
        for a given source (flux completeness).
        Parameters
        ----------
        psf_model: numpy ndarray
            Array with the PSF model for the target source. It has shape [n_pixels]
        mask: boolean array
            Array of boolean indicating the aperture for the target source.

        Returns
        -------
        FLFRCSAP: float
            Completeness metric
        """
        return psf_model[mask].sum() / psf_model.sum()

    @staticmethod
    def compute_CROWDSAP(psf_models, mask, idx):
        """
        Compute the ratio of target flux relative to flux from all sources within
        the photometric aperture (i.e. 1 - Crowdeness).
        Parameters
        ----------
        psf_models: numpy ndarray
            Array with the PSF models for all targets in the cutout. It has shape
            [n_sources, n_pixels].
        mask: boolean array
            Array of boolean indicating the aperture for the target source.
        idx: int
            Index of the source to compute the metric. It has to be a number between
            0 and psf_models.shape[0].

        Returns
        -------
        CROWDSAP: float
            Crowdeness metric
        """
        ratio = (
            psf_models.multiply(1 / psf_models.sum(axis=0)).tocsr()[idx].toarray()[0]
        )
        return ratio[mask].sum() / mask.sum()
