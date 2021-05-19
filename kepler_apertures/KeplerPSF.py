import os
import warnings

import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches

from .utils import solve_linear_model, _make_A_polar


r_min, r_max = 20, 1044
c_min, c_max = 12, 1112
remove_sat = True
mask_bright = True

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)


class KeplerPSF(object):
    def __init__(self, quarter=5, channel=1):

        # load PSF model
        fname = output = "../data/models/%i/channel_%02i_psf_model.pkl" % (
            quarter,
            channel,
        )
        print(fname)
        if os.path.isfile(fname):
            psf = pickle.load(open(fname, "rb"))
        else:
            raise FileNotFoundError("No PSF files")
        # load PSF edge model
        fname = output = "../data/models/%i/channel_%02i_psf_edge_model_%s.pkl" % (
            quarter,
            channel,
            "rf-quadratic",
        )
        if os.path.isfile(fname):
            psf_edge = pickle.load(open(fname, "rb"))
        else:
            raise FileNotFoundError("No PSF edge file")

        self.DM = psf["A"]
        self.PSF_w = psf["psf_w"]
        self.x_data = psf["x_data"]
        self.y_data = psf["y_data"]
        self.f_data = psf["f_data"]  # in log
        self.rmin = psf["rmin"]
        self.rmax = psf["rmax"]
        self.n_r_knots = psf["n_r_knots"]
        self.n_phi_knots = psf["n_phi_knots"]

        self.r_data = np.hypot(self.x_data, self.y_data)
        self.phy_data = np.arctan2(self.y_data, self.x_data)

        self.f_model = self.DM.dot(self.PSF_w)  # in log

        self.psf_edge_model = psf_edge["polifit_results"]

    def evaluate_PSF(self, flux, dx, dy, gf):
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)

        r_lim = np.polyval(self.psf_edge_model, np.log10(gf)) * 3.0
        r_lim[r_lim < 1.1] = 1.1
        r_lim[r_lim > 7] = 7
        # source_mask = r < 6.0  # r_lim[:, None]
        source_mask = r <= r_lim[:, None]

        phi[phi >= np.pi] = np.pi - 1e-3

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

        mean_model = sparse.csr_matrix(r.shape)
        m = 10 ** dm.dot(self.PSF_w)
        mean_model[source_mask] = m
        # mean_model = mean_model.multiply(psf_mask).tocsr()
        mean_model.eliminate_zeros()
        # psf_models = mean_model.multiply(1 / mean_model.sum(axis=1)).tocsr()

        return mean_model

    def evaluate_PSF2(self, flux, flux_err, dx, dy, gf):
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)

        source_mask = r < 6.0

        phi[phi >= np.pi] = np.pi - 1e-4

        dm = _make_A_polar(
            phi[source_mask].ravel(),
            r[source_mask].ravel(),
            rmin=self.rmin,
            rmax=self.rmax,
            n_r_knots=self.n_r_knots,
            n_phi_knots=self.n_phi_knots,
        )

        mean_model = sparse.csr_matrix(r.shape)
        m = 10 ** dm.dot(self.PSF_w)
        m[~np.isfinite(m)] = 0
        mean_model[source_mask] = m
        mean_model.eliminate_zeros()

        prior_mu = gf
        prior_sigma = np.ones(mean_model.shape[0]) * 10 * gf

        f, fe = (flux).mean(axis=0), ((flux_err ** 2).sum(axis=0) ** 0.5) / (
            flux.shape[0]
        )
        X = mean_model.copy().T

        sigma_w_inv = X.T.dot(X.multiply(1 / fe[:, None] ** 2)).toarray()
        sigma_w_inv += np.diag(1 / (prior_sigma ** 2))
        B = X.T.dot((f / fe ** 2))
        B += prior_mu / (prior_sigma ** 2)
        ws = np.linalg.solve(sigma_w_inv, B)
        werrs = np.linalg.inv(sigma_w_inv).diagonal() ** 0.5

        # Rebuild source mask
        ok = np.abs(ws - gf) / werrs > 3
        ok &= ((ws / gf) < 10) & ((gf / ws) < 10)
        ok &= ws > 10
        ok &= werrs > 0

        gf[ok] = ws[ok]

        source_mask = mean_model.multiply(mean_model.T.dot(gf)).tocsr() > 1
        mean_model = mean_model.multiply(source_mask).tocsr()

        return mean_model, X.T

    def diagnose_metrics(self, psf_models, idx=0, ax=None, plot=True):
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
        # Do special cases when optimizing for only one metric
        self.diagnose_metrics(psf_models, idx=idx, plot=False)
        if target_complet < 0 and target_crowd > 0:
            optim_p = self.cut[np.argmax(self.crowd)]
        elif target_crowd < 0 and target_complet > 0:
            optim_p = self.cut[np.argmax(self.compl)]

        # for isolated sources, only need to optimize for completeness, in case of
        # asking for 2 metrics
        else:
            if target_complet > 0 and target_crowd > 0 and all(self.crowd > 0.99):
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

        # print("Optimize percentile %i" % optim_p)
        mask = (
            psf_models[idx] >= np.percentile(psf_models[idx].data, optim_p)
        ).toarray()[0]

        # recompute metrics for optimal mask
        complet = self.compute_FLFRCSAP(psf_models[idx].toarray()[0], mask)
        crowd = self.compute_CROWDSAP(psf_models, mask, idx)
        return mask, complet, crowd, optim_p

    def _goodness_metric_obj_fun(self, percentile, optim_params):
        """The objective function to minimize with
        scipy.optimize.minimize_scalar
        First sets the alpha regularization penalty then runs
        RegressionCorrector.correct and then computes the over- and
        under-fitting goodness metrics to return a scalar penalty term to
        minimize.
        Uses the paramaters in self.optimization_params.
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

    def plot_mean_PSF(self, ax=None):
        if not hasattr(self, "x_data"):
            raise AttributeError("Class doesn't have attributes to plot PSF model")

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        vmin = -0.5
        vmax = -3
        cax = ax[0].scatter(
            self.x_data,
            self.y_data,
            c=self.f_data,
            marker=".",
            s=2,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(cax, ax=ax[0])
        ax[0].set_title("Data mean flux")
        ax[0].set_ylabel("dy")
        ax[0].set_xlabel("dx")

        cax = ax[1].scatter(
            self.x_data,
            self.y_data,
            c=self.f_model,
            marker=".",
            s=2,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(cax, ax=ax[1])
        ax[1].set_title("Average PSF Model")
        ax[1].set_xlabel("dx")

        return ax

    def plot_aperture(self, flux, mask=None, ax=None, log=False):
        if ax is None:
            fig, ax = plt.subplots(1, figsize=(5, 5))

        pc = ax.pcolor(
            flux,
            shading="auto",
            norm=colors.SymLogNorm(linthresh=50, vmin=0, vmax=2000, base=10)
            if log
            else None,
        )
        plt.colorbar(pc, label="", fraction=0.038, ax=ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("PSF evaluation")
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
            Array with the PSF model for the target source
        mask: boolean array
            Array of boolean indicating the aperture for the target source
        Returns
        -------
        FLFRCSAP: float
            Completeness metric
        """
        return psf_model[mask].sum() / psf_model.sum()

    @staticmethod
    def compute_CROWDSAP(psf_models, mask, i):
        """
        Compute the ratio of target flux relative to flux from all sources within
        the photometric aperture (i.e. 1 - Crowdeness).
        Parameters
        ----------
        psf_models: numpy ndarray
            Array with the PSF models for all targets in the cutout
        mask: boolean array
            Array of boolean indicating the aperture for the target source
        i: int
            Index of the target source in axis = 0 of psf_models
        Returns
        -------
        CROWDSAP: float
            Crowdeness metric
        """
        ratio = psf_models.multiply(1 / psf_models.sum(axis=0)).tocsr()[i].toarray()[0]
        return ratio[mask].sum() / mask.sum()
