"""
Sparse jump model with pinned (always-kept) features.

The sparse jump model of Nystrup et al. (2021) weighs every feature by its between-cluster
sum of squares (BCSS) and lets a Lasso-like constraint ``‖w‖₁ ≤ κ`` push the weight of the
uninformative ones to zero, so that `--max-feats` (κ²) sets the effective number of features
kept. That is exactly what one wants for an exploratory feature set, but it also means a
feature the user considers essential -- the three features of Table 2 of Shu, Yu and Mulvey
(2024), say -- can be dropped at some re-estimations and not at others.

`PinnedSparseJumpModel` keeps a chosen subset of the features in the model at every
re-estimation. The pinned features are simply left out of the soft-thresholding step: they
carry their own BCSS weight, and only the remaining ("free") features are thresholded to
meet the sparsity budget. Pinning therefore guarantees *presence*, not importance -- a
pinned feature that separates the regimes poorly still ends up with a small weight, it just
never falls to exactly zero.
"""

import numpy as np
from numpy.linalg import norm

from jumpmodels.sparse_jump import (SparseJumpModel, binary_search_decrease, compute_BCSS,
                                    soft_thres_l2_normalized)
from jumpmodels.utils import check_2d_array, raise_arr_to_pd_obj, weighted_mean_cluster

# a pinned feature whose BCSS is (near) zero would carry no weight at all; it is floored at
# this fraction of the largest BCSS so that it stays in the model, however faintly
PIN_FLOOR = 1e-3


def solve_lasso_pinned(scores,
                       norm_ub: float,
                       pin_mask,
                       floor: float = PIN_FLOOR,
                       tol: float = 1e-8) -> np.ndarray:
    """
    Solve the feature-weight problem of the sparse jump model with some features pinned.

    The unpinned problem maximizes ``scores · w`` under ``‖w‖₂ ≤ 1``, ``‖w‖₁ ≤ norm_ub`` and
    ``w ≥ 0``; its solution soft-thresholds `scores` at the smallest threshold `t` meeting
    the L1 constraint, then normalizes (`jumpmodels.sparse_jump.solve_lasso`).

    Here `t` is found exactly as the unpinned solver finds it -- on the full score vector,
    by the same bisection -- and the pinned features are then exempted from it::

        y_j = max(scores_j, floor · max(scores))    for a pinned j
        y_j = max(scores_j - t, 0)                  for a free j
        w   = y / ‖y‖₂

    So the pinned features sit outside the selection and keep the weight their own BCSS
    earns them, while the free ones compete for the sparsity budget exactly as before. The
    features that come back in are the ones the threshold would have cut, so ``‖w‖₁`` ends
    up slightly above `norm_ub`: pinning buys presence at the cost of a small overshoot of
    the effective feature count, which is the honest price of the constraint.

    Thresholding the pinned features too and merely flooring them would keep the budget
    tighter, but it would also pin them at a weight so small that they no longer move the
    clustering -- presence in name only. The floor is therefore a last resort, reached only
    by a pinned feature whose BCSS is essentially zero.

    Parameters
    ----------
    scores : array-like
        The per-feature BCSS, normalized by its maximum as in `SparseJumpModel.fit`.

    norm_ub : float
        The upper bound on the L1 norm of the weights, i.e. kappa = sqrt(`max_feats`).

    pin_mask : array-like of bool
        True for the features to keep, aligned with `scores`.

    floor : float, optional (default=`PIN_FLOOR`)
        The smallest score a pinned feature is given, as a fraction of the largest score.
        It only bites for a pinned feature that separates the clusters not at all.

    tol : float, optional (default=1e-8)
        The tolerance of the threshold bisection, as in the unpinned solver.

    Returns
    -------
    np.ndarray
        The feature weights `w`, non-negative and of unit L2 norm. Every pinned entry is
        strictly positive.
    """
    scores = np.maximum(np.asarray(scores, dtype=float).ravel(), 0.)
    pin_mask = np.asarray(pin_mask, dtype=bool).ravel()
    if pin_mask.shape != scores.shape:
        raise ValueError(f"pin_mask 길이({pin_mask.size})가 피처 개수({scores.size})와 다릅니다.")
    scale = scores.max()
    if scale <= 0.:                      # no feature separates the clusters at all
        return np.ones_like(scores) / np.sqrt(scores.size)

    # the threshold of the unpinned problem, mirroring `jumpmodels.sparse_jump.solve_lasso`:
    # a bisection on a genuinely decreasing function, bounded by the second largest score
    distinct = np.unique(scores)
    threshold = 0.
    if distinct.size > 1 and distinct[-2] >= tol:
        l1_of = lambda thres: soft_thres_l2_normalized(scores, thres).sum()
        threshold = binary_search_decrease(l1_of, 0., distinct[-2], norm_ub, tol_x=tol)

    y = np.where(pin_mask,
                 np.maximum(scores, floor * scale),
                 np.maximum(scores - threshold, 0.))
    return y / norm(y)


class PinnedSparseJumpModel(SparseJumpModel):
    """
    Sparse jump model that never drops the features selected by `pin_mask`.

    Same estimator as `jumpmodels.sparse_jump.SparseJumpModel` -- same coordinate descent
    between the jump model fit and the feature weights, same attributes -- with the Lasso
    step replaced by `solve_lasso_pinned`. With an empty mask it reduces to the parent
    model up to the threshold search, so `rolling.init_model` only uses it when something
    is actually pinned.

    Parameters
    ----------
    pin_mask : array-like of bool
        True for the features to keep at every re-estimation, aligned with the columns of
        the feature matrix passed to `fit`.

    pin_floor : float, optional (default=`PIN_FLOOR`)
        The score floor of a pinned feature, see `solve_lasso_pinned`.

    Other parameters are those of `SparseJumpModel`.
    """

    def __init__(self,
                 pin_mask=None,
                 pin_floor: float = PIN_FLOOR,
                 n_components: int = 2,
                 max_feats: float = 100.,
                 jump_penalty: float = 0.,
                 cont: bool = False,
                 grid_size: float = 0.05,
                 mode_loss: bool = True,
                 random_state=0,
                 max_iter: int = 30,
                 tol_w: float = 1e-4,
                 max_iter_jm: int = 1000,
                 tol_jm: float = 1e-8,
                 n_init_jm: int = 10,
                 verbose: int = 0):
        super().__init__(n_components=n_components, max_feats=max_feats,
                         jump_penalty=jump_penalty, cont=cont, grid_size=grid_size,
                         mode_loss=mode_loss, random_state=random_state, max_iter=max_iter,
                         tol_w=tol_w, max_iter_jm=max_iter_jm, tol_jm=tol_jm,
                         n_init_jm=n_init_jm, verbose=verbose)
        self.pin_mask = pin_mask
        self.pin_floor = pin_floor

    def _check_pin_mask(self, n_features: int) -> np.ndarray:
        """Return the pin mask as a boolean array of length `n_features`."""
        if self.pin_mask is None:
            return np.zeros(n_features, dtype=bool)
        mask = np.asarray(self.pin_mask, dtype=bool).ravel()
        if mask.size != n_features:
            raise ValueError(
                f"고정 피처 마스크의 길이({mask.size})가 피처 개수({n_features})와 다릅니다.")
        return mask

    def fit(self, X, ret_ser=None, sort_by: str = "cumret"):
        """
        Fit the model by coordinate descent, keeping the pinned features in.

        This mirrors `SparseJumpModel.fit`: the jump model is fitted on the weighted data,
        the weights are re-optimized from the BCSS of the resulting clustering, and the two
        steps alternate until the weights stop moving. The single difference is that the
        weights come from `solve_lasso_pinned` instead of `jumpmodels.sparse_jump.solve_lasso`.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The feature matrix.

        ret_ser : pd.Series or np.ndarray, optional
            The return series used to sort the states.

        sort_by : str, optional (default="cumret")
            The sorting criterion, see `SparseJumpModel.fit`.

        Returns
        -------
        PinnedSparseJumpModel
            The fitted model.
        """
        X_arr = check_2d_array(X)
        self.n_features_all = X_arr.shape[1]
        pin_mask = self._check_pin_mask(self.n_features_all)
        jm = self.init_jm()
        norm_ub = np.sqrt(self.max_feats)

        w_old = np.ones(self.n_features_all) * 2.       # invalid on purpose: enters the loop
        w = np.ones(self.n_features_all) / np.sqrt(self.n_features_all)
        centers_unweighted = None
        n_iter = 0
        while n_iter < self.max_iter and norm(w - w_old, 1) / norm(w_old, 1) > self.tol_w:
            n_iter += 1
            w_old = w
            feat_weights = np.sqrt(w)
            # warm start from the previous centroids, re-weighted by the current weights
            if n_iter > 1:
                jm.centers_ = centers_unweighted * feat_weights
            jm.fit(X, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
            centers_unweighted = weighted_mean_cluster(X_arr, jm.proba_)
            BCSS = np.asarray(compute_BCSS(X_arr, jm.proba_, centers_unweighted), dtype=float)
            if (BCSS <= 0.).all():                      # everything landed in one cluster
                self.print_log(n_iter, BCSS, w)
                break
            w = solve_lasso_pinned(BCSS / BCSS.max(), norm_ub, pin_mask, floor=self.pin_floor)
            self.print_log(n_iter, BCSS, w)

        self.w = raise_arr_to_pd_obj(w, X, index_key="columns")
        self.feat_weights = raise_arr_to_pd_obj(jm.feat_weights, X, index_key="columns")
        self.centers_ = jm.centers_                     # weighted centers
        self.labels_ = jm.labels_
        self.proba_ = jm.proba_
        if ret_ser is not None:
            self.ret_ = jm.ret_
            self.vol_ = jm.vol_
        return self
