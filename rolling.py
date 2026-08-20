"""
Rolling estimation of the jump model, following the protocol of Shu, Yu and Mulvey (2024).

The optimal model parameters are re-estimated every six months -- on the first trading
day on or after January 1st and July 1st -- over a training window of 3000 trading days
(about twelve years). Between two re-estimations the parameters stay fixed and the
prevailing regime of each day is inferred online, i.e. from the features available at
the end of that day only, using a lookback window of the same length as the training
window (Section 3.4.2 of the article).
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from jumpmodels.jump import JumpModel
except ImportError:  # a source checkout that has not been pip-installed
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from jumpmodels.sparse_jump import SparseJumpModel
from jumpmodels.utils import weighted_mean_cluster

from sparse_pin import PinnedSparseJumpModel

TRADING_DAYS = 252
REFIT_MONTHS = (1, 7)          # January and July, i.e. a semiannual refit
MAX_ANCHOR_GAP_DAYS = 45       # tolerance between the 1st of the month and the first trading day
MODELS = ("jm", "sjm")         # the original discrete JM, and the sparse JM with feature selection


def resolve_max_feats(max_feats, n_features: int, n_pinned: int = 0) -> float:
    """
    Validate the `max_feats` parameter of the sparse jump model, or pick a default.

    `max_feats` is the square of kappa in Nystrup et al. (2021) and roughly represents the
    effective number of features kept by the Lasso-like constraint on the feature weights.
    Pinned features are kept whatever their weight, so the budget cannot be smaller than
    their number without being violated on every re-estimation; it is raised to `n_pinned`
    when it is, and the default never falls below it either.

    Parameters
    ----------
    max_feats : float or None
        The requested value. When None, half of the features is used.

    n_features : int
        The number of features in the feature matrix.

    n_pinned : int, optional (default=0)
        The number of pinned features, see `sparse_pin.PinnedSparseJumpModel`.

    Returns
    -------
    float
        The value to pass to `SparseJumpModel`.
    """
    if max_feats is None:
        return max(2., n_features / 2., float(n_pinned))
    max_feats = float(max_feats)
    if max_feats < 1.:
        raise ValueError(f"max_feats는 1 이상이어야 합니다. 입력값: {max_feats}")
    if max_feats < n_pinned:
        warnings.warn(
            f"max_feats({max_feats:g})가 고정한 피처 개수({n_pinned})보다 작아 {n_pinned}로 늘립니다. "
            f"고정한 피처는 어차피 모두 남으므로 그보다 적게 줄일 수 없습니다.")
        max_feats = float(n_pinned)
    if max_feats > n_features:
        warnings.warn(
            f"max_feats({max_feats:g})가 피처 개수({n_features})보다 커서 {n_features}로 줄입니다. "
            f"이 경우 모든 피처가 동일 가중을 받아 원래 JM과 같아집니다.")
        return float(n_features)
    return max_feats


def init_model(model: str = "jm",
               n_components: int = 2,
               jump_penalty: float = 50.,
               n_init: int = 10,
               random_state: int = 0,
               max_feats: float = None,
               n_features: int = None,
               pin_mask=None):
    """
    Build the model instance used at each re-estimation.

    Parameters
    ----------
    model : str, optional (default="jm")
        "jm" for the original discrete jump model of the article, or "sjm" for the sparse
        jump model of Nystrup et al. (2021), which weighs and selects features.

    n_components : int, optional (default=2)
        The number of regimes.

    jump_penalty : float, optional (default=50.)
        The jump penalty. The sparse model rescales it internally by `1/sqrt(n_features)`,
        so values of a similar magnitude work for both models.

    n_init : int, optional (default=10)
        The number of restarts of the coordinate descent algorithm.

    random_state : int, optional (default=0)
        The seed of the centroid initialization.

    max_feats : float, optional
        Sparse model only: the effective number of features to keep, see `resolve_max_feats`.

    n_features : int, optional
        The number of features, needed to resolve `max_feats`.

    pin_mask : array-like of bool, optional
        Sparse model only: True for the features the selection may never drop, aligned with
        the columns of the feature matrix. When it selects at least one feature, the pinned
        variant of the sparse model is returned.

    Returns
    -------
    JumpModel, SparseJumpModel or PinnedSparseJumpModel
        The unfitted model instance.
    """
    if model == "jm":
        return JumpModel(n_components=n_components, jump_penalty=jump_penalty, cont=False,
                         n_init=n_init, random_state=random_state)
    if model == "sjm":
        n_pinned = 0 if pin_mask is None else int(np.count_nonzero(pin_mask))
        kwargs = dict(n_components=n_components,
                      max_feats=resolve_max_feats(max_feats, n_features, n_pinned=n_pinned),
                      jump_penalty=jump_penalty, cont=False,
                      n_init_jm=n_init, random_state=random_state)
        if n_pinned:
            return PinnedSparseJumpModel(pin_mask=np.asarray(pin_mask, dtype=bool), **kwargs)
        return SparseJumpModel(**kwargs)
    raise ValueError(f"지원하지 않는 모델입니다: {model}. 가능한 값: {MODELS}")


def semiannual_anchors(index) -> list:
    """
    Locate the first trading day on or after January 1st and July 1st of every year covered
    by `index`.

    Parameters
    ----------
    index : pd.Index
        A sorted index of `datetime.date` (or anything `pd.to_datetime` accepts).

    Returns
    -------
    list of (date, int)
        The anchor dates and their integer positions within `index`. Anchors falling on the
        very first row of the data are excluded, since the first trading day of the sample
        is not necessarily the first trading day of a half-year.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(pd.Index(index)))
    anchors, seen = [], set()
    for year in range(dt[0].year, dt[-1].year + 1):
        for month in REFIT_MONTHS:
            target = pd.Timestamp(year=year, month=month, day=1)
            pos = int(dt.searchsorted(target, side="left"))
            if pos == 0 or pos >= len(dt) or pos in seen:
                continue
            if (dt[pos] - target).days > MAX_ANCHOR_GAP_DAYS:
                continue       # a data gap: no trading day close enough to the half-year start
            seen.add(pos)
            anchors.append((index[pos], pos))
    return sorted(anchors, key=lambda pair: pair[1])


def refit_schedule(index, window: int = 3000, min_window: int = 500, start_date=None) -> list:
    """
    Build the list of re-estimation dates, keeping only the anchors backed by enough history.

    Parameters
    ----------
    index : pd.Index
        The index of the feature matrix.

    window : int, optional (default=3000)
        The intended length of the training window, in trading days.

    min_window : int, optional (default=500)
        The shortest training window accepted. Anchors with fewer prior observations are
        skipped, so that the first re-estimation is not fitted on a handful of days.

    start_date : str or datetime.date, optional
        Ignore anchors before this date.

    Returns
    -------
    list of (date, int, int)
        The refit date, its position in `index`, and the training window length actually
        used at that date (`min(window, position)`).
    """
    if min_window > window:
        raise ValueError(f"min_window({min_window})는 window({window})보다 클 수 없습니다.")
    schedule = []
    start = None if start_date is None else pd.Timestamp(start_date).date()
    for date, pos in semiannual_anchors(index):
        if pos < min_window:
            continue
        if start is not None and date < start:
            continue
        schedule.append((date, pos, min(window, pos)))
    return schedule


@dataclass
class RollingJMResult:
    """
    Container for the output of `run_rolling_jm`.

    Attributes
    ----------
    regimes : pd.DataFrame
        Online inferred regimes, indexed by date, with the assigned regime (0 = bull,
        1 = bear), the regime probabilities and the refit date whose parameters were used.

    params : pd.DataFrame
        One row per (refit date, state): the training window, the cluster centroid in the
        original feature units, the in-sample frequency, mean return and volatility of the
        state, and its self-transition probability.

    insample_last : pd.Series
        In-sample regimes of the most recent training window, useful to compare in-sample
        fitting against online inference (Figure 4 of the article).

    schedule : list
        The `(date, position, window)` triples of the re-estimation schedule.

    feature_names : list of str
        The columns of the feature matrix.

    window : int
        The intended training window length.

    model : str
        The model used, "jm" or "sjm".

    feat_weights : pd.DataFrame or None
        Sparse model only: the feature weights of every re-estimation, indexed by refit
        date with one column per feature. A zero weight means the feature was dropped.

    pinned_features : list of str
        Sparse model only: the features pinned into the model, i.e. those whose weight is
        positive at every re-estimation by construction. Empty when nothing was pinned.
    """
    regimes: pd.DataFrame
    params: pd.DataFrame
    insample_last: pd.Series
    schedule: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    window: int = 3000
    model: str = "jm"
    feat_weights: pd.DataFrame = None
    pinned_features: list = field(default_factory=list)


def run_rolling_jm(X: pd.DataFrame,
                   ret_ser: pd.Series,
                   jump_penalty: float = 50.,
                   window: int = 3000,
                   min_window: int = 500,
                   n_components: int = 2,
                   clip_mul: float = 3.,
                   n_init: int = 10,
                   random_state: int = 0,
                   start_date=None,
                   model: str = "jm",
                   max_feats: float = None,
                   pin_feats=None,
                   verbose: bool = True) -> RollingJMResult:
    """
    Re-estimate a jump model every six months and infer the regimes online in between.

    At each refit date the model is fitted on the `window` trading days that *precede* the
    date, so that no future information enters the parameters. The fitted parameters then
    generate the online regime inference from the refit date up to (but excluding) the next
    one. Clipping and standardization are refitted on each training window as well.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix, indexed by date.

    ret_ser : pd.Series
        The return series used to sort the states, typically the excess return. The state
        with the higher cumulative return over the training window becomes the bull regime
        (label 0).

    jump_penalty : float, optional (default=50.)
        The jump penalty of the model.

    window : int, optional (default=3000)
        The training window length, in trading days.

    min_window : int, optional (default=500)
        The shortest training window accepted; see `refit_schedule`.

    n_components : int, optional (default=2)
        The number of regimes.

    clip_mul : float, optional (default=3.)
        Winsorization threshold, in training-window standard deviations.

    n_init : int, optional (default=10)
        Number of restarts of the coordinate descent algorithm.

    random_state : int, optional (default=0)
        Seed of the centroid initialization.

    start_date : str or datetime.date, optional
        Ignore refit dates before this date.

    model : str, optional (default="jm")
        "jm" for the original discrete jump model, or "sjm" for the sparse jump model,
        which weighs the features and drops the noisy ones. The sparse model is worth
        using once the feature set grows beyond the three features of the article.

    max_feats : float, optional
        Sparse model only: the effective number of features to keep. Defaults to half of
        the features, see `resolve_max_feats`.

    pin_feats : iterable of str, optional
        Sparse model only: the columns of `X` the feature selection may never drop. Use
        `features.resolve_pinned_features` to turn feature-set names such as "paper" into
        the column names expected here.

    verbose : bool, optional (default=True)
        Whether to print the progress of the re-estimations.

    Returns
    -------
    RollingJMResult
        The online inferred regimes, the estimated parameters of every refit, the in-sample
        regimes of the last training window, and -- for the sparse model -- the feature
        weights of every re-estimation.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X는 pandas DataFrame이어야 합니다.")
    if model not in MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model}. 가능한 값: {MODELS}")
    if model == "sjm" and X.shape[1] <= 3:
        warnings.warn(
            f"피처가 {X.shape[1]}개뿐이라 sparse JM의 피처 선택 효과가 거의 없습니다. "
            f"커스텀 변수를 추가하거나 --model jm 을 쓰는 편이 낫습니다.")

    requested = [str(name) for name in (pin_feats or [])]
    unknown = [name for name in requested if name not in X.columns]
    if unknown:
        raise KeyError(
            f"고정할 피처 {unknown}을 피처 행렬에서 찾을 수 없습니다. "
            f"사용 가능한 피처: {list(X.columns)}")
    pinned = [col for col in X.columns if col in set(requested)]
    if pinned and model != "sjm":
        warnings.warn(
            f"피처 고정은 피처를 선택하는 sparse JM 전용이라 --model {model} 에서는 무시합니다. "
            f"고정이 필요하면 --model sjm 을 쓰세요.")
        pinned = []
    if pinned and len(pinned) == X.shape[1]:
        warnings.warn(
            f"피처 {X.shape[1]}개를 모두 고정해 sparse JM의 피처 선택이 사실상 꺼집니다. "
            f"가중은 여전히 BCSS에 따라 달라지지만 탈락하는 피처는 없습니다.")
    pin_mask = X.columns.isin(pinned) if pinned else None

    ret_ser = ret_ser.reindex(X.index)
    if ret_ser.isna().any():
        raise ValueError("수익률 시리즈가 피처 행렬의 모든 날짜를 포함하지 않습니다.")

    schedule = refit_schedule(X.index, window=window, min_window=min_window, start_date=start_date)
    if not schedule:
        raise ValueError(
            f"재추정 시점을 만들 수 없습니다. 피처 {len(X)}행 (기간: {X.index[0]} ~ {X.index[-1]})으로는 "
            f"최소 학습창 {min_window}거래일을 확보한 1월/7월 첫 영업일이 없습니다. "
            f"--min-window(및 --window)나 --warmup 을 줄이거나 더 긴 데이터를 사용해 주세요.")

    short = [(d, w) for d, _, w in schedule if w < window]
    if short:
        warnings.warn(
            f"{len(schedule)}개 재추정 시점 중 {len(short)}개가 데이터 부족으로 학습창이 축소되었습니다 "
            f"(가장 짧은 창: {min(w for _, w in short)}거래일, 목표: {window}거래일).")

    proba_cols = [f"proba_{i}" for i in range(n_components)]
    regime_parts, param_rows, weight_rows = [], [], {}
    insample_last = None
    n_obs = len(X)

    for i, (refit_date, pos, win) in enumerate(schedule):
        train_slice = slice(pos - win, pos)
        X_train = X.iloc[train_slice]
        ret_train = ret_ser.iloc[train_slice]

        # clipping and standardization are refitted on the current training window only
        clipper, scaler = DataClipperStd(mul=clip_mul), StandardScalerPD()
        X_train_processed = scaler.fit_transform(clipper.fit_transform(X_train))

        model_ins = init_model(model, n_components=n_components, jump_penalty=jump_penalty,
                               n_init=n_init, random_state=random_state, max_feats=max_feats,
                               n_features=X.shape[1], pin_mask=pin_mask)
        model_ins.fit(X_train_processed, ret_train, sort_by="cumret")

        # online inference from this refit date until the next one
        seg_end = schedule[i + 1][1] if i + 1 < len(schedule) else n_obs
        X_context = X.iloc[pos - win:seg_end]      # lookback window + the segment itself
        proba_online = model_ins.predict_proba_online(scaler.transform(clipper.transform(X_context)))
        proba_seg = proba_online.iloc[win:]        # drop the lookback rows

        seg = pd.DataFrame(np.asarray(proba_seg), index=proba_seg.index, columns=proba_cols)
        seg.insert(0, "regime", np.asarray(proba_seg).argmax(axis=1))
        seg["refit_date"] = refit_date
        regime_parts.append(seg)

        # parameters of this refit, with the centroids mapped back to the feature units.
        # The sparse model stores its centroids in the weighted space, so they are recomputed
        # on the unweighted features before being un-standardized.
        if model == "sjm":
            centers = weighted_mean_cluster(np.asarray(X_train_processed),
                                            np.asarray(model_ins.proba_))
            weight_rows[refit_date] = pd.Series(np.asarray(model_ins.feat_weights),
                                                index=X.columns)
        else:
            centers = model_ins.centers_
        centers_orig = scaler.scaler.inverse_transform(centers)
        transmat = getattr(model_ins, "transmat_", None)
        if transmat is None:        # the sparse model keeps it on its inner jump model
            transmat = model_ins.jm_ins.transmat_
        labels_train = np.asarray(model_ins.labels_)
        missing = [k for k in range(n_components) if not (labels_train == k).any()]
        if missing:
            warnings.warn(
                f"{refit_date} 재추정의 학습창에 상태 {missing}가 나타나지 않아 해당 파라미터가 "
                f"NaN으로 기록됩니다. 학습창이 한 레짐만 덮고 있을 수 있습니다(--window 확인).")
        for k in range(n_components):
            row = {
                "refit_date": refit_date,
                "train_start": X_train.index[0],
                "train_end": X_train.index[-1],
                "n_train": win,
                "state": k,
                "regime": "bull" if k == 0 else ("bear" if k == n_components - 1 else f"mid{k}"),
                "freq": float((labels_train == k).mean()),
                "ret_ann": float(model_ins.ret_[k] * TRADING_DAYS),
                "vol_ann": float(model_ins.vol_[k] * np.sqrt(TRADING_DAYS)),
                "stay_prob": float(transmat[k, k]),
            }
            row.update({f"center_{col}": float(val) for col, val in zip(X.columns, centers_orig[k])})
            param_rows.append(row)

        if i == len(schedule) - 1:
            insample_last = pd.Series(labels_train, index=X_train.index, name="insample_regime")

        if verbose:
            selected = ""
            if model == "sjm":
                weights = weight_rows[refit_date]
                kept = weights[weights > 0]
                selected = f", 선택된 피처 {len(kept)}/{len(weights)}"
                if pinned:
                    selected += f" (고정 {len(pinned)})"
            print(f"[{i + 1}/{len(schedule)}] refit {refit_date}: "
                  f"학습창 {X_train.index[0]} ~ {X_train.index[-1]} ({win}일), "
                  f"온라인 추론 {seg.index[0]} ~ {seg.index[-1]} ({len(seg)}일), "
                  f"bear 비중 {float((seg.regime == n_components - 1).mean()):.1%}{selected}")

    regimes = pd.concat(regime_parts)
    regimes.index.name = "date"
    feat_weights = None
    if weight_rows:
        feat_weights = pd.DataFrame(weight_rows).T
        feat_weights.index.name = "refit_date"
    return RollingJMResult(regimes=regimes,
                           params=pd.DataFrame(param_rows),
                           insample_last=insample_last,
                           schedule=schedule,
                           feature_names=list(X.columns),
                           window=window,
                           model=model,
                           feat_weights=feat_weights,
                           pinned_features=pinned)
