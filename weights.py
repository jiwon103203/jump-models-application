"""
변수 유형별 가중치 -- how the weight the sparse jump model puts on the features splits
across *kinds* of variable, and how that split moves from one re-estimation to the next.

`feat_weights.csv` already records the weight of every individual feature at every
re-estimation, but a 35-feature run makes that table hard to read: the interesting question
is rarely "what happened to `mad_20`" and usually "is the model separating the regimes on
the level of returns, on realized volatility, or on how far prices have fallen", and whether
that answer is stable over time. Grouping the columns answers it -- with an "extra" feature
set the five groups below cover all 35 columns, and their shares add up to 100% at every
refit date, which is what makes the picture comparable across re-estimations and across runs.

Four groupings are available, from the coarsest to the finest:

- ``"type"`` (default): the five semantic groups of `FEATURE_TYPE_SERIES` -- the level of
  returns, realized volatility, log volatility, downside deviation, and everything the user
  brought in through ``--extra-feature``.
- ``"type-horizon"``: the same groups split by horizon, which separates "the model looks at
  5-day volatility" from "the model looks at 20-day volatility".
- ``"series"``: one group per feature series (`features.feature_series_name`), i.e. `std`,
  `vol-log`, `DD` and so on -- the finest grouping that still merges horizons.
- ``"horizon"``: the horizon alone, ignoring which statistic is measured over it.

The weights themselves are those of the sparse jump model, which normalizes them to unit L2
norm rather than to a sum of one; the shares here are therefore taken on the L1 scale
(``w_j / Σ_j w_j``), which is the scale on which "60% of the weight sits on volatility"
means what it sounds like.
"""

import re

import numpy as np
import pandas as pd

from features import feature_series_name

# the feature series each semantic type covers, in the order the groups are reported in.
# Every series the "paper", "example" and "extra" sets build appears exactly once; anything
# else -- a custom variable added with `--extra-feature` -- falls through to `CUSTOM_TYPE`.
FEATURE_TYPE_SERIES = {
    # the level of returns and the return-per-unit-of-downside-risk built on it
    "return": ("ret", "sortino", "ret-simple", "ret-log", "ret-cumlog"),
    # two-sided dispersion measured over a rolling window, and the daily proxies for it
    "realized-vol": ("std", "var", "mad", "rms", "ret-abs", "ret-sq"),
    # the EWMA volatility on the log scale, its change and its term structure
    "log-vol": ("vol-log", "vol-chg", "vol-ratio"),
    # one-sided dispersion: the downside deviation, on either scale
    "downside": ("DD", "DD-log"),
}
CUSTOM_TYPE = "custom"
FEATURE_TYPES = tuple(FEATURE_TYPE_SERIES) + (CUSTOM_TYPE,)

GROUPINGS = ("type", "type-horizon", "series", "horizon")
NO_HORIZON = "none"            # the group of the columns that carry no horizon in their name

_SERIES_TO_TYPE = {series: type_name
                   for type_name, series_list in FEATURE_TYPE_SERIES.items()
                   for series in series_list}
# a horizon is a plain window length (`std_20`) or a pair of them (`vol-ratio_5-20`); the
# suffix of a custom variable (`VIX_ewm20`) is a transform, not a horizon
_HORIZON_RE = re.compile(r"^\d+(?:-\d+)?$")


def feature_type_name(column: str) -> str:
    """
    Return the semantic type of a feature column: one of `FEATURE_TYPES`.

    The type is read off the column's *series* (`features.feature_series_name`), so every
    horizon of a statistic lands in the same type and a column the feature sets do not build
    -- a custom variable -- is reported as "custom".
    """
    return _SERIES_TO_TYPE.get(feature_series_name(column), CUSTOM_TYPE)


def feature_horizon(column: str) -> str:
    """
    Return the horizon a feature column is measured over, as it appears in its name.

    ``std_20`` gives ``"20"`` and ``vol-ratio_5-20`` gives ``"5-20"``. A column whose name
    carries no horizon (``ret-cumlog``) or whose suffix is a transform rather than a window
    (the ``VIX_ewm20`` of a custom variable) gives `NO_HORIZON`.
    """
    parts = str(column).split("_", 1)
    if len(parts) == 1:
        return NO_HORIZON
    return parts[1] if _HORIZON_RE.match(parts[1]) else NO_HORIZON


def feature_group_name(column: str, grouping: str = "type") -> str:
    """
    Return the group a feature column belongs to under one of the `GROUPINGS`.

    Parameters
    ----------
    column : str
        The feature column name.

    grouping : str, optional (default="type")
        One of `GROUPINGS`, see the module docstring.

    Returns
    -------
    str
        The group name.
    """
    if grouping == "type":
        return feature_type_name(column)
    if grouping == "series":
        return feature_series_name(column)
    if grouping == "horizon":
        return feature_horizon(column)
    if grouping == "type-horizon":
        horizon = feature_horizon(column)
        type_name = feature_type_name(column)
        return type_name if horizon == NO_HORIZON else f"{type_name}_{horizon}"
    raise ValueError(f"지원하지 않는 그룹 기준입니다: {grouping}. 가능한 값: {GROUPINGS}")


def feature_group_map(columns, grouping: str = "type") -> dict:
    """
    Map every group to the feature columns it covers.

    The groups of the ``"type"`` grouping come out in the fixed order of `FEATURE_TYPES` so
    that a stacked figure keeps the same band order across runs; the other groupings, whose
    members depend on the feature set, come out in order of first appearance. Empty groups
    are left out either way.

    Parameters
    ----------
    columns : iterable of str
        The feature columns, e.g. those of `feat_weights`.

    grouping : str, optional (default="type")
        One of `GROUPINGS`.

    Returns
    -------
    dict of str -> list of str
        The columns of each non-empty group.
    """
    columns = [str(col) for col in columns]
    mapping = {}
    for column in columns:
        mapping.setdefault(feature_group_name(column, grouping), []).append(column)
    if grouping == "type":
        return {name: mapping[name] for name in FEATURE_TYPES if name in mapping}
    return mapping


def group_feature_weights(feat_weights: pd.DataFrame, grouping: str = "type") -> pd.DataFrame:
    """
    Aggregate the per-feature weights of every re-estimation into per-group shares.

    Each row of the result is one re-estimation; the group columns are that refit's share of
    the total weight and sum to one, and the trailing ``total`` column is the total itself
    (``‖w‖₁``), kept so that the table stands on its own -- the shares say how the weight is
    split, `total` says how concentrated the weight vector is to begin with.

    A re-estimation at which every weight is zero -- which the fit warns about separately,
    since it means the training window landed in a single cluster -- comes out as NaN shares
    rather than as a division by zero.

    Parameters
    ----------
    feat_weights : pd.DataFrame
        The `feat_weights` table of `rolling.RollingJMResult`: one row per refit date, one
        column per feature, non-negative weights.

    grouping : str, optional (default="type")
        One of `GROUPINGS`, see the module docstring.

    Returns
    -------
    pd.DataFrame
        Indexed by refit date, one column per non-empty group plus ``total``.
    """
    if feat_weights is None or feat_weights.empty:
        raise ValueError("피처 가중 표가 비어 있습니다. 가중치 분해는 --model sjm 에서만 가능합니다.")
    mapping = feature_group_map(feat_weights.columns, grouping)
    if "total" in mapping:      # a custom variable could collide with the trailing column
        raise ValueError("'total'이라는 이름의 그룹이 생겨 합계 열과 충돌합니다. "
                         "커스텀 변수 이름을 바꾸거나 다른 --weight-group 을 쓰세요.")
    totals = pd.DataFrame({name: feat_weights[cols].sum(axis=1) for name, cols in mapping.items()},
                          index=feat_weights.index)
    grand_total = totals.sum(axis=1)
    shares = totals.div(grand_total.replace(0., np.nan), axis=0)
    shares["total"] = grand_total
    shares.index.name = feat_weights.index.name or "refit_date"
    return shares


def weight_group_summary(shares: pd.DataFrame, feat_weights: pd.DataFrame = None,
                         grouping: str = "type") -> pd.DataFrame:
    """
    Summarize the group shares over the whole run, one row per group.

    Parameters
    ----------
    shares : pd.DataFrame
        The output of `group_feature_weights`.

    feat_weights : pd.DataFrame, optional
        The per-feature weights the shares were built from. When given, the summary also
        reports how many features each group holds and how many of them the sparse model
        keeps on average -- a group can carry a large share through one surviving feature
        or through many small ones, and those are different things.

    grouping : str, optional (default="type")
        The grouping `shares` was built with, needed only to resolve `feat_weights`.

    Returns
    -------
    pd.DataFrame
        Indexed by group, with the mean, minimum and maximum share, the share at the last
        re-estimation, and -- with `feat_weights` -- the feature count and the mean number
        of features kept.
    """
    group_cols = [col for col in shares.columns if col != "total"]
    summary = pd.DataFrame({
        "mean_share": shares[group_cols].mean(),
        "min_share": shares[group_cols].min(),
        "max_share": shares[group_cols].max(),
        "last_share": shares[group_cols].iloc[-1],
    })
    if feat_weights is not None:
        mapping = feature_group_map(feat_weights.columns, grouping)
        summary["n_features"] = pd.Series({name: len(cols) for name, cols in mapping.items()})
        summary["mean_kept"] = pd.Series({name: float((feat_weights[cols] > 0).sum(axis=1).mean())
                                          for name, cols in mapping.items()})
    return summary.sort_values("mean_share", ascending=False)
