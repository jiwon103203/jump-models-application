"""
Feature engineering for the jump model, following Shu, Yu and Mulvey (2024),
"Downside Risk Reduction Using Regime-Switching Signals".

Two feature sets are provided:

- ``"paper"`` (default): the three features of Table 2 of the article -- the EWM
  downside deviation with a halflife of 10 days, and EWM Sortino ratios with
  halflives of 20 and 60 days, all computed on the excess return series.
- ``"example"``: the nine-feature set of ``examples/nasdaq/feature.py`` -- EWM
  return, log EWM downside deviation and EWM Sortino ratio over halflives of
  5, 20 and 60 days.
"""

import re

import numpy as np
import pandas as pd

FEATURE_SETS = ("paper", "example")
TRANSFORMS = ("raw", "ewm", "diff", "pct", "log", "logdiff")

PAPER_DD_HL = 10.
PAPER_SORTINO_HLS = (20., 60.)
EXAMPLE_HLS = (5., 20., 60.)


def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) of a return series,
    i.e. the square root of the EWM second moment of the negative part of the returns.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife, in periods, of the exponential weights.

    Returns
    -------
    pd.Series
        The EWM downside deviation.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)


def compute_ewm_sortino(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the EWM Sortino ratio, the ratio of the EWM average return to the EWM
    downside deviation, both taken over the same halflife.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife, in periods, of the exponential weights.

    Returns
    -------
    pd.Series
        The EWM Sortino ratio.
    """
    return ret_ser.ewm(halflife=hl).mean().div(compute_ewm_DD(ret_ser, hl))


def feature_engineer(ret_ser: pd.Series, ver: str = "paper", log_dd: bool = False) -> pd.DataFrame:
    """
    Build the feature matrix fed to the jump model from a (excess) return series.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series, typically the excess return over the risk-free rate.

    ver : str, optional (default="paper")
        Either "paper" (three features, Table 2 of the article) or "example"
        (the nine features used in the Nasdaq example of this repo).

    log_dd : bool, optional (default=False)
        Whether to take the log of the downside deviation feature in the "paper"
        feature set. The "example" set always uses the log scale.

    Returns
    -------
    pd.DataFrame
        The feature matrix, indexed like `ret_ser`.
    """
    if ver == "paper":
        DD = compute_ewm_DD(ret_ser, PAPER_DD_HL)
        feat_dict = {f"DD{'-log' if log_dd else ''}_{PAPER_DD_HL:.0f}": np.log(DD) if log_dd else DD}
        for hl in PAPER_SORTINO_HLS:
            feat_dict[f"sortino_{hl:.0f}"] = compute_ewm_sortino(ret_ser, hl)
        return pd.DataFrame(feat_dict)

    if ver == "example":
        feat_dict = {}
        for hl in EXAMPLE_HLS:
            # Feature 1: EWM-ret
            feat_dict[f"ret_{hl:.0f}"] = ret_ser.ewm(halflife=hl).mean()
            # Feature 2: log(EWM-DD)
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl:.0f}"] = np.log(DD)
            # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD
            feat_dict[f"sortino_{hl:.0f}"] = feat_dict[f"ret_{hl:.0f}"].div(DD)
        return pd.DataFrame(feat_dict)

    raise NotImplementedError(f"지원하지 않는 피처 세트입니다: {ver}. 가능한 값: {FEATURE_SETS}")


############################################
## User-supplied custom variables
############################################

def parse_extra_spec(spec: str) -> tuple:
    """
    Parse a custom variable specification of the form ``column[:transform[:param]]``.

    Examples: ``"VIX"`` (used as is), ``"VIX:ewm:20"`` (EWM smoothed with a halflife of 20
    days), ``"신용스프레드:diff"`` (first difference), ``"거래량:logdiff:5"`` (5-day log change).

    Parameters
    ----------
    spec : str
        The specification string.

    Returns
    -------
    (str, str, float)
        The column name, the transform name, and its numeric parameter (the halflife for
        "ewm", the lag for "diff"/"pct"/"logdiff", ignored otherwise).
    """
    parts = [part.strip() for part in str(spec).split(":")]
    column = parts[0]
    if not column:
        raise ValueError(f"커스텀 변수 지정이 비어 있습니다: '{spec}'")
    transform = (parts[1].lower() if len(parts) > 1 and parts[1] else "raw")
    if transform not in TRANSFORMS:
        raise ValueError(f"지원하지 않는 변환입니다: '{transform}' (지정: '{spec}'). 가능한 값: {TRANSFORMS}")
    if len(parts) > 2 and parts[2]:
        try:
            param = float(parts[2])
        except ValueError:
            raise ValueError(f"변환 파라미터를 숫자로 읽지 못했습니다: '{parts[2]}' (지정: '{spec}')")
    else:
        param = 20. if transform == "ewm" else 1.
    if len(parts) > 3:
        raise ValueError(f"커스텀 변수 지정의 형식은 '열이름[:변환[:파라미터]]' 입니다: '{spec}'")
    return column, transform, param


def apply_transform(ser: pd.Series, transform: str, param: float = 1.) -> pd.Series:
    """
    Apply one of the supported causal transforms to a custom variable.

    Every transform only looks backwards, so a feature built here never leaks future
    information into the regime inference.

    Parameters
    ----------
    ser : pd.Series
        The raw variable.

    transform : str
        One of "raw", "ewm" (EWM mean over a halflife), "diff" (difference over a lag),
        "pct" (percentage change over a lag), "log" (natural log of the level), or
        "logdiff" (log change over a lag).

    param : float, optional (default=1.)
        The halflife for "ewm", the lag for "diff"/"pct"/"logdiff".

    Returns
    -------
    pd.Series
        The transformed variable.
    """
    if transform == "raw":
        return ser.astype(float)
    if transform == "ewm":
        if param <= 0:
            raise ValueError(f"ewm 변환의 반감기는 양수여야 합니다: {param}")
        return ser.astype(float).ewm(halflife=param).mean()
    lag = int(param)
    if transform in ("diff", "pct", "logdiff") and lag < 1:
        raise ValueError(f"{transform} 변환의 시차는 1 이상이어야 합니다: {param}")
    if transform == "diff":
        return ser.astype(float).diff(lag)
    if transform == "pct":
        return ser.astype(float).pct_change(lag)
    if (ser <= 0).any():
        raise ValueError(f"log/logdiff 변환에는 양수 값만 사용할 수 있습니다: '{ser.name}'")
    if transform == "log":
        return np.log(ser.astype(float))
    return np.log(ser.astype(float)).diff(lag)


def _extra_feature_name(column: str, transform: str, param: float) -> str:
    """Build the column name of a transformed custom variable, e.g. ``VIX_ewm20``."""
    slug = re.sub(r"\s+", "", str(column))
    if transform == "raw":
        return slug
    if transform == "log":
        return f"{slug}_log"
    suffix = f"{param:g}"
    return f"{slug}_{transform}{suffix}"


def build_extra_features(df: pd.DataFrame, specs) -> pd.DataFrame:
    """
    Build the custom feature columns described by `specs` from the raw variables in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        The frame holding the raw custom variables, indexed by date.

    specs : iterable of str
        Specifications of the form ``column[:transform[:param]]``, see `parse_extra_spec`.

    Returns
    -------
    pd.DataFrame
        The transformed features, indexed like `df`.
    """
    out = {}
    for spec in specs:
        column, transform, param = parse_extra_spec(spec)
        if column not in df.columns:
            raise KeyError(
                f"커스텀 변수 '{column}'을 데이터에서 찾을 수 없습니다. "
                f"사용 가능한 열: {[col for col in df.columns]}")
        out[_extra_feature_name(column, transform, param)] = apply_transform(df[column], transform, param)
    return pd.DataFrame(out, index=df.index)


def build_features(ret_ser: pd.Series,
                   ver: str = "paper",
                   warmup: int = 252,
                   log_dd: bool = False,
                   extra_features: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build the feature matrix and discard the burn-in period of the exponential weights.

    The EWM statistics at the very beginning of a series are computed from a handful of
    observations and are therefore unstable; the first `warmup` rows are dropped, along
    with any row still holding a NaN or an infinite value.

    Parameters
    ----------
    ret_ser : pd.Series
        The input (excess) return series.

    ver : str, optional (default="paper")
        The feature set, see `feature_engineer`.

    warmup : int, optional (default=252)
        Number of leading rows to discard, roughly one year of daily data.

    log_dd : bool, optional (default=False)
        Whether to log-transform the downside deviation of the "paper" feature set.

    extra_features : pd.DataFrame, optional
        Custom features to append to the return-based ones, typically the output of
        `build_extra_features`. They are aligned on the index of the return series.

    Returns
    -------
    pd.DataFrame
        The feature matrix, free of NaN and infinite values.
    """
    X = feature_engineer(ret_ser, ver=ver, log_dd=log_dd)
    if extra_features is not None and len(extra_features.columns):
        overlap = [col for col in extra_features.columns if col in X.columns]
        if overlap:
            raise ValueError(f"커스텀 변수 이름이 기본 피처와 겹칩니다: {overlap}")
        X = X.join(extra_features.reindex(X.index), how="left")
    if warmup > 0:
        X = X.iloc[warmup:]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        raise ValueError(
            f"피처 계산 후 남은 데이터가 없습니다. 입력 길이는 {len(ret_ser)}행, warmup은 {warmup}행입니다.")
    return X
