"""
Feature engineering for the jump model, following Shu, Yu and Mulvey (2024),
"Downside Risk Reduction Using Regime-Switching Signals".

Three feature sets are provided:

- ``"paper"`` (default): the three features of Table 2 of the article -- the EWM
  downside deviation with a halflife of 10 days, and EWM Sortino ratios with
  halflives of 20 and 60 days, all computed on the excess return series.
- ``"example"``: the nine-feature set of ``examples/nasdaq/feature.py`` -- EWM
  return, log EWM downside deviation and EWM Sortino ratio over halflives of
  5, 20 and 60 days.
- ``"extra"``: the nine features of ``"example"`` plus a broad set of return and
  volatility statistics -- the plain, log, absolute, squared and cumulative return,
  the rolling standard deviation, variance, mean absolute return and RMS return
  over 5/20/60-day windows, and the EWMA volatility with its change over time and
  its ratio across horizons. 34 features in total; pair it with ``--model sjm``,
  which drops the ones that carry no regime information.
"""

import re

import numpy as np
import pandas as pd

FEATURE_SETS = ("paper", "example", "extra")
TRANSFORMS = ("raw", "ewm", "diff", "pct", "log", "logdiff")

PAPER_DD_HL = 10.
PAPER_SORTINO_HLS = (20., 60.)
EXAMPLE_HLS = (5., 20., 60.)
# the "extra" set reuses the horizons of the "example" set, as rolling windows for the
# simple-window statistics and as halflives for the exponentially weighted ones
EXTRA_WINDOWS = (5, 20, 60)
EXTRA_VOL_RATIO_PAIRS = ((5., 20.), (20., 60.))


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


def compute_ewm_vol(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the EWMA volatility of a return series, the RiskMetrics-style square root of
    the EWM second moment of the returns.

    It is the two-sided counterpart of `compute_ewm_DD`: the same statistic taken over
    every return rather than over the negative ones only, so that the pair separates a
    fall in prices from a rise in turbulence.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife, in periods, of the exponential weights.

    Returns
    -------
    pd.Series
        The EWMA volatility, in the units of the returns (not annualized).
    """
    return np.sqrt(ret_ser.pow(2).ewm(halflife=hl).mean())


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


def example_features(ret_ser: pd.Series) -> dict:
    """
    Build the nine features of the "example" set: the EWM return, the log EWM downside
    deviation and the EWM Sortino ratio over halflives of 5, 20 and 60 days.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    Returns
    -------
    dict of {str: pd.Series}
        The features, keyed by column name.
    """
    feat_dict = {}
    for hl in EXAMPLE_HLS:
        # Feature 1: EWM-ret
        feat_dict[f"ret_{hl:.0f}"] = ret_ser.ewm(halflife=hl).mean()
        # Feature 2: log(EWM-DD)
        DD = compute_ewm_DD(ret_ser, hl)
        feat_dict[f"DD-log_{hl:.0f}"] = np.log(DD)
        # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD
        feat_dict[f"sortino_{hl:.0f}"] = feat_dict[f"ret_{hl:.0f}"].div(DD)
    return feat_dict


def extra_features(ret_ser: pd.Series) -> dict:
    """
    Build the return and volatility statistics the "extra" set adds on top of the
    "example" set.

    The "example" features already summarize the *level* of the trailing return (its EWM
    mean) and its *downside* dispersion (the EWM downside deviation); a rolling return is
    therefore not repeated here. What they leave out, and what this function adds, is the
    plain return in its various transforms, two-sided dispersion measured over simple
    windows, and the term structure and dynamics of the EWMA volatility:

    ============================ ==================================================
    ``ret-simple``               the return itself, `r_t`
    ``ret-log``                  the log return, `log(1 + r_t)`
    ``ret-abs``                  the absolute return, `|r_t|`
    ``ret-sq``                   the squared return, `r_t^2`
    ``ret-cumlog``               the cumulative log return since the first observation
    ``std_w``                    the rolling standard deviation over `w` days
    ``var_w``                    the rolling variance over `w` days, i.e. `std_w^2`
    ``mad_w``                    the rolling mean absolute return over `w` days
    ``rms_w``                    the rolling root mean square return over `w` days
    ``vol-log_hl``               the log EWMA volatility at halflife `hl`
    ``vol-chg_hl``               its change over the last `hl` days, a log change
    ``vol-ratio_s-l``            `vol-log_s - vol-log_l`, the log ratio of the two
    ============================ ==================================================

    Volatilities are kept on the log scale, as the "example" set does for its downside
    deviation: they are positive and right-skewed, so the log is the better behaved
    coordinate, and differences of logs are exactly the changes and ratios of the last
    two rows.

    Every statistic looks backwards only, so none of them leaks future information.
    `ret-cumlog` is the one non-stationary column: it trends with the market over the
    whole sample instead of fluctuating around a level, which makes it a poor clustering
    coordinate. It is kept because a cumulative return is a common input, but the sparse
    jump model (`--model sjm`) is expected to drop it.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    Returns
    -------
    dict of {str: pd.Series}
        The features, keyed by column name.
    """
    # a daily excess return of -100% or worse cannot be logged; it does not occur in
    # practice, and any such row is dropped downstream by `build_features`
    log_ret = np.log1p(ret_ser.where(ret_ser > -1.))
    feat_dict = {
        "ret-simple": ret_ser,
        "ret-log": log_ret,
        "ret-abs": ret_ser.abs(),
        "ret-sq": ret_ser.pow(2),
        "ret-cumlog": log_ret.cumsum(),
    }
    for window in EXTRA_WINDOWS:
        rolling = ret_ser.rolling(window)
        feat_dict[f"std_{window:d}"] = rolling.std()
        feat_dict[f"var_{window:d}"] = rolling.var()
        feat_dict[f"mad_{window:d}"] = ret_ser.abs().rolling(window).mean()
        feat_dict[f"rms_{window:d}"] = np.sqrt(ret_ser.pow(2).rolling(window).mean())
    for hl in EXAMPLE_HLS:
        log_vol = np.log(compute_ewm_vol(ret_ser, hl))
        feat_dict[f"vol-log_{hl:.0f}"] = log_vol
        # how much the volatility moved over its own horizon
        feat_dict[f"vol-chg_{hl:.0f}"] = log_vol.diff(int(hl))
    for short_hl, long_hl in EXTRA_VOL_RATIO_PAIRS:
        feat_dict[f"vol-ratio_{short_hl:.0f}-{long_hl:.0f}"] = (
            feat_dict[f"vol-log_{short_hl:.0f}"] - feat_dict[f"vol-log_{long_hl:.0f}"])
    return feat_dict


def feature_engineer(ret_ser: pd.Series, ver: str = "paper", log_dd: bool = False) -> pd.DataFrame:
    """
    Build the feature matrix fed to the jump model from a (excess) return series.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series, typically the excess return over the risk-free rate.

    ver : str, optional (default="paper")
        One of "paper" (three features, Table 2 of the article), "example" (the nine
        features used in the Nasdaq example of this repo) or "extra" (those nine plus the
        return and volatility statistics of `extra_features`, 34 in total).

    log_dd : bool, optional (default=False)
        Whether to take the log of the downside deviation feature in the "paper"
        feature set. The "example" and "extra" sets always use the log scale.

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
        return pd.DataFrame(example_features(ret_ser))

    if ver == "extra":
        return pd.DataFrame({**example_features(ret_ser), **extra_features(ret_ser)})

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
