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

import numpy as np
import pandas as pd

FEATURE_SETS = ("paper", "example")

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


def build_features(ret_ser: pd.Series, ver: str = "paper", warmup: int = 252, log_dd: bool = False) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        The feature matrix, free of NaN and infinite values.
    """
    X = feature_engineer(ret_ser, ver=ver, log_dd=log_dd)
    if warmup > 0:
        X = X.iloc[warmup:]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        raise ValueError(
            f"피처 계산 후 남은 데이터가 없습니다. 입력 길이는 {len(ret_ser)}행, warmup은 {warmup}행입니다.")
    return X
