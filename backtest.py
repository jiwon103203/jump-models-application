"""
The 0/1 strategy of Shu, Yu and Mulvey (2024) and its performance metrics.

The strategy holds 100% of the risky asset while the inferred regime is the bull market,
and switches 100% into the risk-free asset while it is the bear market. The regime
inferred at the end of day `t` is only traded at the end of day `t + delay`, i.e. it
drives the allocation from day `t + delay + 1` onwards; a one-way transaction cost is
charged on every change of allocation.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_COST_BPS = 10.       # one-way transaction cost, in basis points


def build_weights(regime_ser: pd.Series, delay: int = 1, bull_state: int = 0) -> pd.Series:
    """
    Turn a regime sequence into the daily weight on the risky asset.

    Parameters
    ----------
    regime_ser : pd.Series
        The online inferred regime of each day.

    delay : int, optional (default=1)
        The trading delay, in days. The signal of day `t` is applied from day `t + delay + 1`.

    bull_state : int, optional (default=0)
        The state treated as the favorable regime, held with a 100% weight.

    Returns
    -------
    pd.Series
        The weight on the risky asset, in {0., 1.}. The first `delay + 1` days, for which no
        tradable signal exists yet, are fully invested in the risky asset.
    """
    if delay < 0:
        raise ValueError(f"delay는 0 이상이어야 합니다. 입력값: {delay}")
    weights = (regime_ser == bull_state).astype(float).shift(delay + 1)
    return weights.fillna(1.).rename("weight")


def run_0_1_strategy(regime_ser: pd.Series,
                     ret_ser: pd.Series,
                     rf_ser: pd.Series,
                     delay: int = 1,
                     cost_bps: float = DEFAULT_COST_BPS,
                     bull_state: int = 0) -> pd.DataFrame:
    """
    Run the 0/1 strategy over the period covered by `regime_ser`.

    Parameters
    ----------
    regime_ser : pd.Series
        The online inferred regimes.

    ret_ser : pd.Series
        The total return series of the risky asset.

    rf_ser : pd.Series
        The daily risk-free rate.

    delay : int, optional (default=1)
        The trading delay, in days.

    cost_bps : float, optional (default=10.)
        One-way transaction cost in basis points, charged on the traded amount.

    bull_state : int, optional (default=0)
        The state held with a 100% weight in the risky asset.

    Returns
    -------
    pd.DataFrame
        Indexed by date, holding the regime, the weight on the risky asset, the risk-free
        rate, and the total returns of the buy-and-hold (`bh`) and 0/1 (`jm`) strategies.
    """
    idx = regime_ser.index
    ret, rf = ret_ser.reindex(idx), rf_ser.reindex(idx)
    if ret.isna().any() or rf.isna().any():
        raise ValueError("수익률 또는 무위험금리에 레짐 구간을 덮지 못하는 결측이 있습니다.")

    weights = build_weights(regime_ser, delay=delay, bull_state=bull_state)
    # no cost on the first day: the initial allocation is the starting position
    traded = weights.diff().abs().fillna(0.)
    gross = weights * ret + (1. - weights) * rf
    net = gross - traded * (cost_bps / 1e4)
    return pd.DataFrame({
        "regime": regime_ser,
        "weight": weights,
        "traded": traded,
        "rf": rf,
        "bh": ret,
        "jm": net,
    })


def max_drawdown(ret: pd.Series) -> float:
    """Maximum drawdown of the compounded wealth of a total return series."""
    wealth = (1. + ret).cumprod()
    return float((wealth / wealth.cummax() - 1.).min())


def performance_metrics(ret: pd.Series,
                        rf: pd.Series,
                        weight: pd.Series = None,
                        trading_days: int = TRADING_DAYS) -> dict:
    """
    Compute the annualized performance metrics reported in Table 4 of the article.

    Parameters
    ----------
    ret : pd.Series
        The total return series of the strategy.

    rf : pd.Series
        The daily risk-free rate over the same period.

    weight : pd.Series, optional
        The weight on the risky asset, needed for the turnover and leverage rows.

    trading_days : int, optional (default=252)
        Trading days per year used for annualization.

    Returns
    -------
    dict
        Compound annual growth rate, volatility, Sharpe ratio, maximum drawdown, Calmar
        ratio, 5% expected shortfall, and -- when `weight` is given -- turnover and leverage.
    """
    n = len(ret)
    excess = ret - rf.reindex(ret.index)
    vol = float(ret.std(ddof=1) * np.sqrt(trading_days))
    mdd = max_drawdown(ret)
    mean_excess_ann = float(excess.mean() * trading_days)
    var_5 = float(np.quantile(ret, 0.05))
    metrics = {
        "Return": float((1. + ret).prod() ** (trading_days / n) - 1.),
        "Volatility": vol,
        "Sharpe": mean_excess_ann / vol if vol > 0 else np.nan,
        "MDD": mdd,
        "Calmar": mean_excess_ann / abs(mdd) if mdd < 0 else np.nan,
        "ES_0.05": float(ret[ret <= var_5].mean()),
    }
    if weight is not None:
        weight = weight.reindex(ret.index)
        # one round trip (a sell and a buy) counts as 100% turnover
        metrics["Turnover"] = float(weight.diff().abs().sum() / 2. * trading_days / n)
        metrics["Leverage"] = float(weight.mean())
    else:
        metrics["Turnover"] = 0.
        metrics["Leverage"] = 1.
    return metrics


def performance_table(strategy_df: pd.DataFrame, trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Build the buy-and-hold versus 0/1 strategy comparison table.

    Parameters
    ----------
    strategy_df : pd.DataFrame
        The output of `run_0_1_strategy`.

    trading_days : int, optional (default=252)
        Trading days per year used for annualization.

    Returns
    -------
    pd.DataFrame
        Metrics in rows, strategies in columns ("B & H" and "JM 0/1").
    """
    rf = strategy_df["rf"]
    table = {
        "B & H": performance_metrics(strategy_df["bh"], rf, weight=None, trading_days=trading_days),
        "JM 0/1": performance_metrics(strategy_df["jm"], rf, weight=strategy_df["weight"],
                                      trading_days=trading_days),
    }
    return pd.DataFrame(table)


def format_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format the metric table for printing: percentages where the article uses them."""
    pct_rows = ("Return", "Volatility", "MDD", "ES_0.05", "Turnover", "Leverage")
    rows = {}
    for name, values in table.astype(float).iterrows():
        fmt = (lambda x: f"{x * 100:.1f}%") if name in pct_rows else (lambda x: f"{x:.2f}")
        rows[name] = values.map(fmt)
    return pd.DataFrame(rows).T.loc[table.index, table.columns]


def regime_summary(regime_ser: pd.Series, bear_state: int = 1) -> dict:
    """
    Summarize the persistence of an online inferred regime sequence.

    Parameters
    ----------
    regime_ser : pd.Series
        The online inferred regimes.

    bear_state : int, optional (default=1)
        The label of the unfavorable regime.

    Returns
    -------
    dict
        The share of bear days, the number of regime shifts, and the shifts per year.
    """
    n_years = len(regime_ser) / TRADING_DAYS
    n_shifts = int((regime_ser.diff().fillna(0.) != 0).sum())
    return {
        "start": regime_ser.index[0],
        "end": regime_ser.index[-1],
        "n_days": len(regime_ser),
        "bear_share": float((regime_ser == bear_state).mean()),
        "n_shifts": n_shifts,
        "shifts_per_year": n_shifts / n_years if n_years > 0 else np.nan,
    }
