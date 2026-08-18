"""
The 0/1 strategy of Shu, Yu and Mulvey (2024) and its performance metrics.

The strategy holds 100% of the risky asset while the inferred regime is the bull market,
and switches 100% into the risk-free asset while it is the bear market. The regime
inferred at the end of day `t` is only traded at the end of day `t + delay`, i.e. it
drives the allocation from day `t + delay + 1` onwards; a one-way transaction cost is
charged on every change of allocation.

Two relaxations of that textbook setup are supported. Cash limits (`min_cash`, `max_cash`)
bound the share held in the risk-free asset in both regimes, so that the allocation moves
between `1 - max_cash` and `1 - min_cash` instead of between 0 and 1; the defaults
reproduce the pure 0/1 strategy of the article. Transaction costs may be split into a buy
and a sell leg (`buy_cost_bps`, `sell_cost_bps`), as sell-side costs -- taxes and levies on
top of the spread -- are typically the larger of the two.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_COST_BPS = 10.       # one-way transaction cost, in basis points
DEFAULT_MIN_CASH = 0.        # cash floor, held even in the favorable regime
DEFAULT_MAX_CASH = 1.        # cash cap, the share moved out of the risky asset in a bear


def resolve_cash_limits(min_cash: float = DEFAULT_MIN_CASH,
                        max_cash: float = DEFAULT_MAX_CASH) -> tuple:
    """
    Turn a pair of cash limits into the weights on the risky asset in the two regimes.

    Parameters
    ----------
    min_cash : float, optional (default=0.)
        The share of the portfolio kept in the risk-free asset even in the favorable
        regime, so that the bull weight on the risky asset is `1 - min_cash`.

    max_cash : float, optional (default=1.)
        The largest share allowed in the risk-free asset, reached in the unfavorable
        regime, so that the bear weight on the risky asset is `1 - max_cash`.

    Returns
    -------
    (float, float)
        The weights on the risky asset in the bull and in the bear regime.
    """
    min_cash, max_cash = float(min_cash), float(max_cash)
    if not 0. <= min_cash <= 1.:
        raise ValueError(f"min_cash는 0과 1 사이여야 합니다. 입력값: {min_cash}")
    if not 0. <= max_cash <= 1.:
        raise ValueError(f"max_cash는 0과 1 사이여야 합니다. 입력값: {max_cash}")
    if min_cash > max_cash:
        raise ValueError(f"min_cash({min_cash})는 max_cash({max_cash})보다 클 수 없습니다.")
    return 1. - min_cash, 1. - max_cash


def resolve_cost_bps(cost_bps: float = DEFAULT_COST_BPS,
                     buy_cost_bps: float = None,
                     sell_cost_bps: float = None) -> tuple:
    """
    Resolve the buy and sell transaction costs, in basis points.

    Parameters
    ----------
    cost_bps : float, optional (default=10.)
        The symmetric one-way cost, used for whichever leg is not given explicitly.

    buy_cost_bps, sell_cost_bps : float, optional
        The cost of the buy and of the sell leg. `None` falls back to `cost_bps`, which
        reproduces the symmetric cost of the article.

    Returns
    -------
    (float, float)
        The buy and the sell cost, in basis points.
    """
    buy = float(cost_bps if buy_cost_bps is None else buy_cost_bps)
    sell = float(cost_bps if sell_cost_bps is None else sell_cost_bps)
    if buy < 0. or sell < 0.:
        raise ValueError(f"거래비용은 0 이상이어야 합니다. 매수: {buy}, 매도: {sell}")
    return buy, sell


def trade_legs(weights: pd.Series) -> tuple:
    """
    Split the daily change of the weight on the risky asset into a buy and a sell leg.

    Parameters
    ----------
    weights : pd.Series
        The weight on the risky asset.

    Returns
    -------
    (pd.Series, pd.Series)
        The amount bought and the amount sold each day, both non-negative. The first day
        trades nothing: its allocation is the starting position, not a rebalancing.
    """
    change = weights.diff().fillna(0.)
    return change.clip(lower=0.).rename("bought"), change.clip(upper=0.).abs().rename("sold")


def build_weights(regime_ser: pd.Series,
                  delay: int = 1,
                  bull_state: int = 0,
                  min_cash: float = DEFAULT_MIN_CASH,
                  max_cash: float = DEFAULT_MAX_CASH) -> pd.Series:
    """
    Turn a regime sequence into the daily weight on the risky asset.

    Parameters
    ----------
    regime_ser : pd.Series
        The online inferred regime of each day.

    delay : int, optional (default=1)
        The trading delay, in days. The signal of day `t` is applied from day `t + delay + 1`.

    bull_state : int, optional (default=0)
        The state treated as the favorable regime, held with the largest allowed weight.

    min_cash : float, optional (default=0.)
        The cash floor, held in the risk-free asset even in the favorable regime.

    max_cash : float, optional (default=1.)
        The cash cap, the share held in the risk-free asset in the unfavorable regime.

    Returns
    -------
    pd.Series
        The weight on the risky asset, in {1 - max_cash, 1 - min_cash} -- {0., 1.} under
        the default limits. The first `delay + 1` days, for which no tradable signal exists
        yet, are held at the bull weight.
    """
    if delay < 0:
        raise ValueError(f"delay는 0 이상이어야 합니다. 입력값: {delay}")
    bull_weight, bear_weight = resolve_cash_limits(min_cash, max_cash)
    weights = pd.Series(np.where(regime_ser == bull_state, bull_weight, bear_weight),
                        index=regime_ser.index, dtype=float).shift(delay + 1)
    return weights.fillna(bull_weight).rename("weight")


def run_0_1_strategy(regime_ser: pd.Series,
                     ret_ser: pd.Series,
                     rf_ser: pd.Series,
                     delay: int = 1,
                     cost_bps: float = DEFAULT_COST_BPS,
                     bull_state: int = 0,
                     min_cash: float = DEFAULT_MIN_CASH,
                     max_cash: float = DEFAULT_MAX_CASH,
                     buy_cost_bps: float = None,
                     sell_cost_bps: float = None) -> pd.DataFrame:
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
        One-way transaction cost in basis points, charged on the traded amount. Used for
        whichever of the two legs below is not given explicitly.

    bull_state : int, optional (default=0)
        The state held with the largest allowed weight in the risky asset.

    min_cash : float, optional (default=0.)
        The cash floor, held in the risk-free asset even in the favorable regime.

    max_cash : float, optional (default=1.)
        The cash cap, the share held in the risk-free asset in the unfavorable regime.

    buy_cost_bps, sell_cost_bps : float, optional
        The cost of the buy and of the sell leg, in basis points, e.g. 10 and 25 when the
        sell side carries a transaction tax. `None` falls back to `cost_bps`.

    Returns
    -------
    pd.DataFrame
        Indexed by date, holding the regime, the weight on the risky asset, the amounts
        bought and sold, the total traded amount, the transaction cost charged, the
        risk-free rate, and the total returns of the buy-and-hold (`bh`) and 0/1 (`jm`)
        strategies.
    """
    idx = regime_ser.index
    ret, rf = ret_ser.reindex(idx), rf_ser.reindex(idx)
    if ret.isna().any() or rf.isna().any():
        raise ValueError("수익률 또는 무위험금리에 레짐 구간을 덮지 못하는 결측이 있습니다.")

    buy_bps, sell_bps = resolve_cost_bps(cost_bps, buy_cost_bps, sell_cost_bps)
    weights = build_weights(regime_ser, delay=delay, bull_state=bull_state,
                            min_cash=min_cash, max_cash=max_cash)
    # no cost on the first day: the initial allocation is the starting position
    bought, sold = trade_legs(weights)
    cost = bought * (buy_bps / 1e4) + sold * (sell_bps / 1e4)
    gross = weights * ret + (1. - weights) * rf
    return pd.DataFrame({
        "regime": regime_ser,
        "weight": weights,
        "bought": bought,
        "sold": sold,
        "traded": bought + sold,
        "cost": cost,
        "rf": rf,
        "bh": ret,
        "jm": gross - cost,
    })


def max_drawdown(ret: pd.Series) -> float:
    """Maximum drawdown of the compounded wealth of a total return series."""
    wealth = (1. + ret).cumprod()
    return float((wealth / wealth.cummax() - 1.).min())


def performance_metrics(ret: pd.Series,
                        rf: pd.Series,
                        weight: pd.Series = None,
                        trading_days: int = TRADING_DAYS,
                        cost: pd.Series = None) -> dict:
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

    cost : pd.Series, optional
        The daily transaction cost charged, needed for the annualized cost drag row.

    Returns
    -------
    dict
        Compound annual growth rate, volatility, Sharpe ratio, maximum drawdown, Calmar
        ratio, 5% expected shortfall, the annualized cost drag, and -- when `weight` is
        given -- turnover and leverage.
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
    # the transaction cost already deducted from `ret`, annualized
    metrics["Cost"] = float(cost.reindex(ret.index).mean() * trading_days) if cost is not None else 0.
    return metrics


def performance_table(strategy_df: pd.DataFrame,
                     trading_days: int = TRADING_DAYS,
                     others: dict = None,
                     label: str = "JM 0/1") -> pd.DataFrame:
    """
    Build the buy-and-hold versus 0/1 strategy comparison table.

    Parameters
    ----------
    strategy_df : pd.DataFrame
        The output of `run_0_1_strategy` for the main strategy.

    trading_days : int, optional (default=252)
        Trading days per year used for annualization.

    others : dict of {str: pd.DataFrame}, optional
        Further strategies to add as columns, e.g. `{"HMM 0/1": hmm_strategy_df}`. Each is
        evaluated over its own period, which the caller is expected to have aligned.

    label : str, optional (default="JM 0/1")
        The column name of the main strategy.

    Returns
    -------
    pd.DataFrame
        Metrics in rows, strategies in columns.
    """
    rf = strategy_df["rf"]
    table = {
        "B & H": performance_metrics(strategy_df["bh"], rf, weight=None, trading_days=trading_days),
        label: performance_metrics(strategy_df["jm"], rf, weight=strategy_df["weight"],
                                   trading_days=trading_days, cost=strategy_df.get("cost")),
    }
    for name, other in (others or {}).items():
        table[name] = performance_metrics(other["jm"], other["rf"], weight=other["weight"],
                                          trading_days=trading_days, cost=other.get("cost"))
    return pd.DataFrame(table)


def format_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    """Format the metric table for printing: percentages where the article uses them."""
    pct_rows = ("Return", "Volatility", "MDD", "ES_0.05", "Turnover", "Leverage")
    rows = {}
    for name, values in table.astype(float).iterrows():
        if name in pct_rows:
            fmt = lambda x: f"{x * 100:.1f}%"
        elif name == "Cost":        # the annual cost drag is a fraction of a percent
            fmt = lambda x: f"{x * 100:.2f}%"
        else:
            fmt = lambda x: f"{x:.2f}"
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


def delay_robustness_table(regimes: dict,
                           ret_ser: pd.Series,
                           rf_ser: pd.Series,
                           delays=(1, 5, 10),
                           cost_bps: float = DEFAULT_COST_BPS,
                           metrics=("Return", "Sharpe", "Calmar"),
                           trading_days: int = TRADING_DAYS,
                           bull_state: int = 0,
                           min_cash: float = DEFAULT_MIN_CASH,
                           max_cash: float = DEFAULT_MAX_CASH,
                           buy_cost_bps: float = None,
                           sell_cost_bps: float = None) -> pd.DataFrame:
    """
    Build the trading delay robustness table of Table 5 of the article.

    Each regime sequence is traded under several delays, so that the decay of performance
    with a slower implementation becomes visible; the buy-and-hold benchmark, which never
    trades, is reported once.

    Parameters
    ----------
    regimes : dict of {str: pd.Series}
        The regime sequences to compare, e.g. `{"JM": jm_regimes, "HMM": hmm_regimes}`.

    ret_ser : pd.Series
        The total return series of the risky asset.

    rf_ser : pd.Series
        The daily risk-free rate.

    delays : iterable of int, optional (default=(1, 5, 10))
        The trading delays, in days. The signal of day `t` is applied from `t + delay + 1`.

    cost_bps : float, optional (default=10.)
        One-way transaction cost in basis points, used for whichever leg is not given
        explicitly below.

    metrics : iterable of str, optional (default=("Return", "Sharpe", "Calmar"))
        The rows of the table, taken from the keys of `performance_metrics`.

    trading_days : int, optional (default=252)
        Trading days per year used for annualization.

    bull_state : int, optional (default=0)
        The state held with the largest allowed weight in the risky asset.

    min_cash, max_cash : float, optional (defaults=0. and 1.)
        The cash limits applied to every strategy, see `resolve_cash_limits`.

    buy_cost_bps, sell_cost_bps : float, optional
        The cost of the buy and of the sell leg, in basis points, see `resolve_cost_bps`.

    Returns
    -------
    pd.DataFrame
        Metrics in rows; columns are a MultiIndex of (model, delay), starting with the
        buy-and-hold column. All models are evaluated over the common period covered by
        every regime sequence.
    """
    if not regimes:
        raise ValueError("비교할 레짐 시퀀스가 없습니다.")
    # a common evaluation period, so that the columns are comparable
    index = None
    for regime_ser in regimes.values():
        index = regime_ser.index if index is None else index.intersection(regime_ser.index)
    if len(index) == 0:
        raise ValueError("모델들이 공통으로 덮는 기간이 없습니다.")

    columns, data = [], []
    bh_ret, bh_rf = ret_ser.reindex(index), rf_ser.reindex(index)
    if bh_ret.isna().any() or bh_rf.isna().any():
        raise ValueError("수익률 또는 무위험금리에 비교 구간을 덮지 못하는 결측이 있습니다.")
    columns.append(("B & H", ""))
    data.append(performance_metrics(bh_ret, bh_rf, weight=None, trading_days=trading_days))
    for name, regime_ser in regimes.items():
        for delay in delays:
            strategy = run_0_1_strategy(regime_ser.reindex(index), ret_ser, rf_ser,
                                        delay=delay, cost_bps=cost_bps, bull_state=bull_state,
                                        min_cash=min_cash, max_cash=max_cash,
                                        buy_cost_bps=buy_cost_bps, sell_cost_bps=sell_cost_bps)
            columns.append((name, delay))
            data.append(performance_metrics(strategy["jm"], strategy["rf"],
                                            weight=strategy["weight"], trading_days=trading_days,
                                            cost=strategy["cost"]))
    table = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(columns, names=["model", "delay"])).T
    return table.loc[[metric for metric in metrics if metric in table.index]]
