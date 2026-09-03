"""
The investment simulation of the article: the "TopK strategy", which "involves investing in
the TopK stocks with the highest predicted scores each trading day and selling them after
holding ``dt`` days, where ``dt`` represents the model's prediction period", with
``TopK = 30``, ``dt = 10`` and a transaction cost of 0.3% in the paper's backtest, against
an equally weighted portfolio of the same universe as the benchmark.

Holding for ``dt`` days while opening a new position every day means ``dt`` overlapping
tranches are alive at any time. Each tranche is an equally weighted basket of the day's
TopK stocks, bought at the *next* day's price and sold ``dt`` days later -- the same
buy-tomorrow convention the labels use, so that a tranche's total return is exactly the
label the model is trained on. The strategy's daily return is the average over the tranches
that are alive that day.

The transaction cost rate the article quotes is not split into a buy and a sell leg;
``cost_mode`` covers both readings ("per_side", the default, charges the rate on the way in
and again on the way out; "round_trip" splits one rate over the two legs).
"""

import numpy as np
import pandas as pd

from metrics import cumulative_excess_return, cumulative_return, investment_metrics

DEFAULT_TOPK = 30
DEFAULT_HOLDING = 10
DEFAULT_COST = 0.003
COST_MODES = ("per_side", "round_trip")


def resolve_costs(cost=DEFAULT_COST, cost_mode="per_side"):
    """``(buy_cost, sell_cost)`` from the quoted rate and the reading of it."""
    if cost_mode not in COST_MODES:
        raise ValueError(f"cost_mode must be one of {COST_MODES}, got {cost_mode!r}")
    if cost < 0:
        raise ValueError("the transaction cost cannot be negative")
    return (cost, cost) if cost_mode == "per_side" else (cost / 2, cost / 2)


def price_frame_from_panel(panel, field=None):
    """The wide price frame (dates x stocks) the backtest trades on, taken from a `PanelData`."""
    return panel.field_frame(panel.config.price_field if field is None else field)


def daily_returns(price_frame):
    """Simple daily returns of every stock, ``price_d / price_{d-1} - 1``."""
    price_frame = price_frame.sort_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = price_frame.astype(float).pct_change()
    return returns.replace([np.inf, -np.inf], np.nan)


def run_topk_backtest(predictions, price_frame, pred_col, topk=DEFAULT_TOPK,
                      holding=DEFAULT_HOLDING, cost=DEFAULT_COST, cost_mode="per_side",
                      universe=None, date_col="date", stock_col="stock"):
    """Run the TopK strategy and return its daily return series next to the benchmark's.

    ``predictions`` is the long-format frame `train.predict` produces, ``price_frame`` the
    wide price frame to trade on, and ``pred_col`` the score column (``"pred_10"`` for the
    10-day model the article backtests). ``universe`` optionally restricts both the stock
    selection and the equally weighted benchmark to an index's constituents -- a set of
    stocks, or a mapping from date to the set of stocks in the index that day.

    The result has one row per trading day, with ``strategy``, ``benchmark``, ``excess`` and
    their cumulative curves.
    """
    if pred_col not in predictions.columns:
        raise ValueError(f"the prediction frame has no column {pred_col!r}")
    if topk < 1:
        raise ValueError("topk must be at least 1")
    if holding < 1:
        raise ValueError("the holding period must be at least one day")
    buy_cost, sell_cost = resolve_costs(cost, cost_mode)

    price_frame = price_frame.sort_index()
    price_frame.index = pd.DatetimeIndex(pd.to_datetime(price_frame.index))
    returns = daily_returns(price_frame)
    dates = price_frame.index
    position_of = pd.Series(np.arange(len(dates)), index=dates)

    frame = predictions[[date_col, stock_col, pred_col]].dropna().copy()
    frame[date_col] = pd.DatetimeIndex(pd.to_datetime(frame[date_col]))
    frame = frame[frame[date_col].isin(dates)]

    def in_universe(date, stocks):
        if universe is None:
            return np.ones(len(stocks), dtype=bool)
        allowed = universe.get(date) if isinstance(universe, dict) else universe
        if allowed is None:
            return np.zeros(len(stocks), dtype=bool)
        return np.array([s in allowed for s in stocks], dtype=bool)

    # tranche_returns[d] collects the daily return of every tranche alive on day d
    tranche_returns = {d: [] for d in range(len(dates))}
    for date, group in frame.groupby(date_col, sort=True):
        signal_position = int(position_of[date])
        stocks = group[stock_col].to_numpy()
        keep = in_universe(date, stocks)
        group = group.loc[keep]
        if group.empty:
            continue
        chosen = group.nlargest(min(topk, len(group)), pred_col)[stock_col].to_numpy()
        # bought at the next day's price, sold `holding` days later: the tranche earns the
        # daily returns of days signal + 2 ... signal + holding + 1
        start, end = signal_position + 2, signal_position + holding + 1
        if end >= len(dates):
            continue
        available = [s for s in chosen if s in returns.columns]
        if not available:
            continue
        for day in range(start, end + 1):
            day_return = returns.iloc[day].loc[available].astype(float)
            day_return = day_return.fillna(0.0).mean()
            if day == start:
                day_return = (1 + day_return) * (1 - buy_cost) - 1
            if day == end:
                day_return = (1 + day_return) * (1 - sell_cost) - 1
            tranche_returns[day].append(day_return)

    benchmark_mask = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    if universe is not None:
        for date in returns.index:
            benchmark_mask.loc[date] = in_universe(date, returns.columns.to_numpy())
    benchmark = returns.where(benchmark_mask).mean(axis=1, skipna=True)

    strategy = pd.Series({dates[d]: (np.mean(v) if v else np.nan)
                          for d, v in tranche_returns.items()}).sort_index()
    result = pd.DataFrame({"strategy": strategy, "benchmark": benchmark.reindex(strategy.index)})
    result = result.dropna(subset=["strategy"])
    result["benchmark"] = result["benchmark"].fillna(0.0)
    result["excess"] = result["strategy"] - result["benchmark"]
    result["CR"] = cumulative_return(result["strategy"]).to_numpy()
    result["benchmark_CR"] = cumulative_return(result["benchmark"]).to_numpy()
    result["CER"] = cumulative_excess_return(result["strategy"], result["benchmark"]).to_numpy()
    return result.rename_axis("date")


def backtest_metrics(backtest_frame):
    """``AR``, ``IR``, ``RoMaD`` and the drawdown of a `run_topk_backtest` result."""
    return investment_metrics(backtest_frame["strategy"], backtest_frame["benchmark"])
