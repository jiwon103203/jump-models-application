"""
The evaluation metrics of the article and its supplementary material.

Prediction metrics ("Evaluation Metrics" of the supplement):

- ``IC``   -- the daily cross-sectional information coefficient, averaged over the test days;
- ``ICIR`` -- that average divided by the standard deviation of the daily ICs;
- ``MSE``  -- the mean squared error of the (cross-sectionally standardised) returns;
- ``F1``   -- the harmonic mean of a precision and a recall defined on the *sign* of the
  return, with the counts ``TP = #(y > 0 and y_hat > 0)``, ``FP = #(y < 0 and y_hat > 0)``
  and ``FN = #(y > 0 and y_hat < 0)`` of the supplement.

Investment metrics: the cumulative return ``CR`` and cumulative excess return ``CER`` are
*arithmetic* sums of daily returns in the supplement, and ``AR``, ``IR`` and ``RoMaD`` are
derived from ``CER``. ``backtest.run_topk_backtest`` produces the return series they take.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _as_1d(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    return a


def daily_ic(y_true, y_pred):
    """The cross-sectional IC of one trading day.

    The supplement writes it as the demeaned inner product over ``N`` divided by the product
    of the standard deviations, which is the Pearson correlation with the population
    (``ddof = 0``) standard deviation. Returns ``nan`` for a cross-section with fewer than
    two usable stocks or with no dispersion, so that such days drop out of the average.
    """
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    usable = np.isfinite(y_true) & np.isfinite(y_pred)
    if usable.sum() < 2:
        return np.nan
    y_true, y_pred = y_true[usable], y_pred[usable]
    std_true, std_pred = y_true.std(ddof=0), y_pred.std(ddof=0)
    if std_true <= 0 or std_pred <= 0:
        return np.nan
    centred = (y_pred - y_pred.mean()) @ (y_true - y_true.mean()) / y_true.shape[0]
    return float(centred / (std_pred * std_true))


def ic_series(df, date_col="date", label_col="label", pred_col="pred"):
    """The series of daily ICs, indexed by date."""
    grouped = df.groupby(date_col, sort=True)
    return pd.Series({date: daily_ic(g[label_col].to_numpy(), g[pred_col].to_numpy())
                      for date, g in grouped}, name="IC").astype(float)


def ic_icir(df, date_col="date", label_col="label", pred_col="pred"):
    """``(IC, ICIR)``: the mean of the daily ICs and that mean over their standard deviation."""
    ics = ic_series(df, date_col, label_col, pred_col).dropna()
    if ics.empty:
        return float("nan"), float("nan")
    mean = float(ics.mean())
    std = float(ics.std(ddof=0))
    return mean, (float("nan") if std == 0 else mean / std)


def mse(y_true, y_pred):
    """The mean squared error over every usable observation."""
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    usable = np.isfinite(y_true) & np.isfinite(y_pred)
    if usable.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[usable] - y_pred[usable]) ** 2))


def f1_score(y_true, y_pred):
    """The sign-based F1 score of the supplement.

    Note that the counts are the supplement's: a positive prediction on a stock whose return
    is exactly zero is neither a true nor a false positive, and only strictly positive
    labels can be missed.
    """
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    usable = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[usable], y_pred[usable]
    tp = float(np.sum((y_true > 0) & (y_pred > 0)))
    fp = float(np.sum((y_true < 0) & (y_pred > 0)))
    fn = float(np.sum((y_true > 0) & (y_pred < 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def prediction_metrics(df, date_col="date", label_col="label", pred_col="pred"):
    """``IC``, ``ICIR``, ``MSE`` and ``F1`` for one forward prediction period."""
    ic, icir = ic_icir(df, date_col, label_col, pred_col)
    labels, preds = df[label_col].to_numpy(), df[pred_col].to_numpy()
    return {"IC": ic, "ICIR": icir, "MSE": mse(labels, preds), "F1": f1_score(labels, preds)}


def cumulative_return(returns):
    """``CR(t) = sum_{i<=t} r(i)`` -- the arithmetic accumulation the supplement defines."""
    return pd.Series(returns).astype(float).cumsum()


def cumulative_excess_return(returns, benchmark_returns):
    """``CER(t) = sum_{i<=t} (r(i) - r_b(i))``."""
    excess = pd.Series(returns).astype(float) - pd.Series(benchmark_returns).astype(float)
    return excess.cumsum()


def max_drawdown(curve):
    """The maximum observed loss from a peak to a trough of ``curve``, before a new peak.

    ``curve`` is a cumulative (additive) return curve, so the drawdown is measured in return
    points rather than as a ratio, matching the ``CER`` curve ``RoMaD`` divides into.
    """
    curve = pd.Series(curve).astype(float)
    if curve.empty:
        return float("nan")
    # the curve starts at zero invested capital, so the running peak starts there too
    running_peak = pd.concat([pd.Series([0.0]), curve]).cummax()
    drawdown = running_peak - pd.concat([pd.Series([0.0]), curve])
    return float(drawdown.max())


def investment_metrics(returns, benchmark_returns, trading_days_per_year=TRADING_DAYS_PER_YEAR):
    """``AR``, ``IR`` and ``RoMaD`` of the cumulative excess return, plus the raw curves.

    - ``AR   = CER(T) / T * 252``, the annualised excess return;
    - ``IR   = mean(excess) / std(excess) * sqrt(252)``;
    - ``RoMaD = AR / MaxDrawdown(CER)``.
    """
    returns = pd.Series(returns).astype(float)
    benchmark_returns = pd.Series(benchmark_returns).astype(float).reindex(returns.index)
    excess = (returns - benchmark_returns).dropna()
    n_days = len(excess)
    if n_days == 0:
        return {"AR": float("nan"), "IR": float("nan"), "RoMaD": float("nan"),
                "MaxDrawdown": float("nan"), "CR": float("nan"), "CER": float("nan")}
    cer = excess.cumsum()
    annualised = float(cer.iloc[-1] / n_days * trading_days_per_year)
    std = float(excess.std(ddof=1))
    ir = float("nan") if std == 0 else float(excess.mean() / std * np.sqrt(trading_days_per_year))
    mdd = max_drawdown(cer)
    romad = float("nan") if not mdd > 0 else annualised / mdd
    return {"AR": annualised, "IR": ir, "RoMaD": romad, "MaxDrawdown": mdd,
            "CR": float(returns.sum()), "CER": float(cer.iloc[-1])}
