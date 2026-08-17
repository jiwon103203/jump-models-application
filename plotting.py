"""
Plotting helpers for the rolling jump model pipeline.

These figures are drawn with plain `matplotlib` rather than `jumpmodels.plot`, whose
publication settings require a local LaTeX installation.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BULL_COLOR, BEAR_COLOR = "#2b8a3e", "#c92a2a"


def _percent_axis(ax: plt.Axes) -> None:
    """Format the y-axis of `ax` as percentages."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))


def _save(fig: plt.Figure, filepath: str) -> str:
    """Save `fig` to `filepath`, creating the folder if needed, and close it."""
    folder = os.path.dirname(os.path.abspath(filepath))
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_regimes_and_cumret(strategy_df: pd.DataFrame,
                            filepath: str,
                            title: str = "JM-guided 0/1 strategy",
                            figsize=(16, 7)) -> str:
    """
    Plot the cumulative excess returns of the buy-and-hold and 0/1 strategies, with the
    periods spent in the risk-free asset shaded in red.

    Parameters
    ----------
    strategy_df : pd.DataFrame
        The output of `backtest.run_0_1_strategy`.

    filepath : str
        Where to save the figure.

    title : str, optional
        The figure title.

    figsize : tuple, optional
        The figure size, in inches.

    Returns
    -------
    str
        The path of the saved figure.
    """
    dates = pd.to_datetime(pd.Index(strategy_df.index))
    rf = strategy_df["rf"]
    fig, ax = plt.subplots(figsize=figsize)
    for col, label, color in (("bh", "Buy & hold", "#1c7ed6"), ("jm", "JM 0/1 strategy", "#f08c00")):
        ax.plot(dates, (strategy_df[col] - rf).cumsum(), label=label, color=color, lw=1.4)

    # shade the days actually held in the risk-free asset (signal + trading delay applied)
    in_cash = (strategy_df["weight"] < .5).to_numpy()
    ax.fill_between(dates, 0, 1, where=in_cash, transform=ax.get_xaxis_transform(),
                    color=BEAR_COLOR, alpha=.18, step="pre", label="Bear (in cash)")

    ax.set(title=title, ylabel="Cumulative excess return")
    _percent_axis(ax)
    ax.grid(alpha=.25)
    ax.legend(loc="upper left")
    return _save(fig, filepath)


def annualize_center(feature: str, values, trading_days: int = 252):
    """
    Scale a centroid to annualized units, following the convention of Figure 3 of the article:
    the downside deviation is multiplied by sqrt(2) and the Sortino ratio divided by sqrt(2),
    so that both are comparable with a volatility and a Sharpe ratio, before annualizing.

    Parameters
    ----------
    feature : str
        The feature name, used to pick the scaling rule.

    values : array-like
        The centroid values of that feature.

    trading_days : int, optional (default=252)
        Trading days per year.

    Returns
    -------
    (array-like, bool)
        The scaled values, and whether any scaling was applied.
    """
    if feature.startswith("DD-log"):
        return values, False        # already on a log scale: leave it alone
    if feature.startswith("DD"):
        return values * np.sqrt(2. * trading_days), True
    if feature.startswith("sortino"):
        return values * np.sqrt(trading_days / 2.), True
    if feature.startswith("ret"):
        return values * trading_days, True
    return values, False


def plot_refit_params(params: pd.DataFrame,
                      feature_names: list,
                      filepath: str,
                      title: str = "Estimated centroids by regime from the rolling JM fit",
                      annualize: bool = True,
                      figsize=(16, 3.2)) -> str:
    """
    Plot how the estimated centroid of each feature evolves across re-estimations, one panel
    per feature and one line per regime (the counterpart of Figure 3 of the article).

    Parameters
    ----------
    params : pd.DataFrame
        The `params` table of `rolling.RollingJMResult`.

    feature_names : list of str
        The feature columns to plot.

    filepath : str
        Where to save the figure.

    title : str, optional
        The figure title.

    annualize : bool, optional (default=True)
        Whether to plot the centroids in annualized units, see `annualize_center`.

    figsize : tuple, optional
        The size of a single panel, in inches; the height is multiplied by the panel count.

    Returns
    -------
    str
        The path of the saved figure.
    """
    n_feat = len(feature_names)
    fig, axes = plt.subplots(n_feat, 1, figsize=(figsize[0], figsize[1] * n_feat), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feature_names):
        scaled = False
        for regime, color in (("bull", BULL_COLOR), ("bear", BEAR_COLOR)):
            sub = params[params.regime == regime]
            if sub.empty:
                continue
            values = sub[f"center_{feat}"].to_numpy()
            if annualize:
                values, scaled = annualize_center(feat, values)
            ax.plot(pd.to_datetime(sub.refit_date), values,
                    marker="o", ms=3, lw=1.3, color=color, label=regime)
        ax.set(ylabel=f"{feat} (ann.)" if scaled else feat)
        ax.grid(alpha=.25)
    axes[0].set_title(title)
    axes[0].legend(loc="upper left")
    return _save(fig, filepath)


def plot_weights(strategy_df: pd.DataFrame, filepath: str, figsize=(16, 3.)) -> str:
    """
    Plot the weight on the risky asset over time.

    Parameters
    ----------
    strategy_df : pd.DataFrame
        The output of `backtest.run_0_1_strategy`.

    filepath : str
        Where to save the figure.

    figsize : tuple, optional
        The figure size, in inches.

    Returns
    -------
    str
        The path of the saved figure.
    """
    dates = pd.to_datetime(pd.Index(strategy_df.index))
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(dates, strategy_df["weight"], where="post", color="#f08c00", lw=1.2)
    ax.set(title="Weight on the risky asset", ylabel="Weight", ylim=(-.05, 1.05))
    _percent_axis(ax)
    ax.grid(alpha=.25)
    return _save(fig, filepath)
