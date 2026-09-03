"""
A synthetic market panel, so that the whole pipeline can be run end to end without the
China A-shares data of the article.

The generator mirrors the structure FactorGCL assumes: returns are driven by ``K`` *prior*
factors that stocks load on through their industry membership, by ``M`` *hidden* factors
with continuous loadings that cut across industries, and by an idiosyncratic term. The
factor returns are autocorrelated, which is what makes future returns partly predictable
from the past window -- with i.i.d. factor returns no model could beat an IC of zero, and
the run would say nothing about the implementation.

The panel it emits has exactly the columns `data.PanelData` expects, plus the industry map
for the prior factor exposures.
"""

import numpy as np
import pandas as pd

from data import FEATURE_FIELDS


def make_synthetic_panel(n_stocks=120, n_days=900, n_industries=8, n_hidden_factors=4,
                         start="2015-01-01", factor_ar=0.3, hidden_ar=0.3,
                         factor_vol=0.010, hidden_vol=0.008, idio_vol=0.015,
                         idio_ar=0.0, seed=0):
    """Return ``(panel, industry)``.

    ``panel`` is the long-format price-volume frame (``date``, ``stock`` and the six fields
    of `data.FEATURE_FIELDS`); ``industry`` is a Series mapping each stock to its industry,
    the prior factor of the experiments.

    ``factor_ar`` and ``hidden_ar`` are the autocorrelations of the factor returns: the
    higher they are, the more of tomorrow's return is visible in the past window and the
    higher the achievable IC.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    stocks = np.array([f"S{i:04d}" for i in range(n_stocks)])
    industries = np.array([f"IND{i:02d}" for i in range(n_industries)])
    membership = rng.integers(0, n_industries, size=n_stocks)

    # exposures: 0/1 on the industry (the prior factors), continuous on the hidden factors
    prior_beta = np.zeros((n_stocks, n_industries))
    prior_beta[np.arange(n_stocks), membership] = 1.0
    hidden_beta = rng.uniform(0.0, 1.0, size=(n_stocks, n_hidden_factors))

    def ar1(n_series, rho, vol):
        shocks = rng.normal(0.0, vol, size=(n_days, n_series))
        series = np.zeros((n_days, n_series))
        series[0] = shocks[0]
        for t in range(1, n_days):
            series[t] = rho * series[t - 1] + np.sqrt(max(1 - rho ** 2, 0.0)) * shocks[t]
        return series

    prior_returns = ar1(n_industries, factor_ar, factor_vol)
    hidden_returns = ar1(n_hidden_factors, hidden_ar, hidden_vol)
    idio = ar1(n_stocks, idio_ar, idio_vol)
    returns = prior_returns @ prior_beta.T + hidden_returns @ hidden_beta.T + idio

    close = 10.0 * np.exp(np.cumsum(returns, axis=0))
    # intraday levels around the close, and a volume that reacts to the size of the move
    noise = lambda scale: rng.normal(0.0, scale, size=close.shape)
    open_ = close * (1 + noise(0.004))
    high = np.maximum(open_, close) * (1 + np.abs(noise(0.003)))
    low = np.minimum(open_, close) * (1 - np.abs(noise(0.003)))
    vwap = (open_ + close + high + low) / 4
    base_volume = rng.uniform(5e5, 5e6, size=n_stocks)
    volume = base_volume[None, :] * np.exp(2.0 * np.abs(returns) + noise(0.3))

    fields = {"open": open_, "high": high, "low": low, "close": close, "vwap": vwap,
              "volume": volume}
    panel = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n_stocks),
        "stock": np.tile(stocks, n_days),
        **{name: fields[name].reshape(-1) for name in FEATURE_FIELDS},
    })
    industry = pd.Series(industries[membership], index=stocks, name="industry")
    return panel, industry
