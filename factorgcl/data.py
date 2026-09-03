"""
The data pipeline of FactorGCL, following the "Data Preprocessing" section of the
supplementary material.

The raw input is a long-format panel of day-level price-volume data -- one row per
``(date, stock)`` with the six fields ``open``, ``high``, ``low``, ``close``, ``vwap`` and
``volume`` (``D = 6``) -- plus a mapping from stock to prior factor, the secondary industry
in the article's experiments (83 industries, a 0/1 exposure).

For a trading day ``t`` one *sample* is the whole cross-section of that day:

- ``x``       ``(N, T, D)``  the past ``T = 60`` days of data, ending at ``t``;
- ``x_future`` ``(N, T', D)`` the following ``T' = 20`` days, used only by the contrastive
  branch during training;
- ``beta``    ``(N, K)``     the prior factor exposures, the incidence matrix of ``G_p``;
- ``label``   ``(N, L)``     the future returns over the ``L`` horizons.

Dimensionless processing, per the supplement: the five price fields are divided by the
current closing price and the trading volume by the average trading volume, so a sample
carries relative prices and a relative volume rather than levels.

The labels are the multi-period future returns

    y~_t = (price_{t+dt+1} - price_{t+1}) / price_{t+1},   price = VWAP

cross-sectionally standardised, ``y_t = (y~_t - mu_t) / sigma_t``.

Two preprocessing steps are described in the supplement only as "drop samples with too many
missing values, fill missing values with 0, and clip extreme values"; the thresholds are
this implementation's, and are exposed as `PreprocessConfig` fields:
``max_missing_ratio``, ``feature_clip`` and ``label_mad_clip``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# the six day-level price-volume fields, in the channel order of the (N, T, D) tensors
PRICE_FIELDS = ("open", "high", "low", "close", "vwap")
VOLUME_FIELDS = ("volume",)
FEATURE_FIELDS = PRICE_FIELDS + VOLUME_FIELDS

DEFAULT_SEQ_LEN = 60          # T,  the length of the historical sequence
DEFAULT_FUTURE_LEN = 20       # T', the length of the future sequence
DEFAULT_PERIODS = (1, 5, 10, 20)   # the forward prediction periods (Delta t)

FUTURE_NORM_MODES = ("anchor", "window")


@dataclass
class PreprocessConfig:
    """Every knob of the preprocessing, with the article's values as defaults."""

    seq_len: int = DEFAULT_SEQ_LEN
    future_len: int = DEFAULT_FUTURE_LEN
    periods: tuple = DEFAULT_PERIODS
    price_field: str = "vwap"          # the price the labels are computed on
    # a stock-day whose historical window misses more than this fraction of its values is
    # dropped from the cross-section ("drop samples with too many missing values")
    max_missing_ratio: float = 0.2
    # normalised features are clipped to +/- this value ("clip extreme values")
    feature_clip: float = 10.0
    # raw cross-sectional returns are winsorised at median +/- this many median absolute
    # deviations before being standardised; set to None to skip
    label_mad_clip: float = 5.0
    # how the future window is made dimensionless: "anchor" reuses the anchor day's closing
    # price and the historical average volume, keeping both branches on one scale; "window"
    # normalises the future window by its own last close and its own average volume
    future_norm: str = "anchor"
    # the smallest cross-section worth keeping as a training sample
    min_stocks_per_day: int = 2

    def __post_init__(self):
        if self.future_norm not in FUTURE_NORM_MODES:
            raise ValueError(f"future_norm must be one of {FUTURE_NORM_MODES}")
        self.periods = tuple(int(p) for p in self.periods)
        if not self.periods or min(self.periods) < 1:
            raise ValueError("the forward prediction periods must be positive integers")


def pivot_panel(df, date_col="date", stock_col="stock", fields=FEATURE_FIELDS):
    """Turn the long-format panel into one ``(n_dates, n_stocks)`` matrix per field.

    Returns ``(dates, stocks, matrices)`` with ``dates`` sorted and ``matrices`` a dict of
    float arrays holding ``nan`` wherever a stock has no row on a date (a suspension, or a
    listing that starts later).
    """
    missing = [c for c in (date_col, stock_col) if c not in df.columns]
    missing += [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"the panel is missing the columns {missing}")
    df = df.sort_values([date_col, stock_col])
    dates = np.array(sorted(df[date_col].unique()))
    stocks = np.array(sorted(df[stock_col].unique()))
    date_index = pd.Index(dates)
    stock_index = pd.Index(stocks)
    rows = date_index.get_indexer(df[date_col])
    cols = stock_index.get_indexer(df[stock_col])
    matrices = {}
    for f in fields:
        m = np.full((len(dates), len(stocks)), np.nan, dtype=np.float64)
        m[rows, cols] = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=np.float64)
        matrices[f] = m
    return dates, stocks, matrices


def build_prior_exposures(industry, stocks, factors=None):
    """The prior factor exposure matrix ``beta`` of shape ``(N, K)``.

    ``industry`` maps a stock to its prior factor (its secondary industry in the article);
    it may be a dict, a Series indexed by stock, or a DataFrame with ``stock`` and
    ``industry`` columns. The exposure is 1 when the stock belongs to the factor and 0
    otherwise. ``factors`` fixes the column order -- pass the full universe of industries so
    that every rolling window shares one factor space.

    Returns ``(beta, factors)``. A stock with an unknown industry gets an all-zero row; the
    hypergraph convolution treats it as an isolated node, which contributes nothing to and
    receives nothing from the prior hypergraph.
    """
    if isinstance(industry, pd.DataFrame):
        industry = industry.set_index("stock")["industry"]
    industry = pd.Series(industry, dtype=object)
    values = industry.reindex(pd.Index(stocks))
    if factors is None:
        factors = np.array(sorted({v for v in values.dropna().unique()}))
    else:
        factors = np.asarray(factors)
    factor_index = pd.Index(factors)
    beta = np.zeros((len(stocks), len(factors)), dtype=np.float32)
    positions = factor_index.get_indexer(values.fillna("__unknown__"))
    known = positions >= 0
    beta[np.flatnonzero(known), positions[known]] = 1.0
    return beta, factors


def compute_raw_labels(price, periods):
    """The raw multi-period future returns ``(price_{t+dt+1} - price_{t+1}) / price_{t+1}``.

    ``price`` is the ``(n_dates, n_stocks)`` matrix of the label price (VWAP in the article);
    the result is ``(n_dates, n_stocks, len(periods))`` with ``nan`` where the horizon runs
    past the end of the sample.
    """
    n_dates, n_stocks = price.shape
    labels = np.full((n_dates, n_stocks, len(periods)), np.nan)
    base = np.full_like(price, np.nan)
    base[:-1] = price[1:]                       # price_{t+1}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, dt in enumerate(periods):
            future = np.full_like(price, np.nan)
            if dt + 1 < n_dates:
                future[:-(dt + 1)] = price[dt + 1:]     # price_{t+dt+1}
            valid = np.isfinite(base) & (base > 0) & np.isfinite(future)
            ret = np.where(valid, (future - base) / np.where(base == 0, np.nan, base), np.nan)
            labels[:, :, k] = ret
    return labels


def standardize_labels(raw_labels, mad_clip=5.0):
    """Cross-sectional standardisation of the raw returns, one date and horizon at a time.

    Extreme values are winsorised first, at the cross-sectional median plus or minus
    ``mad_clip`` scaled median absolute deviations (``1.4826 * MAD``, the normal-consistent
    scaling), so that a single limit-up stock cannot dominate the mean and the standard
    deviation. Pass ``None`` to standardise the raw returns as they are.
    """
    out = np.full_like(raw_labels, np.nan)
    for k in range(raw_labels.shape[2]):
        panel = raw_labels[:, :, k]
        for i in range(panel.shape[0]):
            row = panel[i]
            usable = np.isfinite(row)
            if usable.sum() < 2:
                continue
            values = row[usable]
            if mad_clip is not None:
                median = np.median(values)
                mad = np.median(np.abs(values - median)) * 1.4826
                if mad > 0:
                    values = np.clip(values, median - mad_clip * mad, median + mad_clip * mad)
            mean, std = values.mean(), values.std(ddof=0)
            if std <= 0:
                continue
            standardized = np.full_like(row, np.nan)
            standardized[usable] = (values - mean) / std
            out[i, :, k] = standardized
    return out


def _normalize_window(window, price_base, volume_base, feature_clip):
    """Make one ``(N, T, D)`` window dimensionless.

    ``window`` holds raw levels in the channel order of `FEATURE_FIELDS`; the price channels
    are divided by ``price_base`` and the volume channel by ``volume_base``, both ``(N, 1)``.
    Remaining missing values are filled with 0 and the result is clipped, as the supplement
    prescribes.
    """
    out = np.array(window, dtype=np.float32, copy=True)
    n_price = len(PRICE_FIELDS)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_price = np.where(np.isfinite(price_base) & (price_base > 0), price_base, np.nan)
        out[:, :, :n_price] = out[:, :, :n_price] / safe_price[:, :, None]
        safe_volume = np.where(np.isfinite(volume_base) & (volume_base > 0), volume_base, np.nan)
        out[:, :, n_price:] = out[:, :, n_price:] / safe_volume[:, :, None]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if feature_clip is not None:
        out = np.clip(out, -feature_clip, feature_clip)
    return out


@dataclass
class CrossSection:
    """One trading day's sample: the cross-section of stocks with everything the model needs."""

    date: object
    stocks: np.ndarray                  # (N,) the stock identifiers, the node order
    x: np.ndarray                       # (N, T, D)
    beta: np.ndarray                    # (N, K)
    label: np.ndarray                   # (N, L)
    label_mask: np.ndarray              # (N, L) True where the label is usable
    x_future: np.ndarray = None         # (N, T', D), None outside the training range
    raw_label: np.ndarray = None        # (N, L) the returns before standardisation


class PanelData:
    """The whole panel, prepared once and sliced into per-day `CrossSection` samples.

    Windows are cut on demand from the wide matrices rather than materialised for every
    stock-day, so memory stays at the size of the panel itself.
    """

    def __init__(self, df, industry, config=None, date_col="date", stock_col="stock",
                 factors=None):
        self.config = config or PreprocessConfig()
        self.dates, self.stocks, self._matrices = pivot_panel(df, date_col, stock_col)
        self.beta, self.factors = build_prior_exposures(industry, self.stocks, factors)
        # (n_dates, n_stocks, D) in the channel order of FEATURE_FIELDS
        self._features = np.stack([self._matrices[f] for f in FEATURE_FIELDS], axis=2)
        price = self._matrices[self.config.price_field]
        self.raw_labels = compute_raw_labels(price, self.config.periods)
        self.labels = standardize_labels(self.raw_labels, self.config.label_mad_clip)
        self._close = self._matrices["close"]
        self._volume = self._matrices["volume"]

    @property
    def num_periods(self):
        return len(self.config.periods)

    @property
    def num_factors(self):
        return self.beta.shape[1]

    @property
    def input_size(self):
        return len(FEATURE_FIELDS)

    def field_frame(self, field):
        """One raw field as a wide frame (dates x stocks) -- what the backtest trades on."""
        if field not in self._matrices:
            raise ValueError(f"unknown field {field!r}, expected one of {FEATURE_FIELDS}")
        return pd.DataFrame(self._matrices[field],
                            index=pd.DatetimeIndex(pd.to_datetime(self.dates)),
                            columns=self.stocks)

    def usable_date_indices(self, need_future=False, need_label=True):
        """The anchor days that have a full historical window -- and, when asked, a full
        future window and at least one usable label."""
        cfg = self.config
        n_dates = len(self.dates)
        first = cfg.seq_len - 1
        last = n_dates - 1
        if need_future:
            last = min(last, n_dates - 1 - cfg.future_len)
        if need_label:
            last = min(last, n_dates - 1 - (max(cfg.periods) + 1))
        return [i for i in range(first, last + 1)]

    def valid_stock_mask(self, index, need_future=False, need_label=True):
        """Which stocks take part in the cross-section of the day at ``index``."""
        cfg = self.config
        window = self._features[index - cfg.seq_len + 1: index + 1]          # (T, n_stocks, D)
        missing_ratio = np.isnan(window).mean(axis=(0, 2))
        mask = missing_ratio <= cfg.max_missing_ratio
        anchor_close = self._close[index]
        mask &= np.isfinite(anchor_close) & (anchor_close > 0)
        if need_label:
            mask &= np.isfinite(self.labels[index]).any(axis=1)
        if need_future:
            future = self._features[index + 1: index + 1 + cfg.future_len]
            if future.shape[0] < cfg.future_len:
                return np.zeros_like(mask)
            mask &= np.isnan(future).mean(axis=(0, 2)) <= cfg.max_missing_ratio
        return mask

    def cross_section(self, index, with_future=False, need_label=True):
        """Build the `CrossSection` of the trading day at ``index``, or ``None`` when too few
        stocks survive the missing-value rules."""
        cfg = self.config
        mask = self.valid_stock_mask(index, need_future=with_future, need_label=need_label)
        if mask.sum() < cfg.min_stocks_per_day:
            return None
        cols = np.flatnonzero(mask)
        # (T, n_sel, D) -> (n_sel, T, D)
        window = np.transpose(self._features[index - cfg.seq_len + 1: index + 1, cols], (1, 0, 2))
        anchor_close = self._close[index, cols][:, None]                     # (n_sel, 1)
        volume_window = self._volume[index - cfg.seq_len + 1: index + 1, cols]
        with np.errstate(invalid="ignore"):
            volume_base = np.nanmean(np.where(np.isfinite(volume_window), volume_window, np.nan),
                                     axis=0)[:, None]
        x = _normalize_window(window, anchor_close, volume_base, cfg.feature_clip)

        x_future = None
        if with_future:
            future = np.transpose(self._features[index + 1: index + 1 + cfg.future_len, cols],
                                  (1, 0, 2))
            if cfg.future_norm == "anchor":
                future_price_base, future_volume_base = anchor_close, volume_base
            else:
                future_price_base = self._close[index + cfg.future_len, cols][:, None]
                future_volume = self._volume[index + 1: index + 1 + cfg.future_len, cols]
                with np.errstate(invalid="ignore"):
                    future_volume_base = np.nanmean(
                        np.where(np.isfinite(future_volume), future_volume, np.nan), axis=0)[:, None]
            x_future = _normalize_window(future, future_price_base, future_volume_base,
                                         cfg.feature_clip)

        label = self.labels[index, cols]
        label_mask = np.isfinite(label)
        return CrossSection(date=self.dates[index], stocks=self.stocks[cols], x=x,
                            beta=self.beta[cols], label=np.nan_to_num(label, nan=0.0).astype(np.float32),
                            label_mask=label_mask, x_future=x_future,
                            raw_label=self.raw_labels[index, cols])

    def iter_cross_sections(self, indices, with_future=False, need_label=True):
        for i in indices:
            sample = self.cross_section(i, with_future=with_future, need_label=need_label)
            if sample is not None:
                yield sample
