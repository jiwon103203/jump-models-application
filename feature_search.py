"""
Search the feature options for the configuration with the best Sharpe ratio, and test the
choice out of sample so that the search does not simply reward overfitting.

The pipeline exposes a handful of feature options -- the feature set (``--feature-set``),
the scale of the downside deviation (``--log-dd``) and the series taken out of the matrix
(``--remove-series``) -- and the combinations multiply quickly. This module runs them all,
or as many of them as is practical, through the same rolling backtest and ranks them.

Two things keep that ranking honest.

1. **The choice is made on one stretch of history and judged on another.** The backtest
   period is split into a selection (in-sample, IS) part and a hold-out (out-of-sample,
   OOS) part that follows it. Every combination is ranked by its Sharpe ratio over the IS
   part only, and the greedy elimination of feature series also looks at IS Sharpe ratios
   only, so the OOS part takes no part in any decision. The regimes themselves are online
   in both parts -- each re-estimation sees only the days before it -- so the OOS Sharpe of
   the chosen combination is a genuine out-of-sample number for the *choice* as well as for
   the model parameters.

2. **The selection process itself is put to the test.** The OOS part is cut into a few
   consecutive blocks, and at the start of each block the combination with the best Sharpe
   ratio over everything before it is picked and traded through the block (walk-forward
   re-selection). On top of that, the deflated Sharpe ratio of Bailey and López de Prado
   (2014) says how likely the IS winner is to be a real improvement once the number of
   combinations tried is taken into account.

The candidates are generated per (feature set, log-DD) group:

- ``exhaustive``: every subset of the removable series is removed in turn (as long as one
  column is left), which is exact but grows as `2^k`;
- ``greedy``: backward elimination -- remove the series whose removal raises the IS Sharpe
  ratio the most, and repeat until no removal helps;
- ``single``: the full set and every single-series removal;
- ``none``: the full set only, which compares feature sets and scales but removes nothing;
- ``auto`` (default): ``exhaustive`` where it costs at most `max_subsets` runs (the
  "paper" and "example" sets) and ``greedy`` elsewhere (the 35-column "extra" set, whose
  15 series would make `2^15` runs).

Series holding a pinned feature (``--pin-feature``, sparse model only) are never removed,
and series the user removed (``--remove-series``) stay removed in every candidate.
"""

import itertools
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest import performance_metrics, regime_summary, resolve_cost_bps, run_0_1_strategy
from features import (FEATURE_SETS, build_features, feature_series, feature_series_name,
                      feature_set_columns, resolve_pinned_features, resolve_removed_features)
from rolling import refit_schedule, run_rolling_jm

TRADING_DAYS = 252
REMOVAL_MODES = ("auto", "exhaustive", "greedy", "single", "none")
LOG_DD_MODES = ("both", "off", "on")
DEFAULT_OOS_FRAC = .3          # share of the backtest period held out for the OOS test
DEFAULT_WF_FOLDS = 3           # walk-forward re-selection blocks within the OOS part
DEFAULT_MAX_SUBSETS = 64       # largest exhaustive search `auto` accepts for a group
DEFAULT_MAX_RUNS = 300         # hard cap on the rolling fits of one search
MIN_PERIOD_DAYS = 252          # below this, a period's Sharpe ratio is mostly noise
EULER_GAMMA = 0.5772156649015329


############################################
## Candidates
############################################

@dataclass(frozen=True)
class Candidate:
    """
    One feature configuration: a feature set, the scale of its downside deviation, and the
    series removed from it.

    Attributes
    ----------
    feature_set : str
        One of `features.FEATURE_SETS`.

    log_dd : bool
        Whether the downside deviation is log-transformed (``--log-dd``).

    removed : tuple of str
        The series the search removed, in the order of the group's pool.

    fixed : tuple of str
        The removal specifications the user gave (``--remove-series``) that apply to this
        feature set; they are removed in every candidate of the group.
    """
    feature_set: str
    log_dd: bool = False
    removed: tuple = ()
    fixed: tuple = ()

    @property
    def remove_series(self) -> list:
        """Everything removed from the matrix: the user's specifications and the search's."""
        return list(self.fixed) + list(self.removed)

    @property
    def label(self) -> str:
        """A short name, e.g. ``extra log-dd -var -ret-cumlog``."""
        parts = [self.feature_set] + (["log-dd"] if self.log_dd else [])
        return " ".join(parts + [f"-{name}" for name in self.removed])

    @property
    def cli(self) -> str:
        """The command line options that reproduce the configuration in `run_pipeline.py`."""
        parts = [f"--feature-set {self.feature_set}"] + (["--log-dd"] if self.log_dd else [])
        return " ".join(parts + [f"--remove-series {name}" for name in self.remove_series])


@dataclass
class SearchGroup:
    """
    The candidates sharing a feature set and a downside deviation scale, and the state of
    the search over the series that may be removed from them.

    Attributes
    ----------
    feature_set, log_dd, fixed
        See `Candidate`.

    columns : list of str
        The columns of the group's full matrix, the user's removals already applied.

    pool : list of str
        The series the search may remove: every series of `columns` except those holding a
        pinned feature.

    mode : str
        The removal search actually run for the group, one of `REMOVAL_MODES` but "auto".

    current : tuple of str
        Greedy search only: the series removed so far.

    current_score : float
        Greedy search only: the IS Sharpe ratio of `current`.

    done : bool
        Greedy search only: whether the elimination has stopped.

    steps : list of dict
        Greedy search only: the decision taken at each step, for the log.
    """
    feature_set: str
    log_dd: bool
    fixed: tuple
    columns: list
    pool: list
    mode: str
    current: tuple = ()
    current_score: float = np.nan
    done: bool = False
    steps: list = field(default_factory=list)

    def candidate(self, removed=()) -> Candidate:
        """The candidate removing `removed` on top of the user's removals, in pool order."""
        removed = tuple(name for name in self.pool if name in set(removed))
        return Candidate(self.feature_set, self.log_dd, removed, self.fixed)

    def remaining(self, removed) -> list:
        """The columns left once the series of `removed` are taken out."""
        removed = set(removed)
        return [col for col in self.columns if feature_series_name(col) not in removed]

    @property
    def name(self) -> str:
        return self.feature_set + (" log-dd" if self.log_dd else "")


def resolve_jobs(jobs) -> int:
    """Turn ``--search-jobs`` into a number of worker processes: 0 or negative means every core."""
    jobs = int(jobs)
    if jobs <= 0:
        jobs = os.cpu_count() or 1
    return max(1, jobs)


def parse_feature_sets(sets) -> list:
    """Turn ``--search-sets`` (a comma separated string or a list) into feature set names."""
    if sets is None:
        return list(FEATURE_SETS)
    if isinstance(sets, str):
        sets = [part for part in sets.replace(" ", "").split(",") if part]
    out = []
    for name in sets:
        name = str(name).strip().lower()
        if name not in FEATURE_SETS:
            raise ValueError(f"지원하지 않는 피처 세트입니다: '{name}'. 가능한 값: {FEATURE_SETS}")
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("탐색할 피처 세트가 비어 있습니다.")
    return out


def log_dd_values(mode: str) -> list:
    """The ``--log-dd`` settings ``--search-log-dd`` asks for."""
    if mode not in LOG_DD_MODES:
        raise ValueError(f"지원하지 않는 --search-log-dd 값입니다: {mode}. 가능한 값: {LOG_DD_MODES}")
    return {"both": [False, True], "off": [False], "on": [True]}[mode]


def _matching_specs(columns, specs) -> list:
    """The removal specifications that match at least one of `columns`, in order."""
    out = []
    for spec in specs or ():
        try:
            resolve_removed_features(columns, [spec])
        except KeyError:
            continue
        out.append(spec)
    return out


def _pinned_columns(columns, pin_features, feature_set, log_dd, fixed) -> list:
    """
    The pinned columns of a group, resolved one specification at a time.

    A pin that names a feature another set does not build ("DD_10" in the "example" set) is
    not an error during a search -- it simply pins nothing in that group.
    """
    pinned = []
    for spec in pin_features or ():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pinned += resolve_pinned_features(columns, [spec], ver=feature_set,
                                                  log_dd=log_dd, remove_series=fixed)
        except KeyError:
            continue
    return list(dict.fromkeys(pinned))


def build_groups(feature_sets,
                 log_dd_mode: str = "both",
                 custom_columns=(),
                 remove_series=None,
                 pin_features=None,
                 model: str = "jm",
                 removal: str = "auto",
                 max_subsets: int = DEFAULT_MAX_SUBSETS) -> list:
    """
    Lay out the (feature set, log-DD) groups of a search and the series each may remove.

    Parameters
    ----------
    feature_sets : iterable of str
        The feature sets to search, see `parse_feature_sets`.

    log_dd_mode : str, optional (default="both")
        One of `LOG_DD_MODES`. A set the log transform does not change ("example", which is
        always on the log scale) is searched once only.

    custom_columns : iterable of str, optional
        The custom feature columns, which every set carries and the search may remove.

    remove_series : iterable of str, optional
        The user's removal specifications. Those matching a group's columns are removed in
        every candidate of that group; the others are ignored for that group.

    pin_features : iterable of str, optional
        The user's pin specifications. Sparse model only: the series holding a pinned column
        are kept out of the removable pool.

    model : str, optional (default="jm")
        The model of the run; pins only matter for "sjm".

    removal : str, optional (default="auto")
        One of `REMOVAL_MODES`.

    max_subsets : int, optional (default=64)
        Under "auto", the largest number of removal subsets searched exhaustively.

    Returns
    -------
    list of SearchGroup
    """
    if removal not in REMOVAL_MODES:
        raise ValueError(f"지원하지 않는 --search-removal 값입니다: {removal}. 가능한 값: {REMOVAL_MODES}")
    custom_columns = [str(col) for col in custom_columns or ()]
    groups = []
    for feature_set in parse_feature_sets(feature_sets):
        for log_dd in log_dd_values(log_dd_mode):
            if log_dd and feature_set_columns(feature_set, True) == feature_set_columns(feature_set, False):
                if log_dd_mode == "on":      # asked for the log scale only: keep the set once
                    log_dd = False
                else:
                    continue
            all_columns = feature_set_columns(feature_set, log_dd=log_dd) + custom_columns
            fixed = tuple(_matching_specs(all_columns, remove_series))
            columns = [col for col in all_columns
                       if col not in set(resolve_removed_features(all_columns, fixed))]
            pinned = (_pinned_columns(columns, pin_features, feature_set, log_dd, fixed)
                      if model == "sjm" else [])
            kept_series = {feature_series_name(col) for col in pinned}
            pool = [name for name in feature_series(columns) if name not in kept_series]
            mode = removal
            if mode == "auto":
                mode = "exhaustive" if 2 ** len(pool) - 1 <= max_subsets else "greedy"
            groups.append(SearchGroup(feature_set, log_dd, fixed, columns, pool, mode))
    return groups


def grid_candidates(group: SearchGroup) -> list:
    """
    The candidates a group evaluates up front: all of them for "exhaustive", the full set
    and its single removals for "single", and the full set alone for "greedy" and "none"
    (the greedy elimination generates the rest step by step).
    """
    if group.mode == "exhaustive":
        subsets = itertools.chain.from_iterable(
            itertools.combinations(group.pool, size) for size in range(len(group.pool) + 1))
    elif group.mode == "single":
        subsets = [()] + [(name,) for name in group.pool]
    else:
        subsets = [()]
    return [group.candidate(subset) for subset in subsets if group.remaining(subset)]


def greedy_candidates(group: SearchGroup) -> list:
    """The next step of a greedy elimination: every one-series extension of `current`."""
    out = []
    for name in group.pool:
        if name in group.current:
            continue
        removed = group.current + (name,)
        if group.remaining(removed):
            out.append(group.candidate(removed))
    return out


############################################
## Periods and metrics
############################################

def to_date(value):
    """Parse a date given on the command line into the `datetime.date` the index holds."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return pd.Timestamp(str(value).strip()).date()
    except ValueError as exc:
        raise ValueError(f"날짜를 읽지 못했습니다: '{value}' ({exc})") from exc


def split_periods(index, oos_start=None, oos_frac: float = DEFAULT_OOS_FRAC) -> tuple:
    """
    Split the evaluation index into a selection (IS) part and the hold-out (OOS) part that
    follows it.

    Parameters
    ----------
    index : pd.Index
        The sorted evaluation dates.

    oos_start : str or datetime.date, optional
        The first day of the OOS part. When None, the last `oos_frac` of the days are held out.

    oos_frac : float, optional (default=0.3)
        The share of the days held out when `oos_start` is not given.

    Returns
    -------
    (date, date, date)
        The first IS day, the first OOS day and the last day.
    """
    index = pd.Index(index)
    if len(index) < 2:
        raise ValueError("평가 구간이 너무 짧아 IS/OOS로 나눌 수 없습니다.")
    if oos_start is not None:
        start = to_date(oos_start)
        pos = int(index.searchsorted(start, side="left"))
        if pos <= 0 or pos >= len(index):
            raise ValueError(
                f"--search-oos-start {start} 가 평가 구간 {index[0]} ~ {index[-1]} 의 안쪽이 아닙니다. "
                f"IS와 OOS 모두 하루 이상 남도록 지정해 주세요.")
    else:
        if not 0. < oos_frac < 1.:
            raise ValueError(f"--search-oos-frac 는 0과 1 사이여야 합니다. 입력값: {oos_frac}")
        pos = int(round(len(index) * (1. - oos_frac)))
        pos = min(max(pos, 1), len(index) - 1)
    n_is, n_oos = pos, len(index) - pos
    if min(n_is, n_oos) < MIN_PERIOD_DAYS:
        warnings.warn(
            f"IS {n_is}거래일 / OOS {n_oos}거래일로, 한쪽이 1년({MIN_PERIOD_DAYS}거래일)보다 짧아 "
            f"Sharpe 비교가 불안정할 수 있습니다.")
    return index[0], index[pos], index[-1]


def _mask(index, start=None, end=None, closed_end: bool = False) -> np.ndarray:
    """Boolean mask of the dates in `[start, end)` (or `[start, end]` with `closed_end`)."""
    index = pd.Index(index)
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= np.asarray(index >= start)
    if end is not None:
        mask &= np.asarray(index <= end) if closed_end else np.asarray(index < end)
    return mask


def period_metrics(strategy: pd.DataFrame,
                   start=None,
                   end=None,
                   closed_end: bool = False,
                   column: str = "jm",
                   trading_days: int = TRADING_DAYS) -> dict:
    """
    The performance metrics of one column of a `run_0_1_strategy` frame over a period.

    The strategy is computed over its whole history and only then sliced, so the position
    held on the first day of the period is the one the signal had built up to it, and no
    artificial trade is booked at the start of the period.
    """
    part = strategy.loc[_mask(strategy.index, start, end, closed_end)]
    if len(part) < 2:
        return {"Sharpe": np.nan, "Return": np.nan, "MDD": np.nan, "Turnover": np.nan, "n_days": len(part)}
    weight = part["weight"] if column == "jm" else None
    cost = part["cost"] if column == "jm" else None
    metrics = performance_metrics(part[column], part["rf"], weight=weight,
                                  trading_days=trading_days, cost=cost)
    metrics["n_days"] = len(part)
    return metrics


def deflated_sharpe(ret: pd.Series,
                    rf: pd.Series,
                    sharpe: float,
                    trial_sharpes,
                    trading_days: int = TRADING_DAYS) -> dict:
    """
    The deflated Sharpe ratio of Bailey and López de Prado (2014).

    The best of `N` backtests has a higher Sharpe ratio than any one of them would on its
    own, even when none of them has any skill. The deflated Sharpe ratio is the probability
    that the true Sharpe ratio of the winner exceeds the one the best of `N` skill-less
    trials would be expected to reach, given the dispersion of the trials and the skewness
    and fat tails of the winner's returns. Above 0.95 the winner's edge survives the search
    that found it; well below, the search may simply have found the luckiest trial.

    Parameters
    ----------
    ret, rf : pd.Series
        The daily returns of the winner over the selection period, and the risk-free rate.

    sharpe : float
        The winner's annualized Sharpe ratio over that period.

    trial_sharpes : iterable of float
        The annualized Sharpe ratios of every trial, the winner included.

    trading_days : int, optional (default=252)

    Returns
    -------
    dict
        `dsr` (the probability), `sr0` (the annualized Sharpe ratio the best skill-less
        trial is expected to reach) and `n_trials`.
    """
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float) / np.sqrt(trading_days)
    n_trials = len(trials)
    sr = float(sharpe) / np.sqrt(trading_days)
    excess = (ret - rf.reindex(ret.index)).dropna()
    n = len(excess)
    if n < 3 or not np.isfinite(sr):
        return {"dsr": np.nan, "sr0": np.nan, "n_trials": n_trials}
    sr0 = 0.
    if n_trials > 1:
        spread = float(np.std(trials, ddof=1))
        sr0 = spread * ((1. - EULER_GAMMA) * norm.ppf(1. - 1. / n_trials)
                        + EULER_GAMMA * norm.ppf(1. - 1. / (n_trials * math.e)))
    skew = float(excess.skew())
    kurt = float(excess.kurt()) + 3.          # pandas reports the excess kurtosis
    denom = 1. - skew * sr + (kurt - 1.) / 4. * sr ** 2
    if not np.isfinite(denom) or denom <= 0.:
        return {"dsr": np.nan, "sr0": sr0 * np.sqrt(trading_days), "n_trials": n_trials}
    z = (sr - sr0) * np.sqrt(n - 1.) / np.sqrt(denom)
    return {"dsr": float(norm.cdf(z)), "sr0": float(sr0 * np.sqrt(trading_days)), "n_trials": n_trials}


def walk_forward(strategies: dict,
                 oos_index,
                 is_start,
                 folds: int = DEFAULT_WF_FOLDS,
                 n_features: dict = None,
                 buy_bps: float = 10.,
                 sell_bps: float = 10.,
                 trading_days: int = TRADING_DAYS) -> tuple:
    """
    Re-select the best configuration at the start of each OOS block, and trade it through
    the block.

    At the start of block `b` every configuration is ranked by its Sharpe ratio over
    `[is_start, start of b)` -- the IS part and the OOS blocks already traded -- and the
    winner's returns over the block are kept. Concatenated, the blocks give the OOS track
    record of the *selection rule*, which is what one would actually have earned by
    re-running the search periodically. When the winner changes between two blocks, the
    trade from the old position to the new one is charged instead of the one the new
    configuration's own backtest booked on that day.

    Parameters
    ----------
    strategies : dict of {str: pd.DataFrame}
        The `run_0_1_strategy` frames of the configurations, on a common index.

    oos_index : pd.Index
        The OOS dates.

    is_start : date
        The first day of the selection period.

    folds : int, optional (default=3)
        The number of OOS blocks.

    n_features : dict of {str: int}, optional
        Ties are broken in favor of the configuration with fewer features.

    buy_bps, sell_bps : float, optional
        The cost of the two legs of a trade, to charge the switch between configurations.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        One row per block (dates, the configuration picked, its Sharpe ratio over the
        selection window and over the block), and the concatenated daily returns.
    """
    oos_index = pd.Index(oos_index)
    folds = max(1, min(int(folds), len(oos_index)))
    blocks = np.array_split(np.arange(len(oos_index)), folds)
    n_features = n_features or {}
    rows, pieces = [], []
    previous, previous_weight = None, None
    for b, positions in enumerate(blocks):
        start, end = oos_index[positions[0]], oos_index[positions[-1]]
        scores = {label: period_metrics(frame, is_start, start, trading_days=trading_days)["Sharpe"]
                  for label, frame in strategies.items()}
        ranked = sorted((label for label in scores if np.isfinite(scores[label])),
                        key=lambda label: (-scores[label], n_features.get(label, 0)))
        if not ranked:
            raise ValueError("워크포워드 선택 구간에서 Sharpe를 계산할 수 있는 조합이 없습니다.")
        chosen = ranked[0]
        frame = strategies[chosen]
        part = frame.loc[_mask(frame.index, start, end, closed_end=True)]
        ret = part["jm"].copy()
        if previous is not None and chosen != previous:
            # the new configuration's own backtest traded from its own previous weight; the
            # walk-forward portfolio trades from the weight the old one left it with
            move = float(part["weight"].iloc[0] - previous_weight)
            actual = max(move, 0.) * buy_bps / 1e4 + max(-move, 0.) * sell_bps / 1e4
            ret.iloc[0] += float(part["cost"].iloc[0]) - actual
        pieces.append(ret)
        rows.append({"block": b + 1, "start": start, "end": end, "n_days": len(part),
                     "selected": chosen, "select_sharpe": scores[chosen],
                     "block_sharpe": period_metrics(frame, start, end, closed_end=True,
                                                    trading_days=trading_days)["Sharpe"]})
        previous, previous_weight = chosen, float(part["weight"].iloc[-1])
    return pd.DataFrame(rows).set_index("block"), pd.concat(pieces)


############################################
## The search
############################################

def _fit_regime(X: pd.DataFrame, ret: pd.Series, fit_kwargs: dict) -> tuple:
    """
    Run one rolling fit and return its online regimes -- the unit of work of a search, kept
    at module level so that it can be sent to a worker process.

    Warnings are collected rather than printed: a search runs the same fit dozens of times,
    and the same "training window shortened" warning dozens of times is noise.
    """
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = run_rolling_jm(X, ret, verbose=False, **fit_kwargs)
            regime, error = result.regimes["regime"], None
        except Exception as exc:          # one failing configuration must not end the search
            regime, error = None, f"{type(exc).__name__}: {exc}"
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    return regime, error, messages, time.perf_counter() - started


@dataclass
class SearchRecord:
    """The outcome of one distinct feature matrix: the candidate that built it and its backtest."""
    candidate: Candidate
    columns: tuple
    stage: str
    aliases: list = field(default_factory=list)
    regime: pd.Series = None
    strategy: pd.DataFrame = None
    error: str = None
    seconds: float = np.nan


class FeatureSearch:
    """
    Run candidates through the rolling backtest, once per distinct feature matrix.

    Two candidates that end up with the same columns -- the "paper" set without its downside
    deviation is the same two Sortino ratios whatever the scale of the deviation it lost --
    share one fit: the second is recorded as an alias of the first.
    """

    def __init__(self, signal_ret, bt_ret, bt_rf, extra_df=None, warmup: int = 252,
                 fit_kwargs: dict = None, strategy_kwargs: dict = None, pin_features=None,
                 jobs: int = 1,
                 max_runs: int = DEFAULT_MAX_RUNS, trading_days: int = TRADING_DAYS,
                 verbose: bool = True):
        self.signal_ret, self.bt_ret, self.bt_rf = signal_ret, bt_ret, bt_rf
        self.extra_df, self.warmup = extra_df, warmup
        self.fit_kwargs = dict(fit_kwargs or {})
        self.strategy_kwargs = dict(strategy_kwargs or {})
        self.pin_features = list(pin_features or [])
        self.jobs, self.max_runs = resolve_jobs(jobs), int(max_runs)
        self.trading_days, self.verbose = trading_days, verbose
        self.records = {}           # columns -> SearchRecord
        self.by_label = {}          # candidate label -> SearchRecord
        self.messages = {}          # warning text -> number of fits that raised it
        self.n_runs = 0
        self.capped = False
        self.is_period = (None, None)
        self._executor = None

    def __enter__(self):
        if self.jobs > 1:
            self._executor = ProcessPoolExecutor(max_workers=self.jobs)
        return self

    def __exit__(self, *exc):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def features(self, candidate: Candidate) -> pd.DataFrame:
        """The feature matrix of a candidate."""
        return build_features(self.signal_ret, ver=candidate.feature_set, warmup=self.warmup,
                              log_dd=candidate.log_dd, extra_features=self.extra_df,
                              remove_series=candidate.remove_series or None)

    def first_refit(self, candidate: Candidate):
        """The first day of the candidate's online regimes, found without fitting anything."""
        X = self.features(candidate)
        schedule = refit_schedule(X.index, window=self.fit_kwargs.get("window", 3000),
                                  min_window=self.fit_kwargs.get("min_window", 500),
                                  start_date=self.fit_kwargs.get("start_date"))
        return schedule[0][0] if schedule else None

    def job_kwargs(self, candidate: Candidate, X: pd.DataFrame) -> dict:
        """
        The fit options of one candidate: the shared ones, plus the columns its pins resolve to.
        Pins only steer the sparse model, and what a group name such as "paper" stands for
        depends on the candidate's feature set and removals, so they are resolved here.
        """
        kwargs = dict(self.fit_kwargs)
        if self.pin_features and kwargs.get("model", "jm") == "sjm":
            kwargs["pin_feats"] = _pinned_columns(X.columns, self.pin_features, candidate.feature_set,
                                                  candidate.log_dd, candidate.remove_series)
        return kwargs

    def score(self, record: SearchRecord) -> float:
        """The selection criterion: the Sharpe ratio over the IS period."""
        if record.strategy is None:
            return np.nan
        start, end = self.is_period
        return period_metrics(record.strategy, start, end, trading_days=self.trading_days)["Sharpe"]

    def evaluate(self, candidates, stage: str) -> list:
        """
        Backtest the candidates not seen yet and return the record of every one of them.

        Candidates past `max_runs` are dropped with a warning, and the search is marked as
        capped so that the greedy loop stops.
        """
        out, todo, seen = [], [], {}
        for candidate in candidates:
            if candidate.label in self.by_label:
                out.append(self.by_label[candidate.label])
                continue
            X = self.features(candidate)
            columns = tuple(X.columns)
            record = self.records.get(columns) or seen.get(columns)
            if record is None:
                if self.n_runs + len(todo) >= self.max_runs:
                    self.capped = True
                    continue
                record = SearchRecord(candidate, columns, stage)
                seen[columns] = record
                todo.append((record, X))
            elif candidate.label != record.candidate.label and candidate.label not in record.aliases:
                record.aliases.append(candidate.label)
            self.by_label[candidate.label] = record
            out.append(record)
        if self.capped and self.verbose:
            print(f"  ! 실행 한도(--search-max-runs {self.max_runs})에 도달해 나머지 후보를 건너뜁니다.")

        jobs = [(X, self.signal_ret.reindex(X.index), self.job_kwargs(record.candidate, X))
                for record, X in todo]
        if self._executor is not None and len(jobs) > 1:
            results = self._executor.map(_fit_regime, *zip(*jobs))
        else:
            results = (_fit_regime(*job) for job in jobs)
        for (record, _), (regime, error, messages, seconds) in zip(todo, results):
            self.n_runs += 1
            record.regime, record.error, record.seconds = regime, error, seconds
            for message in messages:
                self.messages[message] = self.messages.get(message, 0) + 1
            if regime is not None:
                record.strategy = run_0_1_strategy(regime, self.bt_ret, self.bt_rf,
                                                   bull_state=0, **self.strategy_kwargs)
            self.records[record.columns] = record
            if self.verbose:
                status = (f"IS Sharpe {self.score(record):6.3f}" if error is None
                          else f"실패 — {error}")
                print(f"  [{self.n_runs:>3}] {record.candidate.label:<40} 피처 {len(record.columns):>2}개  "
                      f"{status}  ({seconds:.0f}초)")
        return out


def run_search(signal_ret: pd.Series,
               bt_ret: pd.Series,
               bt_rf: pd.Series,
               extra_df: pd.DataFrame = None,
               baseline: Candidate = None,
               feature_sets=None,
               log_dd_mode: str = "both",
               removal: str = "auto",
               remove_series=None,
               pin_features=None,
               max_subsets: int = DEFAULT_MAX_SUBSETS,
               min_improve: float = 0.,
               oos_start=None,
               oos_frac: float = DEFAULT_OOS_FRAC,
               wf_folds: int = DEFAULT_WF_FOLDS,
               jobs: int = 1,
               max_runs: int = DEFAULT_MAX_RUNS,
               warmup: int = 252,
               fit_kwargs: dict = None,
               strategy_kwargs: dict = None,
               trading_days: int = TRADING_DAYS,
               verbose: bool = True) -> dict:
    """
    Search the feature options for the best Sharpe ratio, and test the winner out of sample.

    Parameters
    ----------
    signal_ret : pd.Series
        The return series the features are built on and the states are sorted by
        (`data.signal_ret` of `run_pipeline.prepare_inputs`).

    bt_ret, bt_rf : pd.Series
        The return and risk-free series the strategy trades, see
        `backtest.resolve_backtest_basis`.

    extra_df : pd.DataFrame, optional
        The custom feature columns, carried by every feature set.

    baseline : Candidate, optional
        The configuration the run was given on the command line, which the winner is judged
        against. Defaults to the "paper" set of the article.

    feature_sets, log_dd_mode, removal, max_subsets
        The candidates to search, see `build_groups`.

    remove_series, pin_features
        The user's ``--remove-series`` and ``--pin-feature``, see `build_groups`.

    min_improve : float, optional (default=0.)
        Greedy search only: the smallest rise of the IS Sharpe ratio a removal must bring to
        be accepted. A positive value stops the elimination from chasing noise.

    oos_start, oos_frac
        Where the OOS part begins, see `split_periods`.

    wf_folds : int, optional (default=3)
        The number of walk-forward re-selection blocks within the OOS part.

    jobs : int, optional (default=1)
        Worker processes; 0 or negative uses every core.

    max_runs : int, optional (default=300)
        The largest number of rolling fits the search may run.

    warmup : int, optional (default=252)
        Passed to `features.build_features`.

    fit_kwargs : dict, optional
        Passed to `rolling.run_rolling_jm` (model, jump penalty, window, ...).

    strategy_kwargs : dict, optional
        Passed to `backtest.run_0_1_strategy` (delay, costs, cash limits).

    trading_days : int, optional (default=252)

    verbose : bool, optional (default=True)

    Returns
    -------
    dict
        `results` (one row per distinct configuration, ranked by IS Sharpe ratio),
        `walkforward` (the block table), `returns` (the daily returns of buy-and-hold, the
        baseline, the IS winner, the recommendation and the walk-forward rule), `summary`
        (the headline numbers), `best`, `baseline` and `recommended` (their `Candidate`s),
        `periods` and `groups`.
    """
    fit_kwargs = dict(fit_kwargs or {})
    strategy_kwargs = dict(strategy_kwargs or {})
    baseline = baseline or Candidate("paper")
    model = fit_kwargs.get("model", "jm")
    custom_columns = list(extra_df.columns) if extra_df is not None else []
    groups = build_groups(feature_sets, log_dd_mode=log_dd_mode, custom_columns=custom_columns,
                          remove_series=remove_series, pin_features=pin_features, model=model,
                          removal=removal, max_subsets=max_subsets)
    # a removal applies to whichever sets hold what it names ("var" exists in "extra" only),
    # the baseline included; one that matches no set at all is a typo, as in a plain run
    baseline_columns = feature_set_columns(baseline.feature_set, log_dd=baseline.log_dd) + custom_columns
    baseline = Candidate(baseline.feature_set, baseline.log_dd, (),
                         tuple(_matching_specs(baseline_columns, baseline.fixed)))
    unmatched = [spec for spec in remove_series or ()
                 if spec not in baseline.fixed and not any(spec in group.fixed for group in groups)]
    if unmatched:
        raise KeyError(f"제거할 피처 {unmatched}가 탐색하는 어느 피처 세트에도 없습니다. "
                       f"시리즈 이름(예: 'var'), 개별 피처 이름(예: 'var_20'), 또는 커스텀 변수 지정"
                       f"(예: 'VIX:ewm:20') 중 하나여야 합니다.")
    buy_bps, sell_bps = resolve_cost_bps(strategy_kwargs.get("cost_bps", 10.),
                                         strategy_kwargs.get("buy_cost_bps"),
                                         strategy_kwargs.get("sell_cost_bps"))

    with FeatureSearch(signal_ret, bt_ret, bt_rf, extra_df=extra_df, warmup=warmup,
                       fit_kwargs=fit_kwargs, strategy_kwargs=strategy_kwargs,
                       pin_features=pin_features if model == "sjm" else None, jobs=jobs,
                       max_runs=max_runs, trading_days=trading_days, verbose=verbose) as search:
        # every candidate's online regimes start on or before its group's full set -- removing
        # a series only ever lengthens the feature matrix -- so the latest first refit of the
        # full sets is a start every candidate covers, and the periods can be fixed up front
        starts = [search.first_refit(group.candidate()) for group in groups]
        starts.append(search.first_refit(baseline))
        if any(start is None for start in starts):
            raise ValueError("재추정 시점이 없는 피처 세트가 있습니다. --window/--min-window/--warmup 을 확인해 주세요.")
        eval_start = max(starts)
        eval_index = signal_ret.index[_mask(signal_ret.index, eval_start)]
        eval_index = eval_index.intersection(bt_ret.dropna().index)
        is_start, oos_start_date, end = split_periods(eval_index, oos_start, oos_frac)
        search.is_period = (is_start, oos_start_date)

        if verbose:
            n_grid = sum(len(grid_candidates(group)) for group in groups)
            print(f"\n변수 옵션 탐색: 그룹 {len(groups)}개, 1단계 후보 {n_grid}개 "
                  f"(병렬 {search.jobs}개, 실행 한도 {max_runs}회)")
            for group in groups:
                print(f"  · {group.name:<14} 피처 {len(group.columns):>2}개, 제거 후보 시리즈 "
                      f"{len(group.pool)}개 → {group.mode}")
            print(f"선택(IS) 구간 {is_start} ~ {oos_start_date} 전일, "
                  f"검증(OOS) 구간 {oos_start_date} ~ {end} — 순위는 IS Sharpe로만 매깁니다.")

        # stage 1: the baseline, every full set, and the exhaustive/single grids
        first = [baseline] + [candidate for group in groups for candidate in grid_candidates(group)]
        search.evaluate(first, stage="grid")

        # stage 2+: greedy backward elimination, all greedy groups advancing in lockstep so
        # that one batch of fits keeps every worker busy
        greedy = [group for group in groups if group.mode == "greedy"]
        for group in greedy:
            group.current_score = search.score(search.by_label[group.candidate().label])
        step = 0
        while not search.capped:
            active = [group for group in greedy if not group.done and np.isfinite(group.current_score)]
            batches = {group.name: greedy_candidates(group) for group in active}
            for group in active:
                if not batches[group.name]:
                    group.done = True
            active = [group for group in active if not group.done]
            if not active:
                break
            step += 1
            if verbose:
                print(f"탐욕 제거 {step}단계: " + ", ".join(
                    f"{group.name}(현재 -{len(group.current)}개, IS {group.current_score:.3f})"
                    for group in active))
            search.evaluate([c for group in active for c in batches[group.name]],
                            stage=f"greedy-{step}")
            for group in active:
                scored = [(search.score(search.by_label[c.label]), c) for c in batches[group.name]
                          if c.label in search.by_label]
                scored = [(s, c) for s, c in scored if np.isfinite(s)]
                if not scored:
                    group.done = True
                    continue
                best_score, best = max(scored, key=lambda pair: pair[0])
                if best_score > group.current_score + min_improve:
                    added = [name for name in best.removed if name not in group.current]
                    group.steps.append({"step": step, "removed": added[0] if added else "",
                                        "from": group.current_score, "to": best_score})
                    if verbose:
                        print(f"  → {group.name}: -{added[0] if added else '?'} 채택 "
                              f"(IS Sharpe {group.current_score:.3f} → {best_score:.3f})")
                    group.current, group.current_score = best.removed, best_score
                else:
                    group.done = True
                    if verbose:
                        print(f"  → {group.name}: 더 제거해도 IS Sharpe가 오르지 않아 종료 "
                              f"(최고 후보 {best_score:.3f} ≤ 현재 {group.current_score:.3f}"
                              f"{f' + {min_improve:g}' if min_improve else ''})")

    if verbose and search.messages:
        print("탐색 중 발생한 경고 (같은 문구는 한 번만):")
        for message, count in search.messages.items():
            print(f"  · ({count}회) {message}")

    return summarize_search(search, groups, baseline, is_start, oos_start_date, end,
                            wf_folds=wf_folds, buy_bps=buy_bps, sell_bps=sell_bps,
                            trading_days=trading_days)


def summarize_search(search: FeatureSearch,
                     groups: list,
                     baseline: Candidate,
                     is_start,
                     oos_start,
                     end,
                     wf_folds: int = DEFAULT_WF_FOLDS,
                     buy_bps: float = 10.,
                     sell_bps: float = 10.,
                     trading_days: int = TRADING_DAYS) -> dict:
    """
    Rank the configurations a search evaluated, test the winner out of sample, and decide
    what to recommend. See `run_search` for the returned dict.

    Every configuration is measured on the dates all of them share, so that no Sharpe ratio
    profits from a calmer or a longer stretch of history than another.
    """
    done = [record for record in search.records.values() if record.strategy is not None]
    failed = [record for record in search.records.values() if record.strategy is None]
    if not done:
        raise ValueError("성공한 조합이 하나도 없습니다: " + "; ".join(
            f"{record.candidate.label}: {record.error}" for record in failed))
    base_record = search.by_label.get(baseline.label)
    if base_record is None or base_record.strategy is None:
        raise ValueError(f"기준 설정({baseline.label})의 백테스트가 실패했습니다: "
                         f"{getattr(base_record, 'error', '평가되지 않음')}")

    common = done[0].strategy.index
    for record in done[1:]:
        common = common.intersection(record.strategy.index)
    common = common[_mask(common, is_start, end, closed_end=True)]
    if len(common) < 2:
        raise ValueError("조합들이 공통으로 덮는 평가 구간이 없습니다.")
    strategies = {record.candidate.label: record.strategy.reindex(common) for record in done}
    n_features = {record.candidate.label: len(record.columns) for record in done}

    rows = []
    for record in done:
        frame = strategies[record.candidate.label]
        metrics = {
            "is": period_metrics(frame, is_start, oos_start, trading_days=trading_days),
            "oos": period_metrics(frame, oos_start, end, closed_end=True, trading_days=trading_days),
            "full": period_metrics(frame, is_start, end, closed_end=True, trading_days=trading_days),
        }
        regimes = regime_summary(record.regime.reindex(common))
        candidate = record.candidate
        row = {"label": candidate.label, "feature_set": candidate.feature_set,
               "log_dd": candidate.log_dd, "removed": " ".join(candidate.removed),
               "n_features": len(record.columns), "stage": record.stage,
               "is_baseline": record is base_record}
        for period, values in metrics.items():
            row[f"sharpe_{period}"] = values["Sharpe"]
        for period in ("is", "oos"):
            row[f"return_{period}"] = metrics[period]["Return"]
            row[f"mdd_{period}"] = metrics[period]["MDD"]
        row.update({"turnover": metrics["full"]["Turnover"], "bear_share": regimes["bear_share"],
                    "shifts_per_year": regimes["shifts_per_year"], "seconds": record.seconds,
                    "aliases": "; ".join(record.aliases), "features": " ".join(record.columns),
                    "cli": candidate.cli, "error": ""})
        rows.append(row)
    for record in failed:
        rows.append({"label": record.candidate.label, "feature_set": record.candidate.feature_set,
                     "log_dd": record.candidate.log_dd, "removed": " ".join(record.candidate.removed),
                     "n_features": len(record.columns), "stage": record.stage,
                     "is_baseline": record is base_record, "features": " ".join(record.columns),
                     "cli": record.candidate.cli, "error": record.error})
    results = pd.DataFrame(rows)
    # the IS Sharpe ratio decides; among equals, the configuration with fewer features wins
    results = results.sort_values(["sharpe_is", "n_features"], ascending=[False, True],
                                  na_position="last").reset_index(drop=True)
    results.insert(0, "rank_is", np.arange(1, len(results) + 1))
    results = results.set_index("rank_is")

    best_row = results.iloc[0]
    base_row = results.loc[results["is_baseline"]].iloc[0]
    best_record = search.by_label[best_row["label"]]
    best_frame, base_frame = strategies[best_row["label"]], strategies[base_row["label"]]

    bh = {period: period_metrics(base_frame, *bounds, column="bh", trading_days=trading_days)
          for period, bounds in (("is", (is_start, oos_start)),
                                 ("oos", (oos_start, end, True)),
                                 ("full", (is_start, end, True)))}
    is_part = best_frame.loc[_mask(common, is_start, oos_start)]
    dsr = deflated_sharpe(is_part["jm"], is_part["rf"], best_row["sharpe_is"],
                          results["sharpe_is"].dropna(), trading_days=trading_days)
    finite = results[["sharpe_is", "sharpe_oos"]].dropna()
    rank_corr = (float(finite["sharpe_is"].corr(finite["sharpe_oos"], method="spearman"))
                 if len(finite) >= 3 else np.nan)

    oos_index = common[_mask(common, oos_start, end, closed_end=True)]
    wf_table, wf_returns = walk_forward(strategies, oos_index, is_start, folds=wf_folds,
                                        n_features=n_features, buy_bps=buy_bps,
                                        sell_bps=sell_bps, trading_days=trading_days)
    wf_table["baseline_sharpe"] = [
        period_metrics(base_frame, row.start, row.end, closed_end=True, trading_days=trading_days)["Sharpe"]
        for row in wf_table.itertuples()]
    wf_table["bh_sharpe"] = [
        period_metrics(base_frame, row.start, row.end, closed_end=True, column="bh",
                       trading_days=trading_days)["Sharpe"]
        for row in wf_table.itertuples()]
    oos_rf = base_frame.loc[oos_index, "rf"]
    wf_metrics = performance_metrics(wf_returns, oos_rf, trading_days=trading_days)

    # the IS winner is recommended only if it keeps its edge over the baseline out of sample;
    # otherwise the search found nothing that generalizes, and the baseline stays
    is_baseline = bool(best_row["is_baseline"])
    passed = is_baseline or bool(best_row["sharpe_oos"] >= base_row["sharpe_oos"])
    recommended_row = best_row if passed else base_row
    recommended = search.by_label[recommended_row["label"]].candidate

    returns = pd.DataFrame({
        "B&H": base_frame["bh"],
        f"baseline: {base_row['label']}": base_frame["jm"],
        f"best IS: {best_row['label']}": best_frame["jm"],
        "walk-forward": wf_returns.reindex(common),
        "rf": base_frame["rf"],
    })
    returns.index.name = "date"
    returns["period"] = np.where(_mask(common, oos_start), "OOS", "IS")

    summary = pd.Series({
        "is_start": is_start, "oos_start": oos_start, "end": end,
        "n_is": int(_mask(common, is_start, oos_start).sum()),
        "n_oos": len(oos_index),
        "n_configs": len(done), "n_failed": len(failed), "n_runs": search.n_runs,
        "capped": search.capped,
        "bh_sharpe_is": bh["is"]["Sharpe"], "bh_sharpe_oos": bh["oos"]["Sharpe"],
        "bh_sharpe_full": bh["full"]["Sharpe"],
        "baseline": base_row["label"], "baseline_sharpe_is": base_row["sharpe_is"],
        "baseline_sharpe_oos": base_row["sharpe_oos"],
        "best": best_row["label"], "best_sharpe_is": best_row["sharpe_is"],
        "best_sharpe_oos": best_row["sharpe_oos"], "best_n_features": int(best_row["n_features"]),
        "best_cli": best_row["cli"],
        "dsr": dsr["dsr"], "dsr_sr0": dsr["sr0"], "dsr_n_trials": dsr["n_trials"],
        "rank_corr_is_oos": rank_corr,
        "wf_folds": len(wf_table), "wf_sharpe": wf_metrics["Sharpe"],
        "oos_passed": passed,
        "recommended": recommended.label, "recommended_cli": recommended.cli,
    }, name="search")

    return {"results": results, "walkforward": wf_table, "returns": returns, "summary": summary,
            "best": best_record.candidate, "baseline": base_record.candidate,
            "recommended": recommended, "periods": (is_start, oos_start, end), "groups": groups}
