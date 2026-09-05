"""
유사 국면 탐색 -- describing every bear episode the rolling model called, finding the past
episode the current one most resembles, and reading a projected end date off the two.

The rolling fit produces a regime for every day; a *episode* is one uninterrupted run of the
bear state. Once each of them is reduced to a handful of numbers -- how deep the fall was,
how volatile, how far prices had already fallen when the model turned defensive, how much of
the episode the jump penalty was holding in place -- the current episode can be compared
against every past one on the same footing, and the closest match becomes a concrete answer
to "국면이 언제 끝나는가": the current episode is put next to the path the matched one took.

Four things are computed here, in this order.

1. `extract_episodes` cuts the regime series into episodes and measures each one: length,
   the benchmark return over it, its drawdown, and whether it was a false signal, i.e. a
   defensive call that preceded no fall at all.
2. `episode_metrics` describes each episode with the variables the comparison runs on --
   volatility, Sortino, the drawdown already realized at entry, what the first weeks after
   entry looked like, and the split of the episode into the days the *features* put in the
   bear state and the days the *jump penalty* kept there (`rolling.state_losses`).
3. `rank_similar_episodes` standardizes those variables across episodes and ranks the past
   ones by their distance to a target episode, by default the most recent one.
4. `compare_episode_paths` puts the normalized price path of the target next to that of the
   match, and searches for the lag that best aligns the two: two episodes can trace the same
   shape at different speeds, and the lag is that difference in days.

`length_scenarios` then turns the episode lengths into projected end dates -- the median of
the past episodes, and the length of each close match.

Two cautions on reading the output. The metrics of `episode_metrics` describe an episode
*after the fact* -- the forward-window columns look at the days following the entry, so the
table is a description of history and not a signal that could have been traded. And a false
signal is flagged, never dropped: whether an episode that preceded no fall belongs in the
comparison set is a judgement about the market, not about the code, so `select_episodes`
takes the filters as arguments and applies none of them by default.
"""

import numpy as np
import pandas as pd

from features import compute_ewm_sortino

TRADING_DAYS = 252
DEFAULT_HORIZON = 22           # trading days after entry the forward-window columns cover (~1 month)
DEFAULT_PATH_HORIZON = 60      # trading days of the normalized path comparison
DEFAULT_MAX_LAG = 30           # widest lag the path alignment searches over
MIN_ALIGN_OVERLAP = 10         # fewest overlapping days a candidate lag must leave
ALIGN_CRITERIA = ("rmse", "corr")

# the variables the similarity search runs on by default: what the model saw (the loss gap
# and the penalty share), what the market was doing (volatility, Sortino, the volatility
# relative to the state the model had fitted), where the fall stood at entry, and what the
# first weeks after entry looked like
DEFAULT_SIMILARITY_METRICS = ("loss_diff_mean", "vol_5_mean", "vol_20_mean", "sortino_20_mean",
                              "vol_ratio_state", "state_flip", "dd_at_entry", "ret_fwd",
                              "mdd_fwd", "penalty_share")


def _mdd(returns) -> float:
    """
    The maximum drawdown of a return series, measured from the level in force before it.

    The curve starts at 1 on the day *before* the first return, so that a single losing day
    shows that day's loss rather than the zero a one-point curve would give.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return np.nan
    curve = np.concatenate([[1.], np.cumprod(1. + returns)])
    return float((curve / np.maximum.accumulate(curve) - 1.).min())


def _cumret(returns) -> float:
    """The compounded return of a series, NaN when it is empty."""
    returns = np.asarray(returns, dtype=float)
    return float(np.prod(1. + returns) - 1.) if returns.size else np.nan


def extract_episodes(regime,
                     ret,
                     state: int = 1,
                     false_signal_return: float = 0.) -> pd.DataFrame:
    """
    Cut the regime series into episodes of one state and measure each of them.

    An episode is a maximal run of consecutive days in `state`. It is measured on the
    benchmark return of exactly those days -- with no trading delay, since the point is to
    describe what the market did while the model was calling that regime, not what a
    strategy would have earned from the call (`backtest.run_0_1_strategy` does that).

    Parameters
    ----------
    regime : pd.Series
        The `regime` column of `rolling.RollingJMResult.regimes`, indexed by date.

    ret : pd.Series
        The benchmark return of every day, indexed by date. Must cover `regime`.

    state : int, optional (default=1)
        The state to cut into episodes, normally the bear state (`n_components - 1`).

    false_signal_return : float, optional (default=0.)
        An episode whose return is above this threshold is flagged as a false signal: the
        model turned defensive and the market rose anyway. It is only ever flagged, never
        removed -- see `select_episodes`.

    Returns
    -------
    pd.DataFrame
        One row per episode, indexed by its start date, with the end date, the length in
        trading days, the compounded benchmark return over it (`period_ret`), its maximum
        drawdown (`mdd`), the `false_signal` flag, and `ongoing`, true for an episode that
        is still running at the end of the sample.
    """
    regime = pd.Series(regime).dropna().astype(int)
    ret = pd.Series(ret, dtype=float).reindex(regime.index)
    if ret.isna().any():
        raise ValueError("수익률 시리즈가 레짐 시리즈의 모든 날짜를 포함하지 않습니다.")

    in_state = (regime == state).to_numpy()
    rows = []
    start = None
    for i, flag in enumerate(in_state):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(in_state) - 1):
            stop = i if not flag else i + 1          # exclusive
            segment = ret.iloc[start:stop]
            period_ret = _cumret(segment)
            rows.append({
                "start": regime.index[start],
                "end": regime.index[stop - 1],
                "length": stop - start,
                "period_ret": period_ret,
                "mdd": _mdd(segment),
                "false_signal": bool(period_ret > false_signal_return),
                "ongoing": stop == len(in_state) and in_state[-1],
            })
            start = None

    episodes = pd.DataFrame(rows, columns=["start", "end", "length", "period_ret", "mdd",
                                           "false_signal", "ongoing"])
    return episodes.set_index("start")


def select_episodes(episodes: pd.DataFrame,
                    min_length: int = 0,
                    drop_false_signal: bool = False,
                    drop_ongoing: bool = True,
                    before=None,
                    exclude=None) -> pd.DataFrame:
    """
    Apply the optional filters to an episode table.

    None of the three narrowing filters is on by default. Whether an episode that preceded
    no fall, or one that lasted three days, belongs in a comparison set is a judgement about
    the market rather than a property of the data, so the caller makes it: `extract_episodes`
    flags a false signal and records every length, and this function is where a choice about
    them is expressed.

    Parameters
    ----------
    episodes : pd.DataFrame
        The output of `extract_episodes`.

    min_length : int, optional (default=0)
        Drop episodes shorter than this many trading days. A forward window of `h` days is
        only fully observed inside an episode of at least `h` days, which is the usual
        reason to set it.

    drop_false_signal : bool, optional (default=False)
        Drop the episodes flagged as false signals.

    drop_ongoing : bool, optional (default=True)
        Drop the episode still running at the end of the sample. It is on by default because
        an unfinished episode has no length to contribute to a comparison of lengths.

    before : optional
        Keep only the episodes that had already ended by this date. This is what makes a
        comparison against a target episode a comparison against its *past*: an episode that
        had not happened yet cannot be evidence about how long the target will run.

    exclude : iterable, optional
        Start dates to drop outright, e.g. the target of a similarity search.

    Returns
    -------
    pd.DataFrame
        The rows of `episodes` that survive, in the same order.
    """
    keep = pd.Series(True, index=episodes.index)
    if min_length > 0:
        keep &= episodes["length"] >= min_length
    if drop_false_signal:
        keep &= ~episodes["false_signal"].astype(bool)
    if drop_ongoing:
        keep &= ~episodes["ongoing"].astype(bool)
    if before is not None:
        keep &= episodes["end"] < before
    if exclude is not None:
        keep &= ~episodes.index.isin(list(exclude))
    return episodes[keep]


def resolve_target_episode(episodes: pd.DataFrame, target=None):
    """
    Resolve a target episode from a date, which need not be the day the episode started.

    An episode is named by its start date, but the date one has in mind for "the current
    regime" is usually a day somewhere inside it, so a date that falls between an episode's
    start and end resolves to that episode.

    Parameters
    ----------
    episodes : pd.DataFrame
        The output of `extract_episodes`.

    target : optional
        A start date, or any date inside an episode. None gives the last episode.

    Returns
    -------
    The start date of the resolved episode.
    """
    if target is None:
        return episodes.index[-1]
    if target in episodes.index:
        return target
    covering = [start for start, row in episodes.iterrows() if start <= target <= row["end"]]
    if covering:
        return covering[-1]
    raise KeyError(f"기준 국면 '{target}'을 국면 표에서 찾을 수 없습니다. 국면의 시작일이거나 "
                   f"국면 안의 날짜여야 합니다. 가능한 시작일: {[str(d) for d in episodes.index]}")


def _state_vol_table(params: pd.DataFrame) -> pd.DataFrame:
    """The annualized in-sample volatility of every state, indexed by refit date."""
    if params is None or params.empty:
        return None
    return params.pivot(index="refit_date", columns="state", values="vol_ann")


def episode_metrics(episodes: pd.DataFrame,
                    regimes: pd.DataFrame,
                    data: pd.DataFrame,
                    params: pd.DataFrame = None,
                    state: int = 1,
                    horizon: int = DEFAULT_HORIZON,
                    trading_days: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Describe every episode with the variables the similarity search compares them on.

    The columns fall into four groups.

    *What the model saw.* ``loss_diff_mean`` is the average of ``loss(bear) − loss(bull)``
    over the episode (`rolling.state_losses`): the more negative, the more decisively the
    features sat in the bear cluster. ``penalty_days`` counts the days on which they did not
    -- the days the bull centroid was actually closer and only the jump penalty kept the
    model defensive -- and ``distance_days`` the rest, which is the split of the episode's
    length into its two causes. ``penalty_share`` is the former as a fraction.

    *What the market was doing.* The realized volatility over 5 and 20 days and the EWM
    Sortino ratio over a halflife of 20, averaged over the episode; and ``vol_ratio_state``,
    the 20-day volatility relative to the volatility the fitted bear state had in its own
    training window, which says whether this episode was calm or violent *by the standard of
    the model that called it*. ``state_flip`` is 1 when the bull state of that fit was the
    lower-volatility one -- normally it is not, since the model separates the regimes on
    volatility, so a 1 marks a re-estimation whose states were arranged unusually.

    *Where the fall stood at entry.* ``dd_at_entry`` is the drawdown already realized when
    the model turned defensive, the measure of how pre-emptive the call was.

    *What followed.* ``ret_fwd`` and ``mdd_fwd`` are the compounded return and the maximum
    drawdown over the `horizon` trading days from the entry, and ``fwd_days`` how many of
    those days the sample actually holds -- short of `horizon` only for an episode near the
    end of the data. These look past the entry day on purpose: they describe the episode
    after the fact and are not available at entry.

    A run made before `rolling.state_losses` existed has no ``loss_*`` columns in its
    `regimes`; the three loss-derived columns then come out as NaN and the similarity search
    simply runs on the remaining variables.

    Parameters
    ----------
    episodes : pd.DataFrame
        The output of `extract_episodes`.

    regimes : pd.DataFrame
        The regimes table of the run: `regime`, the `loss_*` columns when present, and
        `refit_date`.

    data : pd.DataFrame
        The market data, with `close` and `ret` and optionally `excess_ret`, which the
        Sortino ratio prefers when it is there, as the model's own features do.

    params : pd.DataFrame, optional
        `rolling.RollingJMResult.params`, needed for `vol_ratio_state` and `state_flip`.

    state : int, optional (default=1)
        The state the episodes are in.

    horizon : int, optional (default=`DEFAULT_HORIZON`)
        The length of the forward window, in trading days.

    trading_days : int, optional (default=`TRADING_DAYS`)
        Trading days per year, used to annualize the volatilities.

    Returns
    -------
    pd.DataFrame
        `episodes` with the metric columns appended, indexed by episode start date.
    """
    index = regimes.index
    ret = pd.Series(data["ret"], dtype=float).reindex(index)
    close = pd.Series(data["close"], dtype=float).reindex(index)
    base_ret = (pd.Series(data["excess_ret"], dtype=float).reindex(index)
                if "excess_ret" in data else ret)

    ann = float(np.sqrt(trading_days))
    vol_5 = ret.rolling(5).std() * ann
    vol_20 = ret.rolling(20).std() * ann
    sortino_20 = compute_ewm_sortino(base_ret, 20.)
    drawdown = close / close.cummax() - 1.

    loss_cols = [col for col in regimes.columns if col.startswith("loss_")]
    has_losses = f"loss_{state}" in loss_cols and "loss_0" in loss_cols
    loss_diff = regimes[f"loss_{state}"] - regimes["loss_0"] if has_losses else None

    vol_by_refit = _state_vol_table(params)
    refit_of = regimes["refit_date"] if "refit_date" in regimes else None

    rows = {}
    for start, episode in episodes.iterrows():
        pos = index.get_loc(start)
        days = index[pos:pos + int(episode["length"])]
        fwd = index[pos:pos + horizon]

        row = {
            "loss_diff_mean": float(loss_diff.reindex(days).mean()) if has_losses else np.nan,
            "vol_5_mean": float(vol_5.reindex(days).mean()),
            "vol_20_mean": float(vol_20.reindex(days).mean()),
            "sortino_20_mean": float(sortino_20.reindex(days).mean()),
            "dd_at_entry": float(drawdown.loc[start]),
            "ret_fwd": _cumret(ret.reindex(fwd).dropna()),
            "mdd_fwd": _mdd(ret.reindex(fwd).dropna()),
            "fwd_days": int(len(fwd)),
        }

        state_vol = bull_vol = np.nan
        if vol_by_refit is not None and refit_of is not None:
            refit_date = refit_of.loc[start]
            if refit_date in vol_by_refit.index:
                state_vol = float(vol_by_refit.loc[refit_date].get(state, np.nan))
                bull_vol = float(vol_by_refit.loc[refit_date].get(0, np.nan))
        row["vol_ratio_state"] = (row["vol_20_mean"] / state_vol
                                  if state_vol and np.isfinite(state_vol) else np.nan)
        row["state_flip"] = (float(bull_vol < state_vol)
                             if np.isfinite(bull_vol) and np.isfinite(state_vol) else np.nan)

        if has_losses:
            penalty_days = int((loss_diff.reindex(days) > 0.).sum())
            row["penalty_days"] = penalty_days
            row["distance_days"] = int(episode["length"]) - penalty_days
            row["penalty_share"] = penalty_days / float(episode["length"])
        else:
            row["penalty_days"] = row["distance_days"] = np.nan
            row["penalty_share"] = np.nan
        rows[start] = row

    columns = ["loss_diff_mean", "vol_5_mean", "vol_20_mean", "sortino_20_mean",
               "vol_ratio_state", "state_flip", "dd_at_entry", "ret_fwd", "mdd_fwd",
               "fwd_days", "distance_days", "penalty_days", "penalty_share"]
    metrics = pd.DataFrame(rows, index=columns).T.reindex(episodes.index).astype(float)
    # a Sortino ratio over a window with no losing day divides by zero; that is a missing
    # observation for the comparison, not an infinitely good one, and leaving the infinity in
    # would cost every episode the metric rather than only the one that produced it
    return episodes.join(metrics.replace([np.inf, -np.inf], np.nan))


def rank_similar_episodes(metrics: pd.DataFrame,
                          target=None,
                          columns=None,
                          candidates=None) -> pd.DataFrame:
    """
    Rank episodes by how closely they resemble a target episode.

    The metrics are on wildly different scales -- a drawdown of −0.09 against a loss gap of
    −15 -- so each is standardized across the episodes first, and the distance is the root
    mean square of the standardized gaps. Taking the *mean* rather than the sum is what makes
    a metric that is missing for one episode cost nothing: the gap is dropped and the average
    is taken over the metrics both episodes actually have, which `n_metrics` reports. A
    metric that is constant across the episodes carries no information and is dropped too.

    Parameters
    ----------
    metrics : pd.DataFrame
        The output of `episode_metrics`.

    target : optional
        The start date of the episode to search around. Defaults to the last episode, which
        is the current one when the run reaches the end of the data.

    columns : iterable of str, optional
        The metrics to compare on. Defaults to `DEFAULT_SIMILARITY_METRICS`, restricted to
        the ones `metrics` holds.

    candidates : pd.DataFrame or iterable, optional
        The episodes to rank, as a table (e.g. the output of `select_episodes`) or as start
        dates. Defaults to every episode other than the target.

    Returns
    -------
    pd.DataFrame
        Indexed by candidate start date and sorted by increasing distance, with the
        `distance`, its `rank` (1 = closest) and `n_metrics`, the number of metrics the
        distance was averaged over.
    """
    if metrics.empty:
        raise ValueError("국면이 하나도 없어 유사 국면을 찾을 수 없습니다.")
    target = metrics.index[-1] if target is None else target
    if target not in metrics.index:
        raise KeyError(f"기준 국면 '{target}'을 국면 표에서 찾을 수 없습니다. "
                       f"가능한 시작일: {[str(d) for d in metrics.index]}")

    if columns is None:
        # the default set is a superset of what a reduced table may hold, so it is narrowed
        # rather than enforced; an explicit choice is enforced, since a typo in it is a bug
        columns = [col for col in DEFAULT_SIMILARITY_METRICS if col in metrics.columns]
        if not columns:
            raise KeyError(f"기본 유사도 지표 중 국면 표에 있는 것이 없습니다. "
                           f"--similar-metric 으로 지표를 지정해 주세요. "
                           f"사용 가능한 열: {[c for c in metrics.columns]}")
    else:
        columns = list(columns)
        missing = [col for col in columns if col not in metrics.columns]
        if missing:
            raise KeyError(f"유사도 지표 {missing}가 국면 표에 없습니다. "
                           f"사용 가능한 지표: {[c for c in metrics.columns]}")

    values = metrics[columns].astype(float)
    std = values.std()
    usable = [col for col in columns if np.isfinite(std[col]) and std[col] > 0.]
    if not usable:
        raise ValueError("모든 유사도 지표가 국면 간에 동일하거나 결측이라 거리를 계산할 수 없습니다.")
    z = (values[usable] - values[usable].mean()) / std[usable]

    if candidates is None:
        index = metrics.index.drop(target)
    elif isinstance(candidates, pd.DataFrame):
        index = candidates.index.drop(target, errors="ignore")
    else:
        index = pd.Index([c for c in candidates if c != target])
    if len(index) == 0:
        raise ValueError("비교할 다른 국면이 없습니다. 필터를 완화하거나 더 긴 기간을 사용해 주세요.")

    gaps = (z.reindex(index) - z.loc[target]).abs()
    n_metrics = gaps.notna().sum(axis=1)
    distance = np.sqrt(gaps.pow(2).mean(axis=1, skipna=True))

    ranking = pd.DataFrame({"distance": distance, "n_metrics": n_metrics.astype(int)})
    ranking = ranking.sort_values("distance", kind="stable")
    ranking.insert(1, "rank", np.arange(1, len(ranking) + 1))
    ranking.index.name = metrics.index.name or "start"
    return ranking


def normalized_path(close: pd.Series, start, length: int) -> pd.Series:
    """
    The price path of an episode, indexed by trading day from its entry and set to 1 at it.

    Parameters
    ----------
    close : pd.Series
        The close series, indexed by date.

    start : date
        The entry day, day 1 of the path.

    length : int
        How many trading days to take, from the entry day inclusive.

    Returns
    -------
    pd.Series
        Indexed by 1..n, named after the start date.
    """
    pos = close.index.get_loc(start)
    window = close.iloc[pos:pos + int(length)]
    path = window / float(window.iloc[0])
    return pd.Series(path.to_numpy(), index=np.arange(1, len(path) + 1), name=str(start))


def align_paths(target: pd.Series,
                match: pd.Series,
                max_lag: int = DEFAULT_MAX_LAG,
                criterion: str = "rmse",
                min_overlap: int = MIN_ALIGN_OVERLAP):
    """
    Find the lag that best aligns two normalized paths, and apply it.

    Two episodes can trace the same shape at different speeds -- a more volatile market gets
    to the same drawdown sooner -- and comparing them day by day then understates how alike
    they are. The search shifts the match against the target by every lag in
    ``[−max_lag, max_lag]`` and keeps the best one, where ``target[i]`` is compared against
    ``match[i + lag]``: a *positive* lag means the target reached in `i` days what the match
    took ``i + lag`` days to reach, i.e. the target is running that many days ahead.

    Parameters
    ----------
    target, match : pd.Series
        Normalized paths from `normalized_path`, indexed by trading day from entry.

    max_lag : int, optional (default=`DEFAULT_MAX_LAG`)
        The widest lag considered, in trading days.

    criterion : str, optional (default="rmse")
        "rmse" minimizes the root mean square gap between the two levels, which asks the
        paths to sit on top of each other; "corr" maximizes their correlation, which asks
        only that they move together. Two declining paths correlate highly whatever their
        depth, so "rmse" is the stricter of the two.

    min_overlap : int, optional (default=`MIN_ALIGN_OVERLAP`)
        The fewest overlapping days a lag must leave to be considered.

    Returns
    -------
    dict
        ``lag``, the chosen lag; ``rmse`` and ``corr`` at that lag; ``n_overlap``; and
        ``aligned``, a two-column frame of the overlapping part of both paths, indexed by
        the target's trading day.
    """
    if criterion not in ALIGN_CRITERIA:
        raise ValueError(f"지원하지 않는 정렬 기준입니다: {criterion}. 가능한 값: {ALIGN_CRITERIA}")
    target = pd.Series(target, dtype=float)
    match = pd.Series(match, dtype=float)

    best = None
    for lag in range(-int(max_lag), int(max_lag) + 1):
        shifted = pd.Series(match.to_numpy(), index=match.index - lag)
        pair = pd.concat([target.rename("target"), shifted.rename("match")], axis=1).dropna()
        if len(pair) < min_overlap:
            continue
        rmse = float(np.sqrt(((pair["target"] - pair["match"]) ** 2).mean()))
        corr = float(pair["target"].corr(pair["match"]))
        score = rmse if criterion == "rmse" else -(corr if np.isfinite(corr) else -1.)
        if best is None or score < best["score"]:
            best = {"score": score, "lag": lag, "rmse": rmse, "corr": corr,
                    "n_overlap": len(pair), "aligned": pair}
    if best is None:
        raise ValueError(f"두 경로가 {min_overlap}일 이상 겹치는 시차가 없습니다. "
                         f"--path-horizon 을 늘리거나 --max-lag 를 줄여 주세요.")
    best.pop("score")
    best["aligned"].index.name = "day"
    return best


def compare_episode_paths(close: pd.Series,
                          target_start,
                          match_start,
                          horizon: int = DEFAULT_PATH_HORIZON,
                          max_lag: int = DEFAULT_MAX_LAG,
                          criterion: str = "rmse") -> dict:
    """
    Put the normalized path of a target episode next to that of its match, raw and aligned.

    Parameters
    ----------
    close : pd.Series
        The close series, indexed by date and covering both episodes.

    target_start, match_start : date
        The entry days of the two episodes.

    horizon : int, optional (default=`DEFAULT_PATH_HORIZON`)
        How many trading days of each path to take. The target is cut at the end of the data
        when it is still running, so the two paths need not be the same length.

    max_lag : int, optional (default=`DEFAULT_MAX_LAG`)
        The widest lag the alignment searches, see `align_paths`.

    criterion : str, optional (default="rmse")
        The alignment criterion, see `align_paths`.

    Returns
    -------
    dict
        ``paths``, the two normalized paths side by side on the raw day axis; and the whole
        output of `align_paths` -- ``aligned``, ``lag``, ``rmse``, ``corr``, ``n_overlap``.
    """
    target = normalized_path(close, target_start, horizon).rename("target")
    match = normalized_path(close, match_start, horizon).rename("match")
    result = align_paths(target, match, max_lag=max_lag, criterion=criterion)
    paths = pd.concat([target, match], axis=1)
    paths.index.name = "day"
    result["paths"] = paths
    result["target_start"] = target_start
    result["match_start"] = match_start
    return result


def _projected_end(index, start, length: int):
    """
    The date `length` trading days after `start`, extrapolated past the end of the sample.

    Inside the sample the trading calendar is the data's own index; beyond it there is no
    calendar to follow, so the remaining days are counted as business days, which ignores
    holidays and therefore lands a little early on a long projection.
    """
    pos = index.get_loc(start)
    end_pos = pos + int(length) - 1
    if end_pos < len(index):
        return index[end_pos]
    extra = end_pos - (len(index) - 1)
    return pd.bdate_range(pd.Timestamp(index[-1]), periods=extra + 1)[-1].date()


def length_scenarios(episodes: pd.DataFrame,
                     index,
                     target=None,
                     similar: pd.DataFrame = None,
                     top: int = 3,
                     quantiles=(.25, .5, .75),
                     min_length: int = 0,
                     drop_false_signal: bool = False,
                     before=None) -> pd.DataFrame:
    """
    Turn the lengths of the past episodes into projected end dates for a target episode.

    Two readings of "how long does this last" are tabulated side by side: the quantiles of
    every past episode, which is the outside view, and the length of each of the closest
    matches from `rank_similar_episodes`, which is the inside one. Each becomes a total
    length for the target episode, and hence a date.

    Parameters
    ----------
    episodes : pd.DataFrame
        The output of `extract_episodes` or `episode_metrics`.

    index : pd.Index
        The trading days of the sample, used to turn a length into a date.

    target : optional
        The episode to project. Defaults to the last one.

    similar : pd.DataFrame, optional
        The output of `rank_similar_episodes`. Without it only the quantile rows are built.

    top : int, optional (default=3)
        How many of the closest matches to tabulate.

    quantiles : tuple of float, optional (default=(.25, .5, .75))
        The quantiles of the past lengths to report.

    min_length, drop_false_signal, before : optional
        Filters applied to the episodes the quantiles are taken over, see `select_episodes`.
        The two narrowing filters are off by default; `before` should normally be the target
        start, so that the quantiles rest on the episodes that preceded it.

    Returns
    -------
    pd.DataFrame
        One row per scenario, with the basis it rests on, the implied total `length_days`,
        the `elapsed_days` of the target so far, the `remaining_days` and the
        `projected_end` date. A scenario shorter than what has already elapsed comes out
        with a negative `remaining_days`, which is the honest way to say the target has
        already outlasted it.
    """
    target = resolve_target_episode(episodes, target)
    elapsed = int(episodes.loc[target, "length"])

    past = select_episodes(episodes, min_length=min_length, drop_false_signal=drop_false_signal,
                           drop_ongoing=True, before=before, exclude=[target])
    rows = []
    if not past.empty:
        note = f"과거 국면 {len(past)}건"
        for q in quantiles:
            length = float(past["length"].quantile(q))
            rows.append({"scenario": f"p{q * 100:g}", "basis": note, "length_days": length})
    if similar is not None and not similar.empty:
        for start, row in similar.head(top).iterrows():
            if start not in episodes.index:
                continue
            rows.append({"scenario": f"similar-{int(row['rank'])}",
                         "basis": f"{start} (거리 {row['distance']:.2f})",
                         "length_days": float(episodes.loc[start, "length"])})
    if not rows:
        return pd.DataFrame(columns=["scenario", "basis", "length_days", "elapsed_days",
                                     "remaining_days", "projected_end"]).set_index("scenario")

    scenarios = pd.DataFrame(rows).set_index("scenario")
    scenarios["elapsed_days"] = elapsed
    scenarios["remaining_days"] = scenarios["length_days"] - elapsed
    scenarios["projected_end"] = [_projected_end(index, target, max(int(round(n)), 1))
                                  for n in scenarios["length_days"]]
    return scenarios
