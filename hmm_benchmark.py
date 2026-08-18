"""
Hidden Markov model benchmark, following Section 3.3 of Shu, Yu and Mulvey (2024).

A two-state Gaussian HMM is fitted on daily log returns over a rolling window of 3000
trading days. At the end of every day the Viterbi algorithm decodes the hidden state
sequence over the trailing window and the last state is taken as the online inferred
regime; that sequence is then smoothed by a median filter of length `k`, i.e. the high
volatility regime is only signalled when the trailing mean of the raw states exceeds 0.5.

The two states are distinguished by their conditional volatilities: state 0 is the low
volatility (favorable) regime and state 1 the high volatility one.

Requires the optional dependency `hmmlearn` (`pip install hmmlearn`).
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _import_hmmlearn():
    """Import `hmmlearn` lazily, with an actionable message when it is missing."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as err:      # pragma: no cover - depends on the environment
        raise ImportError(
            "HMM 벤치마크에는 hmmlearn 패키지가 필요합니다. `pip install hmmlearn` 후 다시 실행해 주세요."
        ) from err
    # hmmlearn logs a line per fit whose log-likelihood wiggles at the convergence
    # threshold; with hundreds of rolling fits that noise buries the actual progress.
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    return GaussianHMM


@dataclass
class FittedHMM:
    """
    A fitted Gaussian HMM together with the permutation sorting its states by volatility.

    Attributes
    ----------
    model : hmmlearn.hmm.GaussianHMM
        The fitted model, with its states in the arbitrary order returned by the fit.

    rank : np.ndarray
        Maps a state index of `model` to its volatility rank, so that 0 is the lowest
        volatility state.

    means : np.ndarray
        The conditional means, ordered by volatility.

    vols : np.ndarray
        The conditional volatilities, in increasing order.

    transmat : np.ndarray
        The transition probability matrix, reordered by volatility.

    score : float
        The log-likelihood of the retained fit.
    """
    model: object
    rank: np.ndarray
    means: np.ndarray
    vols: np.ndarray
    transmat: np.ndarray
    score: float


def fit_gaussian_hmm(x: np.ndarray,
                     n_components: int = 2,
                     n_init: int = 10,
                     covariance_type: str = "diag",
                     n_iter: int = 100,
                     tol: float = 1e-4,
                     random_state: int = 0) -> FittedHMM:
    """
    Fit a Gaussian HMM several times and keep the fit with the highest log-likelihood.

    The likelihood of an HMM is not concave, so -- as in the article -- the algorithm is
    run from several initializations and only the best fit is retained.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples, 1)
        The observation sequence, typically daily log returns.

    n_components : int, optional (default=2)
        The number of hidden states.

    n_init : int, optional (default=10)
        The number of initializations.

    covariance_type : str, optional (default="diag")
        The covariance parameterization passed to `hmmlearn`.

    n_iter : int, optional (default=100)
        The maximum number of EM iterations per fit.

    tol : float, optional (default=1e-4)
        The EM convergence tolerance.

    random_state : int, optional (default=0)
        The seed of the first initialization; subsequent ones use `random_state + i`.

    Returns
    -------
    FittedHMM or None
        The best fit, or None when every initialization failed.
    """
    GaussianHMM = _import_hmmlearn()
    best, best_score = None, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(max(1, n_init)):
            model = GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                                n_iter=n_iter, tol=tol, random_state=random_state + i)
            try:
                model.fit(x)
                score = float(model.score(x))
            except Exception:       # a degenerate initialization: try the next one
                continue
            if np.isfinite(score) and score > best_score:
                best, best_score = model, score
    if best is None:
        return None

    vols = np.sqrt(np.asarray(best.covars_).reshape(best.n_components, -1)[:, 0])
    order = np.argsort(vols)                    # state 0 = lowest volatility
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return FittedHMM(model=best,
                     rank=rank,
                     means=np.asarray(best.means_).ravel()[order],
                     vols=vols[order],
                     transmat=np.asarray(best.transmat_)[np.ix_(order, order)],
                     score=best_score)


def smooth_states(raw: pd.Series, k: int = 6) -> pd.Series:
    """
    Apply the median filter of Bulla et al. (2011) to an online inferred state sequence.

    The high volatility regime is signalled only when the mean of the last `k` raw states
    exceeds 0.5, which removes the isolated flips caused by the identification errors at
    the end of the training window.

    Parameters
    ----------
    raw : pd.Series
        The raw online inferred states, in {0, 1}.

    k : int, optional (default=6)
        The length of the trailing window. Values below 2 leave the sequence unchanged.

    Returns
    -------
    pd.Series
        The smoothed state sequence, in {0, 1}.
    """
    if k is None or k < 2:
        return raw.astype(int)
    return (raw.rolling(k, min_periods=1).mean() > .5).astype(int)


@dataclass
class RollingHMMResult:
    """
    Container for the output of `run_rolling_hmm`.

    Attributes
    ----------
    regimes : pd.DataFrame
        Indexed by date, with the raw online inferred state, the smoothed regime
        (0 = low volatility, 1 = high volatility) and the refit date in use.

    params : pd.DataFrame
        One row per (refit date, state): the training window, the annualized conditional
        mean and volatility, and the self-transition probability.

    schedule : list
        The `(date, position, window)` triples of the re-estimation schedule.

    window : int
        The intended training window length.

    smooth_k : int
        The length of the median filter.
    """
    regimes: pd.DataFrame
    params: pd.DataFrame
    schedule: list = field(default_factory=list)
    window: int = 3000
    smooth_k: int = 6


def run_rolling_hmm(ret_ser: pd.Series,
                    window: int = 3000,
                    min_window: int = 500,
                    refit_every: int = 21,
                    smooth_k: int = 6,
                    n_components: int = 2,
                    n_init: int = 10,
                    covariance_type: str = "diag",
                    n_iter: int = 100,
                    random_state: int = 0,
                    use_log: bool = True,
                    start_date=None,
                    verbose: bool = True) -> RollingHMMResult:
    """
    Run the rolling HMM benchmark and infer the prevailing regime of every day online.

    The parameters are re-estimated every `refit_every` trading days on the `window` days
    that *precede* the re-estimation date, and the state of every single day is decoded by
    the Viterbi algorithm over its own trailing window, so no future data is ever used.

    The article re-estimates the parameters daily, which costs one full EM fit per day;
    `refit_every` trades that cost against fidelity in the same way the jump model is
    re-estimated every six months. Set it to 1 to follow the article exactly.

    Parameters
    ----------
    ret_ser : pd.Series
        The daily total return series of the risky asset.

    window : int, optional (default=3000)
        The training and lookback window length, in trading days.

    min_window : int, optional (default=500)
        The shortest window accepted; inference starts once this much history exists.

    refit_every : int, optional (default=21)
        Number of trading days between two EM re-estimations. 1 reproduces the article,
        21 re-estimates monthly.

    smooth_k : int, optional (default=6)
        The length of the median filter applied to the online inferred states.

    n_components : int, optional (default=2)
        The number of hidden states.

    n_init : int, optional (default=10)
        The number of initializations per fit.

    covariance_type : str, optional (default="diag")
        The covariance parameterization passed to `hmmlearn`.

    n_iter : int, optional (default=100)
        The maximum number of EM iterations per fit.

    random_state : int, optional (default=0)
        The seed of the initializations.

    use_log : bool, optional (default=True)
        Whether to model log returns, as the article does, rather than simple returns.

    start_date : str or datetime.date, optional
        Start the online inference at this date instead of as early as the data allows.

    verbose : bool, optional (default=True)
        Whether to print the progress of the re-estimations.

    Returns
    -------
    RollingHMMResult
        The online inferred regimes and the estimated parameters of every re-estimation.
    """
    if refit_every < 1:
        raise ValueError(f"refit_every는 1 이상이어야 합니다. 입력값: {refit_every}")
    if min_window > window:
        raise ValueError(f"min_window({min_window})는 window({window})보다 클 수 없습니다.")

    ret_ser = ret_ser.dropna()
    values = np.log1p(ret_ser.to_numpy()) if use_log else ret_ser.to_numpy()
    x = values.reshape(-1, 1)
    n_obs = len(x)

    first_pos = min_window
    if start_date is not None:
        target = pd.Timestamp(start_date).date()
        first_pos = max(first_pos, int(pd.DatetimeIndex(pd.to_datetime(pd.Index(ret_ser.index)))
                                       .searchsorted(pd.Timestamp(target), side="left")))
    if first_pos >= n_obs:
        raise ValueError(
            f"HMM 온라인 추론을 시작할 수 없습니다. 수익률 {n_obs}행으로는 최소 학습창 "
            f"{min_window}거래일을 확보할 수 없습니다.")
    if first_pos < window:
        warnings.warn(
            f"HMM 학습창이 데이터 부족으로 축소됩니다 (시작 시점 {ret_ser.index[first_pos]}에서 "
            f"{first_pos}거래일, 목표: {window}거래일).")

    fitted, schedule, param_rows = None, [], []
    raw_states, refit_used = np.empty(n_obs - first_pos, dtype=int), []
    n_refits_expected = int(np.ceil((n_obs - first_pos) / refit_every))
    current_refit_date = None

    for i, pos in enumerate(range(first_pos, n_obs)):
        win = min(window, pos)
        if i % refit_every == 0:
            candidate = fit_gaussian_hmm(x[pos - win:pos], n_components=n_components, n_init=n_init,
                                         covariance_type=covariance_type, n_iter=n_iter,
                                         random_state=random_state)
            if candidate is None:
                if fitted is None:
                    raise RuntimeError(
                        f"{ret_ser.index[pos]}의 HMM 적합이 모든 초기값에서 실패했습니다.")
                warnings.warn(f"{ret_ser.index[pos]}의 HMM 적합이 실패하여 직전 파라미터를 유지합니다.")
            else:
                fitted = candidate
                current_refit_date = ret_ser.index[pos]
                schedule.append((current_refit_date, pos, win))
                for k in range(n_components):
                    param_rows.append({
                        "refit_date": current_refit_date,
                        "train_start": ret_ser.index[pos - win],
                        "train_end": ret_ser.index[pos - 1],
                        "n_train": win,
                        "state": k,
                        "regime": "low_vol" if k == 0 else ("high_vol" if k == n_components - 1 else f"mid{k}"),
                        "ret_ann": float(fitted.means[k] * TRADING_DAYS),
                        "vol_ann": float(fitted.vols[k] * np.sqrt(TRADING_DAYS)),
                        "stay_prob": float(fitted.transmat[k, k]),
                        "loglik": fitted.score,
                    })
                if verbose and (len(schedule) % max(1, n_refits_expected // 10) == 0 or len(schedule) == 1):
                    print(f"[HMM {len(schedule)}/{n_refits_expected}] refit {current_refit_date}: "
                          f"학습창 {ret_ser.index[pos - win]} ~ {ret_ser.index[pos - 1]} ({win}일), "
                          f"연율 변동성 {fitted.vols[0] * np.sqrt(TRADING_DAYS):.1%} / "
                          f"{fitted.vols[-1] * np.sqrt(TRADING_DAYS):.1%}")

        # Viterbi over the trailing window; the last state is the online inference for `pos`
        decoded = fitted.model.predict(x[pos - win + 1:pos + 1])
        raw_states[i] = int(fitted.rank[decoded[-1]])
        refit_used.append(current_refit_date)

    index = ret_ser.index[first_pos:]
    raw = pd.Series(raw_states, index=index, name="state_raw")
    regimes = pd.DataFrame({
        "state_raw": raw,
        "regime": smooth_states(raw, k=smooth_k),
        "refit_date": refit_used,
    })
    regimes.index.name = "date"
    return RollingHMMResult(regimes=regimes,
                            params=pd.DataFrame(param_rows),
                            schedule=schedule,
                            window=window,
                            smooth_k=smooth_k)
