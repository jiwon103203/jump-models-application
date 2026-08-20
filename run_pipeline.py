#!/usr/bin/env python
"""
End-to-end pipeline: a csv/Excel file of date, close price and risk-free rate in, and
regime signals plus a 0/1 strategy backtest out, following Shu, Yu and Mulvey (2024).

Example
-------
    python run_pipeline.py --input my_index.xlsx --outdir out
    python run_pipeline.py --input my_index.xlsx --hmm --extra-feature VIX:ewm:20

Run `python run_pipeline.py --help` for the full list of options.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (DEFAULT_COST_BPS, DEFAULT_MAX_CASH, DEFAULT_MIN_CASH,
                      delay_robustness_table, format_performance_table, performance_table,
                      regime_summary, resolve_cash_limits, resolve_cost_bps, run_0_1_strategy)
from data_io import (RF_UNITS, TRADING_DAYS, join_extra_table, load_extra_table,
                     load_market_data, normalize_header)
from features import (FEATURE_SETS, build_extra_features, build_features, parse_extra_spec,
                      resolve_pinned_features)
from rolling import MODELS, run_rolling_jm


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        description="회귀 레짐 스위칭 신호(JM) 파이프라인: csv/엑셀 → 레짐 신호 → 0/1 전략 백테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group("입력 데이터")
    group.add_argument("--input", required=True, help="csv 또는 엑셀 파일 경로 (날짜/종가/무위험금리 열)")
    group.add_argument("--sheet", default=None, help="엑셀 시트 이름 또는 번호")
    group.add_argument("--date-col", default=None, help="날짜 열 이름 (미지정 시 자동 인식)")
    group.add_argument("--close-col", default=None, help="종가 열 이름 (미지정 시 자동 인식)")
    group.add_argument("--rf-col", default=None, help="무위험금리 열 이름 (미지정 시 자동 인식)")
    group.add_argument("--rf-unit", default="annual_percent", choices=RF_UNITS,
                       help="무위험금리 단위: 연율 %%(4.25) / 연율 소수(0.0425) / 이미 일간")
    group.add_argument("--trading-days", type=int, default=TRADING_DAYS, help="연간 거래일 수")
    group.add_argument("--start-date", default=None, help="분석 시작일 (예: 1990-01-01)")
    group.add_argument("--end-date", default=None, help="분석 종료일")

    group = parser.add_argument_group("피처")
    group.add_argument("--feature-set", default="paper", choices=FEATURE_SETS,
                       help="paper: 논문 Table 2의 3개 피처 / example: 레포 예제의 9개 피처 / "
                            "extra: example 9개 + 수익률·변동성 파생 25개 + 원 스케일 "
                            "DD_10/DD_20/DD_60 = 37개 (--log-dd 시 35개, --model sjm 권장)")
    group.add_argument("--log-dd", action="store_true",
                       help="원 스케일 downside deviation(paper의 DD_10, extra의 DD_10/20/60)을 "
                            "로그 변환. extra에서는 DD_20/DD_60이 example의 DD-log_20/60과 "
                            "같아지므로 빠집니다")
    group.add_argument("--warmup", type=int, default=252, help="EWM 워밍업으로 버릴 초기 행 수")

    group = parser.add_argument_group("커스텀 변수")
    group.add_argument("--extra-feature", action="append", default=None, metavar="SPEC",
                       help="커스텀 변수를 피처로 추가. 형식은 '열이름[:변환[:파라미터]]' "
                            "(변환: raw/ewm/diff/pct/log/logdiff). 예: --extra-feature VIX:ewm:20. "
                            "여러 번 지정할 수 있습니다")
    group.add_argument("--extra-file", default=None,
                       help="커스텀 변수가 든 별도 csv/엑셀 파일. 날짜 기준으로 결합되며 "
                            "저빈도 값은 직전 값으로 채웁니다. --extra-feature 없이 쓰면 모든 열을 그대로 사용")
    group.add_argument("--extra-sheet", default=None, help="커스텀 변수 파일의 엑셀 시트")
    group.add_argument("--extra-date-col", default=None, help="커스텀 변수 파일의 날짜 열 이름")

    group = parser.add_argument_group("모델 & 재추정")
    group.add_argument("--model", default="jm", choices=MODELS,
                       help="jm: 논문의 원본(이산) 점프 모델 / sjm: 피처 선택이 있는 sparse 점프 모델. "
                            "커스텀 변수로 피처를 늘렸을 때 sjm이 노이즈 피처를 걸러 줍니다")
    group.add_argument("--max-feats", type=float, default=None,
                       help="sjm 전용. 남길 유효 피처 개수(kappa^2). 미지정 시 피처 개수의 절반")
    group.add_argument("--pin-feature", action="append", default=None, metavar="NAME",
                       help="sjm 전용. 재추정마다 항상 남길 피처. 피처 세트 이름(paper/example/extra), "
                            "커스텀 변수 전체를 뜻하는 custom, 전체를 뜻하는 all, 개별 피처 이름"
                            "(예: sortino_60), 또는 커스텀 변수 지정(예: VIX:ewm:20)을 받습니다. "
                            "여러 번 지정할 수 있습니다")
    group.add_argument("--jump-penalty", type=float, default=50., help="점프 페널티 lambda")
    group.add_argument("--window", type=int, default=3000, help="학습창 길이 (거래일)")
    group.add_argument("--min-window", type=int, default=500,
                       help="허용하는 최소 학습창 길이. 데이터가 부족하면 이 길이까지 자동 축소")
    group.add_argument("--n-components", type=int, default=2, help="레짐 개수")
    group.add_argument("--clip-mul", type=float, default=3., help="윈저라이징 표준편차 배수")
    group.add_argument("--n-init", type=int, default=10, help="좌표하강 알고리즘 재시작 횟수")
    group.add_argument("--random-state", type=int, default=0, help="초기화 난수 시드")
    group.add_argument("--refit-start", default=None, help="이 날짜 이후의 재추정 시점만 사용")

    group = parser.add_argument_group("HMM 벤치마크 (hmmlearn 필요)")
    group.add_argument("--hmm", action="store_true", help="2-state 가우시안 HMM 벤치마크를 함께 실행")
    group.add_argument("--hmm-window", type=int, default=None,
                       help="HMM 학습·lookback 창 길이. 미지정 시 --window 와 동일")
    group.add_argument("--hmm-refit-every", type=int, default=21,
                       help="HMM 파라미터 재추정 주기(거래일). 1이면 논문과 동일한 매일 재추정이지만 매우 느림")
    group.add_argument("--hmm-smooth-k", type=int, default=6,
                       help="온라인 추론 상태에 적용할 median filter 길이 (Bulla et al. 2011)")
    group.add_argument("--hmm-n-init", type=int, default=10, help="HMM 적합 초기값 개수")
    group.add_argument("--hmm-covariance-type", default="diag", help="hmmlearn 공분산 형태")
    group.add_argument("--hmm-simple-ret", action="store_true",
                       help="로그수익률 대신 단순수익률로 HMM을 적합")

    group = parser.add_argument_group("전략")
    group.add_argument("--delay", type=int, default=1,
                       help="거래 지연 일수. t일 신호는 t+delay+1일부터 적용")
    group.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                       help="편도 거래비용 (bp). --buy-cost-bps/--sell-cost-bps를 주지 않은 쪽에 적용")
    group.add_argument("--buy-cost-bps", type=float, default=None,
                       help="매수 거래비용 (bp). 미지정 시 --cost-bps 사용 (예: 10)")
    group.add_argument("--sell-cost-bps", type=float, default=None,
                       help="매도 거래비용 (bp). 증권거래세처럼 매도가 더 비쌀 때 지정 (예: 25)")
    group.add_argument("--max-cash", type=float, default=DEFAULT_MAX_CASH,
                       help="bear 레짐에서 허용하는 최대 현금 비중 (0~1). "
                            "1이면 논문과 동일하게 100%% 현금, 0.5면 위험자산을 절반만 줄임")
    group.add_argument("--min-cash", type=float, default=DEFAULT_MIN_CASH,
                       help="bull 레짐에서도 항상 유지할 최소 현금 비중 (0~1)")
    group.add_argument("--delays", default="1,5,10",
                       help="거래 지연 로버스트니스 표(논문 Table 5)에 사용할 지연 일수 목록")
    group.add_argument("--no-robustness", action="store_true", help="지연 로버스트니스 표를 건너뜀")

    group = parser.add_argument_group("출력")
    group.add_argument("--outdir", default="out", help="결과 저장 폴더")
    group.add_argument("--no-plot", action="store_true", help="플롯 생성을 건너뜀")
    group.add_argument("--save-features", action="store_true", help="피처 행렬도 csv로 저장")
    group.add_argument("--plot-font", default=None,
                       help="그림에 사용할 폰트 이름. 한글 라벨이 깨질 때 지정 (예: NanumGothic)")
    group.add_argument("--quiet", action="store_true", help="진행 상황 출력을 최소화")
    return parser


def parse_delays(delays) -> tuple:
    """Turn the `--delays` option into a tuple of non-negative integers."""
    if isinstance(delays, str):
        delays = [part for part in delays.replace(" ", "").split(",") if part]
    out = []
    for value in delays:
        delay = int(value)
        if delay < 0:
            raise ValueError(f"거래 지연은 0 이상이어야 합니다: {delay}")
        out.append(delay)
    if not out:
        raise ValueError("거래 지연 목록이 비어 있습니다.")
    return tuple(out)


def load_data_with_extras(input_path: str,
                          extra_features=None,
                          extra_file: str = None,
                          extra_sheet=None,
                          extra_date_col: str = None,
                          **load_kwargs) -> tuple:
    """
    Load the market data together with any custom variables, and build their features.

    Custom variables may live in the main file, in a second file joined on the date, or in
    both. When a second file is given without any explicit specification, every one of its
    columns is used as a feature as is.

    Parameters
    ----------
    input_path : str
        Path of the main csv/Excel file.

    extra_features : iterable of str, optional
        Specifications of the form ``column[:transform[:param]]``, see
        `features.parse_extra_spec`.

    extra_file, extra_sheet, extra_date_col : optional
        The second file holding custom variables, its sheet and its date column.

    **load_kwargs
        Passed through to `data_io.load_market_data`.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame or None)
        The market data including the raw custom variables, and the transformed custom
        features (None when no custom variable was requested).
    """
    specs = list(extra_features or [])
    extra_table = None
    if extra_file is not None:
        wanted = [parse_extra_spec(spec)[0] for spec in specs] or None
        extra_table = load_extra_table(extra_file, sheet=extra_sheet, date_col=extra_date_col,
                                       columns=None)
        if wanted:      # keep only the columns the specs actually need, if present here
            by_header = {normalize_header(col): col for col in extra_table.columns}
            keep, rename = [], {}
            for name in wanted:
                col = by_header.get(normalize_header(name))
                if col is not None:
                    keep.append(col)
                    rename[col] = name      # refer to it under the name used in the spec
            extra_table = extra_table[keep].rename(columns=rename) if keep else None
        else:
            specs = list(extra_table.columns)

    from_main = [col for col in dict.fromkeys(parse_extra_spec(spec)[0] for spec in specs)
                 if extra_table is None or col not in extra_table.columns]
    data = load_market_data(input_path, extra_cols=from_main, **load_kwargs)
    if extra_table is not None:
        data = join_extra_table(data, extra_table)
    extra_df = build_extra_features(data, specs) if specs else None
    return data, extra_df


def run_pipeline(input_path: str,
                 outdir: str = "out",
                 sheet=None,
                 date_col=None,
                 close_col=None,
                 rf_col=None,
                 rf_unit: str = "annual_percent",
                 trading_days: int = TRADING_DAYS,
                 start_date=None,
                 end_date=None,
                 feature_set: str = "paper",
                 log_dd: bool = False,
                 warmup: int = 252,
                 extra_features=None,
                 extra_file: str = None,
                 extra_sheet=None,
                 extra_date_col: str = None,
                 model: str = "jm",
                 max_feats: float = None,
                 pin_features=None,
                 jump_penalty: float = 50.,
                 window: int = 3000,
                 min_window: int = 500,
                 n_components: int = 2,
                 clip_mul: float = 3.,
                 n_init: int = 10,
                 random_state: int = 0,
                 refit_start=None,
                 hmm: bool = False,
                 hmm_window: int = None,
                 hmm_refit_every: int = 21,
                 hmm_smooth_k: int = 6,
                 hmm_n_init: int = 10,
                 hmm_covariance_type: str = "diag",
                 hmm_simple_ret: bool = False,
                 delay: int = 1,
                 cost_bps: float = DEFAULT_COST_BPS,
                 buy_cost_bps: float = None,
                 sell_cost_bps: float = None,
                 min_cash: float = DEFAULT_MIN_CASH,
                 max_cash: float = DEFAULT_MAX_CASH,
                 delays=(1, 5, 10),
                 robustness: bool = True,
                 plot: bool = True,
                 plot_font: str = None,
                 save_features: bool = False,
                 verbose: bool = True) -> dict:
    """
    Run the whole pipeline and write the results to `outdir`.

    The steps are: load and clean the price file, convert prices into excess returns,
    engineer the EWM downside deviation and Sortino features (plus any custom variable),
    re-estimate the jump model every six months over a rolling window while inferring the
    regimes online in between, optionally run the rolling HMM benchmark, and backtest the
    0/1 strategy -- under the main trading delay and, for the robustness table, under
    several delays. `min_cash`/`max_cash` bound the share held in the risk-free asset,
    `buy_cost_bps`/`sell_cost_bps` charge the two legs of a trade separately, and
    `pin_features` names the features the sparse model must keep at every re-estimation.

    Parameters
    ----------
    input_path : str
        Path of the csv/Excel file holding the date, close price and risk-free rate columns.

    outdir : str, optional (default="out")
        Folder the csv results and figures are written to.

    Other parameters mirror the command line options; see `build_parser`.

    Returns
    -------
    dict
        The intermediate objects: `data`, `X` (features), `result` (RollingJMResult),
        `strategy`, `performance`, `summary`, optionally `hmm_result`, `hmm_strategy` and
        `robustness`, and the list of written files.
    """
    os.makedirs(outdir, exist_ok=True)
    # fail fast on bad limits, and keep the resolved values for the log line and the plot title
    bull_weight, bear_weight = resolve_cash_limits(min_cash, max_cash)
    buy_bps, sell_bps = resolve_cost_bps(cost_bps, buy_cost_bps, sell_cost_bps)
    cost_kwargs = {"cost_bps": cost_bps, "buy_cost_bps": buy_cost_bps,
                   "sell_cost_bps": sell_cost_bps, "min_cash": min_cash, "max_cash": max_cash}

    # 1) raw file(s) -> daily returns, excess returns and custom variables
    data, extra_df = load_data_with_extras(
        input_path, extra_features=extra_features, extra_file=extra_file,
        extra_sheet=extra_sheet, extra_date_col=extra_date_col,
        sheet=sheet, date_col=date_col, close_col=close_col, rf_col=rf_col,
        rf_unit=rf_unit, trading_days=trading_days, start_date=start_date, end_date=end_date)
    if verbose:
        print(f"데이터: {len(data)}거래일, {data.index[0]} ~ {data.index[-1]}")
        print(f"연율 무위험금리 평균: {data.rf.mean() * trading_days:.2%}, "
              f"자산 연율 수익률 평균: {data.ret.mean() * trading_days:.2%}")
        if extra_df is not None:
            print(f"커스텀 변수: {list(extra_df.columns)}")

    # 2) features from the excess return series (+ custom variables)
    X = build_features(data.excess_ret, ver=feature_set, warmup=warmup, log_dd=log_dd,
                       extra_features=extra_df)
    if verbose:
        print(f"피처({feature_set}): {list(X.columns)} / {len(X)}행, {X.index[0]} ~ {X.index[-1]}")

    # feature-set names such as "paper" become the columns they stand for
    pinned = resolve_pinned_features(X.columns, pin_features, ver=feature_set, log_dd=log_dd)
    if verbose and pinned and model == "sjm":       # `run_rolling_jm` warns and ignores them otherwise
        print(f"고정 피처: {pinned} ({len(pinned)}/{X.shape[1]}개, 재추정마다 항상 유지)")

    # 3) semiannual refits on a rolling window + online inference in between
    result = run_rolling_jm(X, data.excess_ret, jump_penalty=jump_penalty, window=window,
                            min_window=min_window, n_components=n_components, clip_mul=clip_mul,
                            n_init=n_init, random_state=random_state, start_date=refit_start,
                            model=model, max_feats=max_feats, pin_feats=pinned, verbose=verbose)

    # 4) 0/1 strategy backtest on the online inferred signal
    strategy = run_0_1_strategy(result.regimes.regime, data.ret, data.rf,
                                delay=delay, bull_state=0, **cost_kwargs)
    summary = regime_summary(result.regimes.regime, bear_state=n_components - 1)

    # 5) optional HMM benchmark over the same period
    hmm_result = hmm_strategy = hmm_summary = None
    model_label = f"{result.model.upper()} 0/1"
    performance = performance_table(strategy, trading_days=trading_days, label=model_label)
    if hmm:
        from hmm_benchmark import run_rolling_hmm
        hmm_result = run_rolling_hmm(data.ret, window=hmm_window or window, min_window=min_window,
                                     refit_every=hmm_refit_every, smooth_k=hmm_smooth_k,
                                     n_components=n_components, n_init=hmm_n_init,
                                     covariance_type=hmm_covariance_type,
                                     random_state=random_state, use_log=not hmm_simple_ret,
                                     start_date=result.regimes.index[0], verbose=verbose)
        hmm_summary = regime_summary(hmm_result.regimes.regime, bear_state=n_components - 1)
        # compare both models over the period they share
        common = result.regimes.index.intersection(hmm_result.regimes.index)
        hmm_strategy = run_0_1_strategy(hmm_result.regimes.regime.reindex(common), data.ret,
                                        data.rf, delay=delay, bull_state=0, **cost_kwargs)
        jm_aligned = strategy
        if len(common) < len(strategy):
            warnings.warn(
                f"{result.model.upper()}({strategy.index[0]}~)과 HMM({hmm_result.regimes.index[0]}~)의 추론 구간이 달라 "
                f"공통 구간 {common[0]} ~ {common[-1]}에서 성과를 비교합니다.")
            jm_aligned = run_0_1_strategy(result.regimes.regime.reindex(common), data.ret, data.rf,
                                          delay=delay, bull_state=0, **cost_kwargs)
        performance = performance_table(jm_aligned, trading_days=trading_days, label=model_label,
                                        others={"HMM 0/1": hmm_strategy})

    # 6) trading delay robustness (Table 5 of the article)
    robustness_table, delay_list = None, parse_delays(delays)
    if robustness:
        regime_dict = {result.model.upper(): result.regimes.regime}
        if hmm_result is not None:
            regime_dict["HMM"] = hmm_result.regimes.regime
        robustness_table = delay_robustness_table(regime_dict, data.ret, data.rf,
                                                  delays=delay_list, trading_days=trading_days,
                                                  **cost_kwargs)

    # 7) write everything out
    written = []
    regimes_out = result.regimes.join(data[["close", "ret", "rf", "excess_ret"]])
    regimes_out = regimes_out.join(strategy[["weight", "jm"]].rename(columns={"jm": "strategy_ret"}))
    outputs = [("regimes.csv", regimes_out),
               ("refit_params.csv", result.params.set_index("refit_date")),
               ("strategy.csv", strategy),
               ("performance.csv", performance),
               ("insample_last_window.csv", result.insample_last)]
    if hmm_result is not None:
        outputs += [("hmm_regimes.csv", hmm_result.regimes),
                    ("hmm_refit_params.csv", hmm_result.params.set_index("refit_date")),
                    ("hmm_strategy.csv", hmm_strategy)]
    if result.feat_weights is not None:
        outputs.append(("feat_weights.csv", result.feat_weights))
    if robustness_table is not None:
        outputs.append(("delay_robustness.csv", robustness_table))
    if save_features:
        outputs.append(("features.csv", X))
    for name, obj in outputs:
        path = os.path.join(outdir, name)
        obj.to_csv(path)
        written.append(path)

    if plot:
        from plotting import (plot_feat_weights, plot_refit_params, plot_regimes_and_cumret,
                              plot_weights, setup_font)
        if plot_font:
            setup_font(plot_font)
        name = result.model.upper()
        # only spell the extras out when they differ from the plain 0/1 strategy of the article
        cost_text = (f"cost={buy_bps:g}bp" if buy_bps == sell_bps
                     else f"cost={buy_bps:g}/{sell_bps:g}bp buy/sell")
        weight_text = ("" if (bear_weight, bull_weight) == (0., 1.)
                       else f", weight {bear_weight:.0%}-{bull_weight:.0%}")
        title = (f"{name} 0/1 strategy (lambda={jump_penalty:g}, window={result.window}, "
                 f"delay={delay}, {cost_text}{weight_text})")
        extra_curves = {"HMM 0/1 strategy": hmm_strategy["jm"]} if hmm_strategy is not None else None
        written.append(plot_regimes_and_cumret(strategy, os.path.join(outdir, "regimes_cumret.png"),
                                               title=title, label=f"{name} 0/1 strategy",
                                               extra_returns=extra_curves))
        written.append(plot_refit_params(result.params, result.feature_names,
                                         os.path.join(outdir, "refit_params.png"),
                                         title=f"Estimated centroids by regime from the rolling {name} fit"))
        written.append(plot_weights(strategy, os.path.join(outdir, "weights.png")))
        if result.feat_weights is not None:
            written.append(plot_feat_weights(result.feat_weights,
                                             os.path.join(outdir, "feat_weights.png"),
                                             pinned=result.pinned_features))

    if verbose:
        print("\n" + "=" * 72)
        print(f"온라인 레짐 구간: {summary['start']} ~ {summary['end']} ({summary['n_days']}거래일)")
        print(f"{result.model.upper():<3} bear 레짐 비중: {summary['bear_share']:.1%}, "
              f"레짐 전환 {summary['n_shifts']}회 (연 {summary['shifts_per_year']:.2f}회)")
        print(f"위험자산 비중: bull {bull_weight:.0%} / bear {bear_weight:.0%} "
              f"(현금 {min_cash:.0%}~{max_cash:.0%}), "
              f"거래비용: 매수 {buy_bps:g}bp / 매도 {sell_bps:g}bp")
        if result.feat_weights is not None:
            mean_weights = result.feat_weights.mean().sort_values(ascending=False)
            kept = (result.feat_weights > 0).mean()
            pinned_set = set(result.pinned_features)
            note = " (*: 고정한 피처)" if pinned_set else ""
            print(f"피처 가중(재추정 평균) / 선택된 비율{note}:")
            for feat, weight in mean_weights.items():
                mark = "*" if feat in pinned_set else " "
                print(f" {mark}{feat:<24} {weight:.3f}   {kept[feat]:.0%}")
        if hmm_summary is not None:
            print(f"HMM 고변동성 비중: {hmm_summary['bear_share']:.1%}, "
                  f"레짐 전환 {hmm_summary['n_shifts']}회 (연 {hmm_summary['shifts_per_year']:.2f}회)")
        print("-" * 72)
        print(format_performance_table(performance).to_string())
        if robustness_table is not None:
            print("-" * 72)
            print(f"거래 지연 로버스트니스 (지연 {', '.join(str(d) for d in delay_list)}일)")
            print(format_performance_table(robustness_table).to_string())
        print("=" * 72)
        print("저장된 파일:")
        for path in written:
            print(f"  {path}")

    return {"data": data, "X": X, "result": result, "strategy": strategy,
            "performance": performance, "summary": summary, "hmm_result": hmm_result,
            "hmm_strategy": hmm_strategy, "hmm_summary": hmm_summary,
            "robustness": robustness_table, "written": written}


def main(argv=None) -> int:
    """Parse the command line arguments and run the pipeline."""
    args = build_parser().parse_args(argv)
    if not args.quiet:
        warnings.simplefilter("always", UserWarning)
    run_pipeline(input_path=args.input, outdir=args.outdir, sheet=args.sheet,
                 date_col=args.date_col, close_col=args.close_col, rf_col=args.rf_col,
                 rf_unit=args.rf_unit, trading_days=args.trading_days,
                 start_date=args.start_date, end_date=args.end_date,
                 feature_set=args.feature_set, log_dd=args.log_dd, warmup=args.warmup,
                 extra_features=args.extra_feature, extra_file=args.extra_file,
                 extra_sheet=args.extra_sheet, extra_date_col=args.extra_date_col,
                 model=args.model, max_feats=args.max_feats, pin_features=args.pin_feature,
                 jump_penalty=args.jump_penalty, window=args.window, min_window=args.min_window,
                 n_components=args.n_components, clip_mul=args.clip_mul, n_init=args.n_init,
                 random_state=args.random_state, refit_start=args.refit_start,
                 hmm=args.hmm, hmm_window=args.hmm_window, hmm_refit_every=args.hmm_refit_every,
                 hmm_smooth_k=args.hmm_smooth_k, hmm_n_init=args.hmm_n_init,
                 hmm_covariance_type=args.hmm_covariance_type, hmm_simple_ret=args.hmm_simple_ret,
                 delay=args.delay, cost_bps=args.cost_bps, buy_cost_bps=args.buy_cost_bps,
                 sell_cost_bps=args.sell_cost_bps, min_cash=args.min_cash, max_cash=args.max_cash,
                 delays=args.delays,
                 robustness=not args.no_robustness,
                 plot=not args.no_plot, plot_font=args.plot_font,
                 save_features=args.save_features, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
