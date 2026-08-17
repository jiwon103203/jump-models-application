#!/usr/bin/env python
"""
End-to-end pipeline: a csv/Excel file of date, close price and risk-free rate in, and
regime signals plus a 0/1 strategy backtest out, following Shu, Yu and Mulvey (2024).

Example
-------
    python run_pipeline.py --input my_index.xlsx --outdir out/

Run `python run_pipeline.py --help` for the full list of options.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (DEFAULT_COST_BPS, format_performance_table, performance_table,
                      regime_summary, run_0_1_strategy)
from data_io import RF_UNITS, TRADING_DAYS, load_market_data
from features import FEATURE_SETS, build_features
from rolling import run_rolling_jm


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
                       help="paper: 논문 Table 2의 3개 피처 / example: 레포 예제의 9개 피처")
    group.add_argument("--log-dd", action="store_true",
                       help="paper 피처 세트에서 downside deviation을 로그 변환")
    group.add_argument("--warmup", type=int, default=252, help="EWM 워밍업으로 버릴 초기 행 수")

    group = parser.add_argument_group("모델 & 재추정")
    group.add_argument("--jump-penalty", type=float, default=50., help="점프 페널티 lambda")
    group.add_argument("--window", type=int, default=3000, help="학습창 길이 (거래일)")
    group.add_argument("--min-window", type=int, default=500,
                       help="허용하는 최소 학습창 길이. 데이터가 부족하면 이 길이까지 자동 축소")
    group.add_argument("--n-components", type=int, default=2, help="레짐 개수")
    group.add_argument("--clip-mul", type=float, default=3., help="윈저라이징 표준편차 배수")
    group.add_argument("--n-init", type=int, default=10, help="좌표하강 알고리즘 재시작 횟수")
    group.add_argument("--random-state", type=int, default=0, help="초기화 난수 시드")
    group.add_argument("--refit-start", default=None, help="이 날짜 이후의 재추정 시점만 사용")

    group = parser.add_argument_group("전략")
    group.add_argument("--delay", type=int, default=1,
                       help="거래 지연 일수. t일 신호는 t+delay+1일부터 적용")
    group.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                       help="편도 거래비용 (bp)")

    group = parser.add_argument_group("출력")
    group.add_argument("--outdir", default="out", help="결과 저장 폴더")
    group.add_argument("--no-plot", action="store_true", help="플롯 생성을 건너뜀")
    group.add_argument("--save-features", action="store_true", help="피처 행렬도 csv로 저장")
    group.add_argument("--quiet", action="store_true", help="진행 상황 출력을 최소화")
    return parser


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
                 jump_penalty: float = 50.,
                 window: int = 3000,
                 min_window: int = 500,
                 n_components: int = 2,
                 clip_mul: float = 3.,
                 n_init: int = 10,
                 random_state: int = 0,
                 refit_start=None,
                 delay: int = 1,
                 cost_bps: float = DEFAULT_COST_BPS,
                 plot: bool = True,
                 save_features: bool = False,
                 verbose: bool = True) -> dict:
    """
    Run the whole pipeline and write the results to `outdir`.

    The steps are: load and clean the price file, convert prices into excess returns,
    engineer the EWM downside deviation and Sortino features, re-estimate the jump model
    every six months over a rolling window while inferring the regimes online in between,
    and finally backtest the 0/1 strategy on the resulting signal.

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
        `strategy`, `performance`, `summary` and the list of written files.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) raw file -> daily returns and excess returns
    data = load_market_data(input_path, sheet=sheet, date_col=date_col, close_col=close_col,
                            rf_col=rf_col, rf_unit=rf_unit, trading_days=trading_days,
                            start_date=start_date, end_date=end_date)
    if verbose:
        print(f"데이터: {len(data)}거래일, {data.index[0]} ~ {data.index[-1]}")
        print(f"연율 무위험금리 평균: {data.rf.mean() * trading_days:.2%}, "
              f"자산 연율 수익률 평균: {data.ret.mean() * trading_days:.2%}")

    # 2) features from the excess return series
    X = build_features(data.excess_ret, ver=feature_set, warmup=warmup, log_dd=log_dd)
    if verbose:
        print(f"피처({feature_set}): {list(X.columns)} / {len(X)}행, {X.index[0]} ~ {X.index[-1]}")

    # 3) semiannual refits on a rolling window + online inference in between
    result = run_rolling_jm(X, data.excess_ret, jump_penalty=jump_penalty, window=window,
                            min_window=min_window, n_components=n_components, clip_mul=clip_mul,
                            n_init=n_init, random_state=random_state, start_date=refit_start,
                            verbose=verbose)

    # 4) 0/1 strategy backtest on the online inferred signal
    strategy = run_0_1_strategy(result.regimes.regime, data.ret, data.rf,
                                delay=delay, cost_bps=cost_bps,
                                bull_state=0)
    performance = performance_table(strategy, trading_days=trading_days)
    summary = regime_summary(result.regimes.regime, bear_state=n_components - 1)

    # 5) write everything out
    written = []
    regimes_out = result.regimes.join(data[["close", "ret", "rf", "excess_ret"]])
    regimes_out = regimes_out.join(strategy[["weight", "jm"]].rename(columns={"jm": "strategy_ret"}))
    for name, obj in (("regimes.csv", regimes_out),
                      ("refit_params.csv", result.params.set_index("refit_date")),
                      ("strategy.csv", strategy),
                      ("performance.csv", performance),
                      ("insample_last_window.csv", result.insample_last)):
        path = os.path.join(outdir, name)
        obj.to_csv(path)
        written.append(path)
    if save_features:
        path = os.path.join(outdir, "features.csv")
        X.to_csv(path)
        written.append(path)

    if plot:
        from plotting import plot_refit_params, plot_regimes_and_cumret, plot_weights
        title = (f"JM 0/1 strategy (lambda={jump_penalty:g}, window={result.window}, "
                 f"delay={delay}, cost={cost_bps:g}bp)")
        written.append(plot_regimes_and_cumret(strategy, os.path.join(outdir, "regimes_cumret.png"),
                                               title=title))
        written.append(plot_refit_params(result.params, result.feature_names,
                                         os.path.join(outdir, "refit_params.png")))
        written.append(plot_weights(strategy, os.path.join(outdir, "weights.png")))

    if verbose:
        print("\n" + "=" * 64)
        print(f"온라인 레짐 구간: {summary['start']} ~ {summary['end']} ({summary['n_days']}거래일)")
        print(f"bear 레짐 비중: {summary['bear_share']:.1%}, "
              f"레짐 전환 {summary['n_shifts']}회 (연 {summary['shifts_per_year']:.2f}회)")
        print("-" * 64)
        print(format_performance_table(performance).to_string())
        print("=" * 64)
        print("저장된 파일:")
        for path in written:
            print(f"  {path}")

    return {"data": data, "X": X, "result": result, "strategy": strategy,
            "performance": performance, "summary": summary, "written": written}


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
                 jump_penalty=args.jump_penalty, window=args.window, min_window=args.min_window,
                 n_components=args.n_components, clip_mul=args.clip_mul, n_init=args.n_init,
                 random_state=args.random_state, refit_start=args.refit_start,
                 delay=args.delay, cost_bps=args.cost_bps,
                 plot=not args.no_plot, save_features=args.save_features, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
