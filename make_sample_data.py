#!/usr/bin/env python
"""
Build a sample input file in the format the pipeline expects.

The closing prices come from the Nasdaq-100 data already shipped in
`examples/nasdaq/data/NDX.csv`; the risk-free rate column is *synthetic* -- a smooth
random walk between 0.1% and 6.0% p.a. -- and exists only to illustrate the layout.
Replace it with an actual short rate before drawing any conclusion from the results.

    python make_sample_data.py --output sample_input.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
NDX_CSV = os.path.join(os.path.dirname(CURR_DIR), "nasdaq", "data", "NDX.csv")


def make_sample(source: str = NDX_CSV, seed: int = 0, with_extra: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame with the three columns of the expected input layout:
    `날짜`, `종가`, `무위험금리` (the latter annualized, in percent).

    With `with_extra`, a fourth column `변동성지수` is added to illustrate the custom
    variable options: the trailing 20-day realized volatility of the index, annualized and
    in percent. It only uses past returns, so it can be fed to the model as is.
    """
    df = pd.read_csv(source)[["date", "close"]].dropna()
    rng = np.random.default_rng(seed)
    # a slowly mean-reverting synthetic short rate, in annualized percent
    steps = rng.normal(0., .02, len(df))
    rf = pd.Series(steps, index=df.index).cumsum() * .5 + 3.
    rf = rf.clip(.1, 6.).round(3)
    out = pd.DataFrame({"날짜": df.date, "종가": df.close.round(4), "무위험금리": rf})
    if with_extra:
        realized_vol = df.close.pct_change().rolling(20).std() * np.sqrt(252) * 100.
        out["변동성지수"] = realized_vol.bfill().round(3)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="파이프라인 입력 형식의 샘플 파일 생성")
    parser.add_argument("--source", default=NDX_CSV, help="종가를 가져올 csv 파일")
    parser.add_argument("--output", default=os.path.join(CURR_DIR, "sample_input.csv"),
                        help="저장할 파일 경로 (.csv 또는 .xlsx)")
    parser.add_argument("--seed", type=int, default=0, help="합성 무위험금리 난수 시드")
    parser.add_argument("--with-extra", action="store_true",
                        help="커스텀 변수 예시로 '변동성지수'(20일 실현변동성) 열을 추가")
    args = parser.parse_args(argv)

    sample = make_sample(args.source, seed=args.seed, with_extra=args.with_extra)
    if args.output.lower().endswith((".xlsx", ".xls")):
        sample.to_excel(args.output, index=False)
    else:
        sample.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"샘플 파일 저장: {args.output} ({len(sample)}행, {sample.날짜.iloc[0]} ~ {sample.날짜.iloc[-1]})")
    print("주의: 무위험금리 열은 형식 예시를 위한 합성 데이터입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
