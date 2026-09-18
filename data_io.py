"""
Helpers for loading a user-supplied price file (csv/Excel) into the return series
required by the jump model pipeline.

The expected input file holds three columns -- date, close price, and risk-free
rate -- under either Korean or English headers, e.g.

    날짜,종가,무위험금리
    1990-01-02,359.69,7.83
    1990-01-03,358.76,7.89

The risk-free rate is assumed to be quoted as an annualized percentage (7.83 for
7.83%) by default; see the `rf_unit` argument for the alternatives.

A benchmark price series can be carried along as well -- a fourth column of the same file
or a separate file -- so that the *relative* return, the asset return with the benchmark
return subtracted, sits next to the excess return. That is the series a sector-level
regime model is fitted on when the market-wide signal is to be taken out of it: a KOSPI
sector index fitted on its own return mostly rediscovers the regimes of the KOSPI, while
the same index fitted on its return minus the KOSPI return shows the sector's own.
See `relative_return` and `attach_benchmark`.
"""

import datetime
import warnings
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Column headers recognized for each role, normalized to lowercase w/o separators.
DATE_ALIASES = ("날짜", "일자", "기준일", "기준일자", "거래일", "date", "dates", "tradedate", "datadate")
CLOSE_ALIASES = ("종가", "수정종가", "종가지수", "가격", "지수", "close", "closeprice", "adjclose",
                 "price", "px", "pxlast", "index", "indexlevel")
RF_ALIASES = ("무위험금리", "무위험수익률", "무위험이자율", "무위험", "금리", "riskfree", "riskfreerate",
              "rf", "rfrate", "rate", "tbill", "tbillyield", "yield")
# Headers recognized for the benchmark (market index) price. They are only consulted in a
# file that holds the benchmark alone: inside the main file, where a header such as "종가"
# would match the asset's own column just as well, the benchmark column is named explicitly.
BENCH_ALIASES = ("벤치마크", "벤치마크종가", "벤치마크지수", "시장", "시장지수", "기준지수",
                 "코스피", "코스피지수", "코스피종가", "kospi", "kospi200", "benchmark",
                 "bench", "benchmarkclose", "bm", "market", "marketindex", "mktindex")

RF_UNITS = ("annual_percent", "annual_decimal", "daily")
# How the benchmark return is taken out of the asset return, see `relative_return`.
REL_METHODS = ("diff", "log", "ratio")
# Which return series the model is fitted on, see `resolve_signal_return`.
SIGNAL_RETURNS = ("auto", "excess", "relative")
# The columns `attach_benchmark` adds.
BENCH_COLUMNS = ("bench_close", "bench_ret", "rel_ret")


def normalize_header(name) -> str:
    """Lowercase a column header and strip separators, for alias matching."""
    return "".join(str(name).split()).replace("_", "").replace("-", "").replace(".", "").lower()


def _find_col(columns, aliases: tuple, role: str, explicit: Optional[str] = None) -> str:
    """
    Locate the column holding `role`, either by an explicit name or by alias matching.
    """
    if explicit is not None:
        if explicit in columns:
            return explicit
        # allow a case/spacing-insensitive match on the explicit name as well
        for col in columns:
            if normalize_header(col) == normalize_header(explicit):
                return col
        raise KeyError(f"열 '{explicit}' (역할: {role})을 찾을 수 없습니다. 파일의 열: {list(columns)}")
    for col in columns:
        if normalize_header(col) in aliases:
            return col
    # fall back to a substring match, e.g. "종가(원)" or "risk free rate (%)"
    for col in columns:
        norm = normalize_header(col)
        if any(alias in norm for alias in aliases):
            return col
    raise KeyError(
        f"'{role}' 역할의 열을 자동으로 찾지 못했습니다. 파일의 열: {list(columns)}. "
        f"--date-col/--close-col/--rf-col 로 직접 지정해 주세요."
    )


def _to_numeric(ser: pd.Series, name: str) -> pd.Series:
    """Coerce a column to float, tolerating thousand separators and percent signs."""
    if pd.api.types.is_numeric_dtype(ser):
        return ser.astype(float)
    cleaned = (ser.astype(str)
                  .str.replace(",", "", regex=False)
                  .str.replace("%", "", regex=False)
                  .str.strip()
                  .replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan}))
    out = pd.to_numeric(cleaned, errors="coerce")
    if out.isna().all():
        raise ValueError(f"열 '{name}'을 숫자로 변환하지 못했습니다.")
    return out


def load_raw_table(filepath: str, sheet=None) -> pd.DataFrame:
    """
    Read a csv/Excel file into a DataFrame, trying the encodings common in Korean data exports.
    """
    lower = str(filepath).lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(filepath, sheet_name=0 if sheet is None else sheet)
    last_err = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError as err:
            last_err = err
    raise last_err


def rf_to_daily(rf_raw: pd.Series, rf_unit: str = "annual_percent", trading_days: int = TRADING_DAYS) -> pd.Series:
    """
    Convert the raw risk-free rate column into a daily simple rate.

    Parameters
    ----------
    rf_raw : pd.Series
        The raw risk-free rate column.

    rf_unit : str, optional (default="annual_percent")
        One of "annual_percent" (7.83 means 7.83% p.a.), "annual_decimal" (0.0783 p.a.),
        or "daily" (already a daily rate, used as is).

    trading_days : int, optional (default=252)
        Trading days per year used to de-annualize.

    Returns
    -------
    pd.Series
        The daily risk-free rate.
    """
    if rf_unit not in RF_UNITS:
        raise ValueError(f"rf_unit 은 {RF_UNITS} 중 하나여야 합니다. 입력값: {rf_unit}")
    median_abs = float(np.nanmedian(np.abs(rf_raw)))
    if rf_unit == "annual_percent":
        if median_abs < 0.5 and median_abs > 0:
            warnings.warn(
                f"무위험금리의 중앙값이 {median_abs:.6f} 입니다. 연율 %(예: 4.25)가 아니라 "
                f"소수(0.0425)로 보입니다. --rf-unit annual_decimal 을 확인해 주세요.")
        return rf_raw / 100. / trading_days
    if rf_unit == "annual_decimal":
        if median_abs > 1.:
            warnings.warn(
                f"무위험금리의 중앙값이 {median_abs:.4f} 입니다. 소수(0.0425)가 아니라 "
                f"연율 %(4.25)로 보입니다. --rf-unit annual_percent 을 확인해 주세요.")
        return rf_raw / trading_days
    return rf_raw.copy()


def relative_return(ret: pd.Series, bench_ret: pd.Series, method: str = "diff") -> pd.Series:
    """
    Take the benchmark return out of an asset return, leaving the asset's own movement.

    A sector index moves with the market and with itself at the same time, and the market
    part is much the larger of the two: a jump model fitted on a KOSPI sector index mostly
    relabels the regimes of the KOSPI itself. Subtracting the benchmark return leaves the
    part that belongs to the sector, so that a regime found on the resulting series is a
    statement about the sector relative to the market -- "이 섹터가 시장을 이기는 국면인가" --
    rather than about the market.

    Parameters
    ----------
    ret : pd.Series
        The asset (sector) simple return.

    bench_ret : pd.Series
        The benchmark (market) simple return, measured over the same intervals as `ret`.

    method : str, optional (default="diff")
        How the two are combined:

        - ``"diff"``: ``ret - bench_ret``, the plain active return, i.e. the subtraction
          taken literally. It stays on the same scale as `ret`, which is why it is the default.
        - ``"log"``: ``log(1 + ret) - log(1 + bench_ret)``, the log active return. It adds
          up across time, so its cumulative sum is exactly the log relative performance.
        - ``"ratio"``: ``(1 + ret) / (1 + bench_ret) - 1``, the return of the ratio index
          (asset divided by benchmark), i.e. of a position long the asset and funded by the
          benchmark, rebalanced every day.

        At daily frequency the three differ by a second-order term only; they part company
        on large moves, where "ratio" is the one that compounds into the relative index.

    Returns
    -------
    pd.Series
        The relative return, indexed like `ret`.
    """
    if method not in REL_METHODS:
        raise ValueError(f"rel_method 는 {REL_METHODS} 중 하나여야 합니다. 입력값: {method}")
    ret = pd.Series(ret, dtype=float)
    bench_ret = pd.Series(bench_ret, dtype=float).reindex(ret.index)
    if method == "diff":
        return ret - bench_ret
    if method == "log":
        return np.log1p(ret.where(ret > -1.)) - np.log1p(bench_ret.where(bench_ret > -1.))
    return (1. + ret).div((1. + bench_ret).where(bench_ret > -1.)) - 1.


def attach_benchmark(data: pd.DataFrame, bench_close: pd.Series,
                     rel_method: str = "diff") -> pd.DataFrame:
    """
    Align a benchmark price series onto the trading days of `data` and derive the
    benchmark return and the benchmark-subtracted return of the asset.

    The benchmark is aligned by carrying its last observed close forward onto the asset's
    calendar, so a day the benchmark did not trade holds its previous close and no future
    value is ever carried backwards. Its return is then measured over exactly the intervals
    the asset return is measured over, which is what makes the subtraction meaningful.

    Parameters
    ----------
    data : pd.DataFrame
        The market data, holding at least `ret`, indexed by `datetime.date`.

    bench_close : pd.Series
        The benchmark close, indexed by `datetime.date`. It may follow its own calendar and
        may extend beyond `data` on either side.

    rel_method : str, optional (default="diff")
        How the benchmark return is taken out, see `relative_return`.

    Returns
    -------
    pd.DataFrame
        `data` with `bench_close`, `bench_ret` and `rel_ret` appended.
    """
    overlap = [col for col in BENCH_COLUMNS if col in data.columns]
    if overlap:
        raise ValueError(f"벤치마크 열 이름이 기존 열과 겹칩니다: {overlap}")
    if "ret" not in data.columns:
        raise KeyError("벤치마크를 붙이려면 'ret' 열이 있어야 합니다.")
    bench = pd.Series(bench_close, dtype=float).dropna().sort_index()
    if bench.empty:
        raise ValueError("벤치마크 가격 데이터가 비어 있습니다.")
    if (bench <= 0).any():
        raise ValueError("벤치마크 가격에 0 이하의 값이 있습니다. 데이터를 확인해 주세요.")

    aligned = bench.reindex(data.index.union(bench.index)).ffill().reindex(data.index)
    out = data.copy()
    out["bench_close"] = aligned
    out["bench_ret"] = aligned.pct_change()
    out["rel_ret"] = relative_return(out["ret"], out["bench_ret"], method=rel_method)

    n_missing = int(out["bench_close"].isna().sum())
    if n_missing:
        first = bench.index[0]
        warnings.warn(
            f"벤치마크가 덮지 못하는 날짜가 {n_missing}일 있습니다 (벤치마크 시작일 {first}). "
            f"해당 행은 상대수익률이 비어 있어 피처 생성 단계에서 제외됩니다.")
    same = (out["rel_ret"].abs() < 1e-12) | out["rel_ret"].isna()
    if bool(same.all()):
        warnings.warn(
            "상대수익률이 전 구간 0입니다. 종가 열과 벤치마크 열이 같은 데이터가 아닌지 "
            "확인해 주세요 (--close-col / --bench-col).")
    return out


def load_benchmark_series(filepath: str,
                          sheet=None,
                          date_col: Optional[str] = None,
                          close_col: Optional[str] = None) -> pd.Series:
    """
    Load a benchmark price file -- a date column and a close column -- into a Series.

    Use it when the benchmark lives in a file of its own; a benchmark column inside the
    main file is read by `load_market_data(bench_col=...)` instead.

    Parameters
    ----------
    filepath : str
        Path of the csv/Excel file holding the benchmark.

    sheet : str or int, optional
        Excel sheet name/index.

    date_col, close_col : str, optional
        Explicit column names; detected from the headers when omitted, the close from
        `BENCH_ALIASES` and `CLOSE_ALIASES` both.

    Returns
    -------
    pd.Series
        The benchmark close, named `bench_close` and indexed by `datetime.date`.
    """
    raw = load_raw_table(filepath, sheet=sheet)
    if raw.empty:
        raise ValueError(f"벤치마크 파일이 비어 있습니다: {filepath}")
    col_date = _find_col(raw.columns, DATE_ALIASES, "날짜", date_col)
    col_close = _find_col(raw.columns, BENCH_ALIASES + CLOSE_ALIASES, "벤치마크 종가", close_col)

    dates = pd.to_datetime(raw[col_date], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"벤치마크 파일의 열 '{col_date}'을 날짜로 변환하지 못했습니다.")
    out = pd.DataFrame({"date": dates, "bench_close": _to_numeric(raw[col_close], col_close)})
    n_bad = int(out.bench_close.isna().sum())
    if n_bad:
        warnings.warn(f"벤치마크 종가가 비어 있는 {n_bad}개 행을 제외합니다.")
    out = (out.dropna(subset=["date", "bench_close"])
              .sort_values("date")
              .drop_duplicates(subset="date", keep="last"))
    if out.empty:
        raise ValueError(f"벤치마크 파일에서 쓸 수 있는 행이 없습니다: {filepath}")
    return out.set_index(out.date.dt.date.rename(None))["bench_close"]


def resolve_signal_return(data: pd.DataFrame, signal_ret: str = "auto") -> str:
    """
    Name the column of `data` the features are to be built on.

    Parameters
    ----------
    data : pd.DataFrame
        The market data, carrying `rel_ret` when a benchmark was given.

    signal_ret : str, optional (default="auto")
        One of `SIGNAL_RETURNS`: "excess" for the excess return of the article, "relative"
        for the benchmark-subtracted return, or "auto", which takes the relative return
        whenever a benchmark is present -- taking the market signal out being the only
        reason to have supplied one -- and the excess return otherwise.

    Returns
    -------
    str
        "rel_ret" or "excess_ret".
    """
    if signal_ret not in SIGNAL_RETURNS:
        raise ValueError(f"signal_ret 은 {SIGNAL_RETURNS} 중 하나여야 합니다. 입력값: {signal_ret}")
    has_bench = "rel_ret" in data.columns
    if signal_ret == "relative":
        if not has_bench:
            raise ValueError(
                "signal_ret='relative' 은 벤치마크가 있어야 합니다. "
                "--relative-benchmark (별도 파일) 또는 --relative-benchmark-col (같은 파일)을 "
                "지정해 주세요.")
        return "rel_ret"
    if signal_ret == "excess":
        return "excess_ret"
    return "rel_ret" if has_bench else "excess_ret"


def load_market_data(filepath: str,
                     sheet=None,
                     date_col: Optional[str] = None,
                     close_col: Optional[str] = None,
                     rf_col: Optional[str] = None,
                     rf_unit: str = "annual_percent",
                     trading_days: int = TRADING_DAYS,
                     start_date=None,
                     end_date=None,
                     extra_cols=None,
                     bench_col: Optional[str] = None,
                     rel_method: str = "diff") -> pd.DataFrame:
    """
    Load a csv/Excel file of date, close price and risk-free rate into a clean daily panel.

    Parameters
    ----------
    filepath : str
        Path to the csv or Excel file.

    sheet : str or int, optional
        Excel sheet name/index. Ignored for csv input.

    date_col, close_col, rf_col : str, optional
        Explicit column names. When omitted the columns are detected from their headers.

    rf_unit : str, optional (default="annual_percent")
        Unit of the risk-free rate column, see `rf_to_daily`.

    trading_days : int, optional (default=252)
        Trading days per year used to de-annualize the risk-free rate.

    start_date, end_date : str or datetime.date, optional
        Optional date filters applied after the returns are computed.

    extra_cols : iterable of str, optional
        Additional columns of the same file to carry along, e.g. custom variables such as
        an implied volatility index or a credit spread. They are converted to floats and
        forward-filled, and kept under their original names.

    bench_col : str, optional
        The name of a benchmark (market index) close column of the same file, e.g. the KOSPI
        next to a KOSPI sector index. Unlike the three main columns it is never detected from
        its header -- a benchmark column named "종가_코스피" and the asset's own "종가" are
        equally good matches for the close role -- so name it explicitly. When the benchmark
        lives in a separate file, use `load_benchmark_series` and `attach_benchmark` instead.

    rel_method : str, optional (default="diff")
        How the benchmark return is taken out of the asset return, see `relative_return`.
        Ignored when `bench_col` is None.

    Returns
    -------
    pd.DataFrame
        Indexed by `datetime.date`, with columns `close`, `rf` (daily risk-free rate),
        `ret` (simple total return), `excess_ret` (`ret` minus `rf`), any `extra_cols`, and,
        when `bench_col` is given, `bench_close`, `bench_ret` and `rel_ret`.
    """
    raw = load_raw_table(filepath, sheet=sheet)
    if raw.empty:
        raise ValueError(f"입력 파일이 비어 있습니다: {filepath}")

    col_date = _find_col(raw.columns, DATE_ALIASES, "날짜", date_col)
    col_close = _find_col(raw.columns, CLOSE_ALIASES, "종가", close_col)
    col_rf = _find_col(raw.columns, RF_ALIASES, "무위험금리", rf_col)
    col_bench = (_find_col(raw.columns, (), "벤치마크 종가", explicit=bench_col)
                 if bench_col is not None else None)
    if col_bench is not None and col_bench == col_close:
        raise ValueError(
            f"벤치마크 열과 종가 열이 같은 열('{col_close}')로 인식되었습니다. "
            f"--close-col 과 --bench-col 을 서로 다른 열로 지정해 주세요.")

    dates = pd.to_datetime(raw[col_date], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"열 '{col_date}'을 날짜로 변환하지 못했습니다.")
    df = pd.DataFrame({
        "date": dates,
        "close": _to_numeric(raw[col_close], col_close),
        "rf_raw": _to_numeric(raw[col_rf], col_rf),
    })
    if col_bench is not None:
        # carried through the row filters under a temporary name, then turned into the
        # benchmark columns once the frame is indexed by date
        df["bench_raw"] = _to_numeric(raw[col_bench], col_bench)
    extra_names = []
    for name in (extra_cols or []):
        col = _find_col(raw.columns, (), f"커스텀 변수 '{name}'", explicit=name)
        df[name] = _to_numeric(raw[col], col)
        extra_names.append(name)

    n_bad_date = int(df.date.isna().sum())
    if n_bad_date:
        warnings.warn(f"날짜를 해석하지 못한 {n_bad_date}개 행을 제외합니다.")
    df = df.dropna(subset=["date"]).sort_values("date")

    n_dup = int(df.date.duplicated().sum())
    if n_dup:
        warnings.warn(f"중복된 날짜 {n_dup}개를 발견하여 마지막 행만 남깁니다.")
        df = df.drop_duplicates(subset="date", keep="last")

    n_bad_close = int(df.close.isna().sum())
    if n_bad_close:
        warnings.warn(f"종가가 비어 있는 {n_bad_close}개 행을 제외합니다.")
    df = df.dropna(subset=["close"])
    if (df.close <= 0).any():
        raise ValueError("종가에 0 이하의 값이 있습니다. 데이터를 확인해 주세요.")

    n_bad_rf = int(df.rf_raw.isna().sum())
    if n_bad_rf:
        warnings.warn(f"무위험금리가 비어 있는 {n_bad_rf}개 행을 직전 값으로 채웁니다.")
        df["rf_raw"] = df.rf_raw.ffill().bfill()

    # `datetime.date` index, matching the convention used across `jumpmodels`
    df = df.set_index(df.date.dt.date.rename(None)).drop(columns="date")
    df["rf"] = rf_to_daily(df.rf_raw, rf_unit=rf_unit, trading_days=trading_days)
    df["ret"] = df.close.pct_change()
    df["excess_ret"] = df.ret - df.rf
    if col_bench is not None:
        # attached before the first row goes: the benchmark return is NaN on that row for the
        # same reason `ret` is, so the two series start on the same day rather than a day apart
        bench_raw = df.pop("bench_raw")
        df = attach_benchmark(df, bench_raw, rel_method=rel_method)
    df = df.dropna(subset=["ret"])

    if start_date is not None:
        df = df.loc[pd.Timestamp(start_date).date():]
    if end_date is not None:
        df = df.loc[:pd.Timestamp(end_date).date()]
    if df.empty:
        raise ValueError("전처리 후 남은 데이터가 없습니다. 날짜 필터와 입력 파일을 확인해 주세요.")
    for name in extra_names:
        n_missing = int(df[name].isna().sum())
        if n_missing:
            warnings.warn(f"커스텀 변수 '{name}'의 결측 {n_missing}개를 직전 값으로 채웁니다.")
            df[name] = df[name].ffill()
    bench_names = list(BENCH_COLUMNS) if col_bench is not None else []
    return df[["close", "rf_raw", "rf", "ret", "excess_ret"] + bench_names + extra_names]


def load_extra_table(filepath: str,
                     sheet=None,
                     date_col: Optional[str] = None,
                     columns=None) -> pd.DataFrame:
    """
    Load custom variables from a second file, indexed by date.

    Parameters
    ----------
    filepath : str
        Path of the csv/Excel file holding a date column and one or more variables.

    sheet : str or int, optional
        Excel sheet name/index.

    date_col : str, optional
        The date column name; detected from the headers when omitted.

    columns : iterable of str, optional
        The variables to keep. All non-date columns are kept when omitted.

    Returns
    -------
    pd.DataFrame
        Indexed by `datetime.date`, holding the requested variables as floats.
    """
    raw = load_raw_table(filepath, sheet=sheet)
    if raw.empty:
        raise ValueError(f"커스텀 변수 파일이 비어 있습니다: {filepath}")
    col_date = _find_col(raw.columns, DATE_ALIASES, "날짜", date_col)
    dates = pd.to_datetime(raw[col_date], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"커스텀 변수 파일의 열 '{col_date}'을 날짜로 변환하지 못했습니다.")

    keep = list(columns) if columns else [col for col in raw.columns if col != col_date]
    out = pd.DataFrame({"date": dates})
    for name in keep:
        col = _find_col(raw.columns, (), f"커스텀 변수 '{name}'", explicit=name)
        out[name] = _to_numeric(raw[col], col)
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset="date", keep="last")
    return out.set_index(out.date.dt.date.rename(None)).drop(columns="date")


def join_extra_table(data: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """
    Align custom variables from a second file onto the trading days of the main data.

    Values are forward-filled, so a variable published at a lower frequency keeps its last
    observed value; no future value is ever carried backwards.

    Parameters
    ----------
    data : pd.DataFrame
        The output of `load_market_data`.

    extra : pd.DataFrame
        The output of `load_extra_table`.

    Returns
    -------
    pd.DataFrame
        `data` with the custom variables joined on its index.
    """
    overlap = [col for col in extra.columns if col in data.columns]
    if overlap:
        raise ValueError(f"커스텀 변수 파일의 열 이름이 기존 열과 겹칩니다: {overlap}")
    aligned = extra.reindex(data.index.union(extra.index)).ffill().reindex(data.index)
    n_missing = int(aligned.isna().any(axis=1).sum())
    if n_missing:
        warnings.warn(
            f"커스텀 변수 파일이 덮지 못하는 날짜가 {n_missing}일 있습니다 "
            f"(주로 데이터 시작 구간). 해당 행은 피처 생성 단계에서 제외됩니다.")
    return data.join(aligned)
