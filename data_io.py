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

RF_UNITS = ("annual_percent", "annual_decimal", "daily")


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


def load_market_data(filepath: str,
                     sheet=None,
                     date_col: Optional[str] = None,
                     close_col: Optional[str] = None,
                     rf_col: Optional[str] = None,
                     rf_unit: str = "annual_percent",
                     trading_days: int = TRADING_DAYS,
                     start_date=None,
                     end_date=None,
                     extra_cols=None) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        Indexed by `datetime.date`, with columns `close`, `rf` (daily risk-free rate),
        `ret` (simple total return), `excess_ret` (`ret` minus `rf`), and any `extra_cols`.
    """
    raw = load_raw_table(filepath, sheet=sheet)
    if raw.empty:
        raise ValueError(f"입력 파일이 비어 있습니다: {filepath}")

    col_date = _find_col(raw.columns, DATE_ALIASES, "날짜", date_col)
    col_close = _find_col(raw.columns, CLOSE_ALIASES, "종가", close_col)
    col_rf = _find_col(raw.columns, RF_ALIASES, "무위험금리", rf_col)

    dates = pd.to_datetime(raw[col_date], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"열 '{col_date}'을 날짜로 변환하지 못했습니다.")
    df = pd.DataFrame({
        "date": dates,
        "close": _to_numeric(raw[col_close], col_close),
        "rf_raw": _to_numeric(raw[col_rf], col_rf),
    })
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
    return df[["close", "rf_raw", "rf", "ret", "excess_ret"] + extra_names]


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
