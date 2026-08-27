"""
Feature scaling for the rolling jump model.

The jump model clusters the days of the training window by their Euclidean distance in the
feature space, so a feature contributes to that distance in proportion to its own spread:
a variable measured in percent moves the cluster assignment far more than one measured in
basis points, whatever either of them says about the regime. Every feature is therefore put
on a common scale before the model sees it, and the scaler is refitted on each training
window so that no information from beyond the refit date leaks into it.

`--scaler` picks how that common scale is defined:

- ``"standard"`` (default): the z-score of the article's protocol -- subtract the mean of
  the training window, divide by its standard deviation.
- ``"robust"``: subtract the median, divide by the interquartile range rescaled to a
  Gaussian standard deviation (``IQR / 1.349``). Heavy-tailed or skewed features -- most of
  the volatility statistics, and macro variables with a few large moves -- inflate their own
  standard deviation, so the z-score squeezes the bulk of their distribution into a narrow
  band while a better-behaved feature keeps a wide one. Matching the interquartile spread
  matches the part of the distribution the clusters actually live in.
- ``"minmax"``: map the training window onto [0, 1] by its min and max. Every feature then
  spans exactly the same interval, but the typical distance between two days shrinks a long
  way below what it is under the other two, so lower `--jump-penalty` along with it.
- ``"none"``: leave the features in their own units. Only sensible when they already share
  a scale.

The winsorization that precedes the scaler (`jumpmodels.preprocess.DataClipperStd`, at
`--clip-mul` training-window standard deviations) is unchanged by this choice: it bounds the
outliers, and the scaler decides what the bounded values are divided by.

Because the scaling is undone before the cluster centroids are written to
``refit_params.csv`` (`FeatureScaler.inverse_transform`), the reported centroids are in the
original feature units under every scaler.
"""

import warnings

import numpy as np
import pandas as pd

SCALERS = ("standard", "robust", "minmax", "none")
DEFAULT_SCALER = "standard"

# Phi^-1(0.75) - Phi^-1(0.25): the interquartile range of a standard normal. Dividing the
# IQR by it makes "robust" agree with "standard" on a Gaussian feature, so that the same
# `--jump-penalty` keeps roughly the same meaning across the two scalers.
IQR_TO_STD = 1.3489795003921634


def resolve_scaler(name) -> str:
    """
    Validate the name of a feature scaler.

    Parameters
    ----------
    name : str or None
        The requested scaler, or None for the default.

    Returns
    -------
    str
        One of `SCALERS`.
    """
    if name is None:
        return DEFAULT_SCALER
    resolved = str(name).strip().lower()
    if resolved not in SCALERS:
        raise ValueError(f"지원하지 않는 스케일러입니다: {name}. 가능한 값: {SCALERS}")
    return resolved


def _column_names(X) -> list:
    """The column labels of `X`, as strings, or positional names when it is an ndarray."""
    if isinstance(X, pd.DataFrame):
        return [str(col) for col in X.columns]
    return [f"col{i}" for i in range(np.asarray(X).shape[1])]


class FeatureScaler:
    """
    Center and rescale a feature matrix, in the manner `kind` picks.

    The interface mirrors the one of `jumpmodels.preprocess.StandardScalerPD`: `fit` /
    `transform` / `fit_transform` keep the index and columns of a DataFrame, and are meant
    to be fitted on a training window and applied to the days that follow it.

    Parameters
    ----------
    kind : str, optional (default="standard")
        One of `SCALERS`; see the module docstring for what each of them centers and
        divides by.

    Attributes
    ----------
    center_ : np.ndarray
        The location subtracted from each column.

    scale_ : np.ndarray
        The spread each column is divided by. Columns whose spread is degenerate -- zero or
        non-finite, i.e. constant over the training window -- fall back to the standard
        deviation where the statistic allows it and to 1. otherwise, so that a constant
        column comes out as a constant instead of a division by zero.

    columns_ : list of str
        The columns seen at fit time, used to check that `transform` is given the same ones.
    """

    def __init__(self, kind: str = DEFAULT_SCALER) -> None:
        self.kind = resolve_scaler(kind)
        self.center_ = None
        self.scale_ = None
        self.columns_ = None

    def fit(self, X):
        """
        Compute the location and spread of every column of `X`.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The training window.

        Returns
        -------
        FeatureScaler
            The fitted scaler.
        """
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError(f"피처 행렬은 2차원이어야 합니다 (받은 차원: {values.ndim}).")
        if values.shape[0] == 0:
            raise ValueError("빈 학습창으로는 스케일러를 적합할 수 없습니다.")
        self.columns_ = _column_names(X)

        std = np.nanstd(values, axis=0)
        if self.kind == "standard":
            center, scale = np.nanmean(values, axis=0), std
        elif self.kind == "robust":
            q25, median, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
            center, scale = median, (q75 - q25) / IQR_TO_STD
            # a column with more than half its mass on one value has a zero IQR without
            # being constant; the standard deviation still describes its spread
            scale = np.where(_degenerate(scale), std, scale)
        elif self.kind == "minmax":
            center = np.nanmin(values, axis=0)
            scale = np.nanmax(values, axis=0) - center
        else:                                     # "none": leave the units alone
            center, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])

        bad = _degenerate(scale)
        if bad.any():
            warnings.warn(
                f"학습창에서 사실상 상수인 피처 {[self.columns_[i] for i in np.flatnonzero(bad)]}는 "
                f"스케일 1로 두고 중심만 이동합니다. 이 창에서는 레짐 정보를 담지 못합니다.")
            scale = np.where(bad, 1., scale)
        self.center_, self.scale_ = np.asarray(center, dtype=float), np.asarray(scale, dtype=float)
        return self

    def transform(self, X):
        """
        Apply the fitted scaling to `X`.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            A matrix with the columns seen at fit time. Values outside the range of the
            training window are *not* bounded -- winsorization is a separate step.

        Returns
        -------
        pd.DataFrame or np.ndarray
            The scaled matrix, as a DataFrame with the index and columns of `X` when `X`
            is one.
        """
        if self.scale_ is None:
            raise RuntimeError("스케일러를 먼저 적합해야 합니다 (fit / fit_transform).")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.scale_):
            raise ValueError(
                f"피처 개수가 적합 시점과 다릅니다: 적합 {len(self.scale_)}개, 변환 "
                f"{values.shape[1] if values.ndim == 2 else '?'}개.")
        if isinstance(X, pd.DataFrame) and _column_names(X) != self.columns_:
            raise ValueError(
                f"피처 열 구성이 적합 시점과 다릅니다: 적합 {self.columns_}, 변환 {_column_names(X)}.")
        scaled = (values - self.center_) / self.scale_
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(scaled, index=X.index, columns=X.columns)
        return scaled

    def fit_transform(self, X):
        """Fit on `X` and return the scaled `X`, as `fit` followed by `transform`."""
        return self.fit(X).transform(X)

    def inverse_transform(self, values) -> np.ndarray:
        """
        Map scaled values back into the original feature units.

        Parameters
        ----------
        values : array-like of shape (n_rows, n_features)
            Rows in the scaled space, typically cluster centroids.

        Returns
        -------
        np.ndarray
            The same rows in the units of the fitted feature matrix.
        """
        if self.scale_ is None:
            raise RuntimeError("스케일러를 먼저 적합해야 합니다 (fit / fit_transform).")
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(self.scale_):
            raise ValueError(
                f"피처 개수가 적합 시점과 다릅니다: 적합 {len(self.scale_)}개, 역변환 "
                f"{arr.shape[1] if arr.ndim == 2 else '?'}개.")
        return arr * self.scale_ + self.center_


def _degenerate(scale: np.ndarray) -> np.ndarray:
    """Mask of the entries of `scale` that cannot be divided by."""
    scale = np.asarray(scale, dtype=float)
    return ~np.isfinite(scale) | (scale <= 0.)
