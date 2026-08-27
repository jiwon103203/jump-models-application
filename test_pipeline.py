#!/usr/bin/env python
"""
Checks for the pieces of the pipeline that are easy to get subtly wrong: the causality of
the feature transforms, the trading delay, the transaction cost accounting, the robustness
table layout and the median filter of the HMM benchmark.

Run directly (`python test_pipeline.py`) or through `pytest test_pipeline.py`.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (build_weights, delay_robustness_table, resolve_cash_limits,
                      resolve_cost_bps, run_0_1_strategy)
from features import (EXAMPLE_HLS, EXTRA_DD_HLS, EXTRA_WINDOWS, FEATURE_SETS, apply_transform,
                      build_extra_features, build_features, compute_ewm_DD, feature_engineer,
                      feature_series, feature_series_name, feature_set_columns, parse_extra_spec,
                      resolve_pinned_features, resolve_removed_features)
from hmm_benchmark import smooth_states
from rolling import (DEFAULT_GRID_SIZE, init_model, refit_schedule, resolve_grid_size,
                     resolve_max_feats, run_rolling_jm, semiannual_anchors)
from scaling import (DEFAULT_SCALER, IQR_TO_STD, SCALERS, FeatureScaler, resolve_scaler)
from sparse_pin import PinnedSparseJumpModel, solve_lasso_pinned

DATES = pd.date_range("2020-01-01", periods=12, freq="D").date
LEVELS = pd.Series([10., 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 11], index=DATES, name="x")
REGIME = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], index=DATES)
RET = pd.Series(np.linspace(-.01, .01, 12), index=DATES)
RF = pd.Series(1e-4, index=DATES)


def test_transforms_are_causal():
    """No transform may let a later observation change an earlier value."""
    assert apply_transform(LEVELS, "raw").equals(LEVELS.astype(float))
    assert np.isnan(apply_transform(LEVELS, "diff", 1).iloc[0])
    assert apply_transform(LEVELS, "diff", 1).iloc[1] == 1.
    assert abs(apply_transform(LEVELS, "pct", 2).iloc[2] - (12 / 10 - 1)) < 1e-12
    assert abs(apply_transform(LEVELS, "logdiff", 1).iloc[1] - np.log(11 / 10)) < 1e-12
    full = apply_transform(LEVELS, "ewm", 3)
    prefix = apply_transform(LEVELS.iloc[:6], "ewm", 3)
    assert np.allclose(full.iloc[:6], prefix)


def test_extra_feature_set():
    """The "extra" set keeps the "example" returns and Sortino ratios and brings its own DD."""
    ret = pd.Series(np.linspace(-.02, .02, 400), index=pd.bdate_range("2020-01-01", periods=400).date)
    example = feature_engineer(ret, ver="example")
    extra = feature_engineer(ret, ver="extra")
    # the log downside deviations are the "example" set's own; everything else it holds is
    # carried over unchanged, in the same order
    shared = [col for col in example.columns if not col.startswith("DD-log")]
    assert list(extra.columns)[:len(shared)] == shared
    assert extra[shared].equals(example[shared])
    assert not [col for col in extra.columns if col.startswith("DD-log")]
    added = [col for col in extra.columns if col not in example.columns]
    assert len(added) == 29 and len(extra.columns) == 35, (len(added), len(extra.columns))
    # every statistic of the list the set was built from is present
    for name in ("ret-simple", "ret-log", "ret-abs", "ret-sq", "ret-cumlog"):
        assert name in added, name
    for window in EXTRA_WINDOWS:
        for stat in ("std", "var", "mad", "rms"):
            assert f"{stat}_{window:d}" in added, f"{stat}_{window:d}"
    for hl in EXAMPLE_HLS:
        assert f"vol-log_{hl:.0f}" in added and f"vol-chg_{hl:.0f}" in added, hl
    assert "vol-ratio_5-20" in added and "vol-ratio_20-60" in added
    # ... and so is the downside deviation family that closes the set, on the raw scale
    DD_columns = [f"DD_{hl:.0f}" for hl in EXTRA_DD_HLS]
    assert added[-4:] == DD_columns == ["DD_5", "DD_10", "DD_20", "DD_60"]
    for hl in EXTRA_DD_HLS:
        assert np.allclose(extra[f"DD_{hl:.0f}"], compute_ewm_DD(ret, hl)), hl
    # --log-dd logs that family in place and changes nothing else; on the log scale the 5-,
    # 20- and 60-day ones are exactly what the "example" set reports
    log_columns = [f"DD-log_{hl:.0f}" for hl in EXTRA_DD_HLS]
    logged = feature_engineer(ret, ver="extra", log_dd=True)
    assert list(logged.columns) == list(extra.columns)[:-len(EXTRA_DD_HLS)] + log_columns
    assert np.allclose(logged[log_columns], np.log(extra[DD_columns]))
    for hl in EXAMPLE_HLS:
        assert np.allclose(logged[f"DD-log_{hl:.0f}"], example[f"DD-log_{hl:.0f}"]), hl
    assert logged.drop(columns=log_columns).equals(extra.drop(columns=DD_columns))

    # the definitions that are easy to get wrong
    assert np.allclose(extra["ret-simple"], ret)
    assert np.allclose(extra["ret-cumlog"], np.log1p(ret).cumsum())
    assert np.allclose(extra["var_20"], extra["std_20"] ** 2, equal_nan=True)
    assert np.allclose(extra["vol-ratio_5-20"], extra["vol-log_5"] - extra["vol-log_20"])
    assert np.allclose(extra["vol-chg_20"], extra["vol-log_20"].diff(20), equal_nan=True)

    try:
        feature_engineer(ret, ver="없는세트")
        raise AssertionError("알 수 없는 피처 세트는 NotImplementedError를 내야 합니다.")
    except NotImplementedError:
        pass


def test_extra_feature_set_is_causal():
    """No "extra" feature may let a later observation change an earlier value."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=600).date
    ret = pd.Series(rng.normal(.0003, .011, 600), index=dates)
    full = feature_engineer(ret, ver="extra")
    for cut in (200, 450):
        assert full.iloc[:cut].equals(feature_engineer(ret.iloc[:cut], ver="extra")), cut


def test_feature_set_columns():
    """The advertised column names of each feature set are the ones actually produced."""
    ret = pd.Series(np.linspace(-.02, .02, 400), index=pd.bdate_range("2020-01-01", periods=400).date)
    for ver in FEATURE_SETS:
        assert feature_set_columns(ver) == list(feature_engineer(ret, ver=ver).columns), ver
    for ver in ("paper", "extra"):       # the two sets holding the 10-day downside deviation
        assert feature_set_columns(ver, log_dd=True) == \
               list(feature_engineer(ret, ver=ver, log_dd=True).columns), ver
    try:
        feature_set_columns("없는세트")
        raise AssertionError("알 수 없는 피처 세트는 NotImplementedError를 내야 합니다.")
    except NotImplementedError:
        pass


def _extra_ret(n=400):
    """A return series long enough for the 60-day windows of the "extra" set."""
    return pd.Series(np.linspace(-.02, .02, n), index=pd.bdate_range("2020-01-01", periods=n).date)


def test_remove_series():
    """A removal specification takes out a whole family of columns, and only that family."""
    ret, columns = _extra_ret(), feature_set_columns("extra")
    # a series is the column name up to the horizon the "_" separates; a column carrying no
    # horizon is a series of its own
    assert feature_series_name("var_20") == "var"
    assert feature_series_name("vol-ratio_5-20") == "vol-ratio"
    assert feature_series_name("ret-cumlog") == "ret-cumlog"
    assert feature_series(columns)[:2] == ["ret", "sortino"]
    assert {"var", "std", "vol-chg", "DD", "ret-simple"} <= set(feature_series(columns))

    assert resolve_removed_features(columns, None) == []
    assert resolve_removed_features(columns, ["var"]) == ["var_5", "var_20", "var_60"]
    assert resolve_removed_features(columns, ["DD"]) == ["DD_5", "DD_10", "DD_20", "DD_60"]
    # "ret" is the EWM return series, not every column whose name starts with those letters
    assert resolve_removed_features(columns, ["ret"]) == ["ret_5", "ret_20", "ret_60"]
    assert resolve_removed_features(columns, ["ret-simple"]) == ["ret-simple"]
    # individual columns, and several specifications at once: column order, no duplicates
    assert resolve_removed_features(columns, ["var_20", "ret-cumlog"]) == ["ret-cumlog", "var_20"]
    assert resolve_removed_features(columns, ["var_20", "var"]) == ["var_5", "var_20", "var_60"]

    # what `feature_set_columns` advertises is what `build_features` builds, removals included
    for specs in (["var"], ["var", "vol-chg", "ret-cumlog"], ["std", "var", "mad", "rms"]):
        advertised = feature_set_columns("extra", remove_series=specs)
        assert advertised == [col for col in columns
                              if col not in resolve_removed_features(columns, specs)], specs
        assert list(build_features(ret, ver="extra", warmup=100,
                                   remove_series=specs).columns) == advertised, specs
    # removed before the rows are, so a series is no longer paid for in warmup rows
    assert len(build_features(ret, ver="extra", warmup=0, remove_series=["vol-chg"])) > \
           len(build_features(ret, ver="extra", warmup=0))
    # ... and the columns that stay are untouched
    full = build_features(ret, ver="extra", warmup=100)
    assert full.drop(columns=["var_5", "var_20", "var_60"]).equals(
        build_features(ret, ver="extra", warmup=100, remove_series=["var"]))

    # a specification that matches nothing is a typo, not a silent no-op
    for bad in (["없는시리즈"], ["var_30"], ["Var"], [" "]):
        try:
            resolve_removed_features(columns, bad)
            raise AssertionError(f"{bad}는 KeyError나 ValueError를 내야 합니다.")
        except (KeyError, ValueError):
            pass
    try:
        build_features(ret, ver="paper", warmup=100, remove_series=["DD", "sortino"])
        raise AssertionError("피처를 모두 제거하면 ValueError를 내야 합니다.")
    except ValueError:
        pass


def test_remove_custom_variable():
    """Custom variables are removed by their series or by the specification they came in as."""
    ret = _extra_ret()
    raw = pd.DataFrame({"VIX": np.linspace(15., 30., len(ret))}, index=ret.index)
    extra = build_extra_features(raw, ["VIX:ewm:20", "VIX:diff:5"])
    build = lambda specs: list(build_features(ret, ver="paper", warmup=100, extra_features=extra,
                                              remove_series=specs).columns)
    paper = feature_set_columns("paper")
    assert build(None) == paper + ["VIX_ewm20", "VIX_diff5"]
    # the specification the variable was added with removes that transform alone ...
    assert build(["VIX:ewm:20"]) == paper + ["VIX_diff5"]
    assert build(["VIX_diff5"]) == paper + ["VIX_ewm20"]
    # ... while the variable's name is the series both transforms belong to
    assert build(["VIX"]) == paper


def test_pinning_a_removed_series():
    """Pinning knows what was removed: a group skips it quietly, a name is refused loudly."""
    columns = feature_set_columns("extra", remove_series=["var"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # the group asks for nothing the matrix lacks
        assert resolve_pinned_features(columns, ["extra"], ver="extra",
                                       remove_series=["var"]) == columns
    # a pin naming a removed column is a contradiction between two options, not an unknown name
    for pin in ("var_20", "var"):
        try:
            resolve_pinned_features(columns, [pin], ver="extra", remove_series=["var"])
            raise AssertionError(f"제거한 '{pin}'을 고정하면 KeyError를 내야 합니다.")
        except KeyError as err:
            assert "제거" in str(err), str(err)


def test_resolve_pinned_features():
    """Pin specifications name feature sets, custom variables or individual columns."""
    columns = feature_set_columns("example") + ["VIX_ewm20"]
    resolve = lambda specs, **kw: resolve_pinned_features(columns, specs, ver="example", **kw)
    assert resolve(None) == [] and resolve([]) == []
    assert resolve(["example"]) == feature_set_columns("example")
    assert resolve(["custom"]) == ["VIX_ewm20"]
    assert resolve(["all"]) == columns
    # a set the matrix only partly holds keeps the shared columns and warns about the rest
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve(["paper"]) == ["sortino_20", "sortino_60"]
        assert any("DD_10" in str(w.message) for w in caught), [str(w.message) for w in caught]
    # the extra set holds every column of the "paper" set, so there that group resolves in full
    # and each downside deviation can be pinned on its own, under the name --log-dd gives it
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # nothing missing, so nothing to warn
        for log_dd, DDs in ((False, ["DD_5", "DD_10", "DD_20", "DD_60"]),
                            (True, ["DD-log_5", "DD-log_10", "DD-log_20", "DD-log_60"])):
            columns_extra = feature_set_columns("extra", log_dd=log_dd)
            resolve_extra = lambda specs: resolve_pinned_features(columns_extra, specs,
                                                                  ver="extra", log_dd=log_dd)
            assert resolve_extra(["paper"]) == ["sortino_20", "sortino_60", DDs[1]], log_dd
            assert resolve_extra(DDs[::-1]) == DDs, log_dd
    # "example" is now the group the extra matrix only partly holds: the DD-log columns are
    # that set's own, so on the raw scale none of the three is there ...
    returns_and_sortinos = [col for col in feature_set_columns("example")
                            if not col.startswith("DD-log")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_pinned_features(feature_set_columns("extra"), ["example"],
                                       ver="extra") == returns_and_sortinos
        assert any("DD-log_5" in str(w.message) and "DD-log_60" in str(w.message)
                   for w in caught), [str(w.message) for w in caught]
    # ... while under --log-dd the set's own family covers all three halflives of the
    # "example" set, so the group resolves in full and there is nothing left to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_pinned_features(feature_set_columns("extra", log_dd=True), ["example"],
                                       ver="extra", log_dd=True) == \
               returns_and_sortinos + ["DD-log_5", "DD-log_20", "DD-log_60"]
    try:
        resolve_pinned_features(feature_set_columns("extra", log_dd=True), ["DD_20"],
                                ver="extra", log_dd=True)
        raise AssertionError("--log-dd 아래에는 DD_20 열이 없으므로 KeyError를 내야 합니다.")
    except KeyError:
        pass
    # individual columns, and the spec a custom variable was added with (its halflife default
    # included), which saves having to know the column name the spec was turned into
    assert resolve(["sortino_60", "VIX:ewm:20"]) == ["sortino_60", "VIX_ewm20"]
    assert resolve(["VIX:ewm"]) == ["VIX_ewm20"]
    # the result follows the column order and is free of duplicates, however the pins came in
    assert resolve(["VIX_ewm20", "ret_5"]) == ["ret_5", "VIX_ewm20"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolve(["paper", "sortino_20"]) == ["sortino_20", "sortino_60"]
    for bad in (["없는피처"], ["VIX"], ["VIX:nope"], [" "]):
        try:
            resolve(bad)
            raise AssertionError(f"{bad}는 KeyError나 ValueError를 내야 합니다.")
        except (KeyError, ValueError):
            pass


def test_extra_feature_specs():
    """Specifications are parsed into named columns, and bad ones are rejected."""
    built = build_extra_features(pd.DataFrame({"x": LEVELS}), ["x", "x:ewm:5", "x:diff:2"])
    assert list(built.columns) == ["x", "x_ewm5", "x_diff2"]
    try:
        build_extra_features(pd.DataFrame({"x": LEVELS}), ["없는열:ewm:5"])
        raise AssertionError("존재하지 않는 열은 KeyError를 내야 합니다.")
    except KeyError as err:
        assert "없는열" in str(err)
    for bad in ("x:nope", "x:ewm:abc", "x:ewm:5:6", ""):
        try:
            parse_extra_spec(bad)
            raise AssertionError(f"'{bad}'는 ValueError를 내야 합니다.")
        except ValueError:
            pass


def test_trading_delay():
    """The regime of day t drives the weight from day t + delay + 1 onwards."""
    invested = (REGIME == 0).astype(float)
    for delay in (1, 5):
        weights = build_weights(REGIME, delay=delay)
        shift = delay + 1
        assert (weights.iloc[shift:].to_numpy() == invested.iloc[:-shift].to_numpy()).all()
        assert (weights.iloc[:shift] == 1.).all()


def test_strategy_returns_and_costs():
    """Strategy returns follow the weights, and costs are charged on traded amounts only."""
    strategy = run_0_1_strategy(REGIME, RET, RF, delay=1, cost_bps=10.)
    expected = (strategy.weight * RET + (1. - strategy.weight) * RF
                - strategy.weight.diff().abs().fillna(0.) * 1e-3)
    assert np.allclose(strategy.jm, expected)
    assert np.allclose(strategy.traded, strategy.bought + strategy.sold)
    assert strategy.traded.iloc[0] == 0.


def test_cash_limits():
    """The cash limits bound the weight on the risky asset in both regimes."""
    assert resolve_cash_limits() == (1., 0.)                    # the pure 0/1 strategy
    assert resolve_cash_limits(min_cash=.05, max_cash=.6) == (.95, .4)
    for bad in ({"min_cash": -.1}, {"max_cash": 1.2}, {"min_cash": .6, "max_cash": .4}):
        try:
            resolve_cash_limits(**bad)
            raise AssertionError(f"{bad}는 ValueError를 내야 합니다.")
        except ValueError:
            pass

    weights = build_weights(REGIME, delay=1, min_cash=.1, max_cash=.7)
    invested = np.where(REGIME == 0, .9, .3)
    assert np.allclose(weights.iloc[2:], invested[:-2])
    assert (weights.iloc[:2] == .9).all()                       # start at the bull weight
    # a limited strategy is never fully in cash, and its returns stay between the extremes
    limited = run_0_1_strategy(REGIME, RET, RF, delay=1, cost_bps=0., min_cash=.1, max_cash=.7)
    assert limited.weight.between(.3, .9).all()
    assert np.allclose(limited.jm, limited.weight * RET + (1. - limited.weight) * RF)


def test_split_transaction_costs():
    """Buys and sells are charged at their own rate, and default to the symmetric cost."""
    assert resolve_cost_bps(10.) == (10., 10.)
    assert resolve_cost_bps(10., sell_cost_bps=25.) == (10., 25.)
    try:
        resolve_cost_bps(10., buy_cost_bps=-1.)
        raise AssertionError("음수 거래비용은 ValueError를 내야 합니다.")
    except ValueError:
        pass

    strategy = run_0_1_strategy(REGIME, RET, RF, delay=1, buy_cost_bps=10., sell_cost_bps=25.)
    change = strategy.weight.diff().fillna(0.)
    assert np.allclose(strategy.bought, change.clip(lower=0.))
    assert np.allclose(strategy.sold, (-change).clip(lower=0.))
    assert np.allclose(strategy.cost, strategy.bought * 1e-3 + strategy.sold * 25e-4)
    assert strategy.bought.iloc[0] == 0. and strategy.sold.iloc[0] == 0.
    gross = strategy.weight * RET + (1. - strategy.weight) * RF
    assert np.allclose(strategy.jm, gross - strategy.cost)
    # the asymmetric cost is dearer than the symmetric one it extends
    symmetric = run_0_1_strategy(REGIME, RET, RF, delay=1, cost_bps=10.)
    assert strategy.cost.sum() > symmetric.cost.sum()


def test_delay_robustness_table():
    """The table holds one buy-and-hold column and one column per (model, delay)."""
    table = delay_robustness_table({"JM": REGIME, "HMM": 1 - REGIME}, RET, RF, delays=(1, 5))
    assert list(table.columns) == [("B & H", ""), ("JM", 1), ("JM", 5), ("HMM", 1), ("HMM", 5)]
    assert list(table.index) == ["Return", "Sharpe", "Calmar"]


def test_median_filter():
    """The filter drops isolated flips but keeps a persistent regime shift."""
    raw = pd.Series([0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1], index=DATES)
    smoothed = smooth_states(raw, k=6)
    assert smoothed.iloc[3] == 0
    assert smoothed.iloc[-1] == 1
    assert smooth_states(raw, k=1).equals(raw.astype(int))


def test_refit_schedule():
    """Re-estimation happens on the first trading day on or after January 1st and July 1st."""
    index = pd.bdate_range("2010-01-01", "2013-12-31").date
    anchors = [date for date, _ in semiannual_anchors(index)]
    assert (pd.Timestamp("2011-01-03").date(), pd.Timestamp("2011-07-01").date()) == \
           (anchors[1], anchors[2]), anchors[:4]
    assert all(date.month in (1, 7) and date.day <= 4 for date in anchors)
    schedule = refit_schedule(index, window=3000, min_window=500)
    assert all(pos >= 500 and win == min(3000, pos) for _, pos, win in schedule)


def test_resolve_max_feats():
    """`max_feats` defaults to half the features, is capped at the feature count, and is >= 1."""
    assert resolve_max_feats(None, 10) == 5.
    assert resolve_max_feats(None, 3) == 2.       # never below two features
    assert resolve_max_feats(3., 10) == 3.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolve_max_feats(20., 10) == 10.  # capped at the feature count
    try:
        resolve_max_feats(0.5, 10)
        raise AssertionError("max_feats < 1은 ValueError를 내야 합니다.")
    except ValueError:
        pass
    # pinned features are kept whatever the budget, so the budget cannot be below their count
    assert resolve_max_feats(None, 4, n_pinned=3) == 3.
    assert resolve_max_feats(None, 10, n_pinned=3) == 5.       # the default still wins when larger
    assert resolve_max_feats(4., 10, n_pinned=3) == 4.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolve_max_feats(2., 10, n_pinned=3) == 3.


def test_resolve_grid_size():
    """The simplex mesh is validated, snapped to a divisor of one, and capped in size."""
    assert resolve_grid_size(.05, 2) == .05
    assert resolve_grid_size(.1, 3) == .1
    for bad in (0., -.1, 1.5):
        try:
            resolve_grid_size(bad, 2)
            raise AssertionError(f"grid_size={bad}는 ValueError를 내야 합니다.")
        except ValueError:
            pass
    # the mesh must divide one; 0.03 becomes 1/33, with a warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snapped = resolve_grid_size(.03, 2)
    assert int(1. / snapped) == 33 and any("grid_size" in str(w.message) for w in caught)
    # a mesh this fine over three regimes would blow the dynamic program up
    try:
        resolve_grid_size(.005, 3)
        raise AssertionError("격자점이 너무 많으면 ValueError를 내야 합니다.")
    except ValueError:
        pass


def test_init_model():
    """The factory returns the plain jump model or the sparse one, and rejects anything else."""
    from jumpmodels.jump import JumpModel
    from jumpmodels.sparse_jump import SparseJumpModel
    assert isinstance(init_model("jm", jump_penalty=50.), JumpModel)
    sjm = init_model("sjm", jump_penalty=50., n_init=4, max_feats=3., n_features=9)
    assert isinstance(sjm, SparseJumpModel)
    assert sjm.max_feats == 3. and sjm.n_init_jm == 4
    assert not isinstance(sjm, PinnedSparseJumpModel)        # nothing pinned: the plain model
    mask = np.array([True, True] + [False] * 7)
    pinned = init_model("sjm", jump_penalty=50., n_init=4, max_feats=3., n_features=9,
                        pin_mask=mask)
    assert isinstance(pinned, PinnedSparseJumpModel)
    assert (pinned.pin_mask == mask).all() and pinned.max_feats == 3.
    # a mask is meaningless for the plain model, which does not weigh features
    assert not hasattr(init_model("jm", pin_mask=mask), "pin_mask")
    # the continuous variant is the default, and `cont=False` restores the model of the article
    assert init_model("jm").cont and init_model("jm").grid_size == DEFAULT_GRID_SIZE
    assert init_model("sjm", n_features=9).cont
    assert not init_model("jm", cont=False).cont
    try:
        init_model("cjm")
        raise AssertionError("알 수 없는 모델은 ValueError를 내야 합니다.")
    except ValueError:
        pass


def _toy_features(n=900, seed=0):
    """A two-regime toy series with one informative feature pair and two pure noise columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n).date
    # alternate bull and bear blocks, so that every training window covers both regimes
    state = np.tile(np.repeat([0, 1], 90), n // 180 + 1)[:n]
    ret = pd.Series(rng.normal(np.where(state == 0, .001, -.001),
                               np.where(state == 0, .006, .02)), index=dates)
    X = pd.DataFrame({
        "ret_20": ret.ewm(halflife=20).mean(),
        "DD_10": np.sqrt(np.minimum(ret, 0.).pow(2).ewm(halflife=10).mean()),
        "noise_a": pd.Series(rng.normal(size=n), index=dates),
        "noise_b": pd.Series(rng.normal(size=n), index=dates),
    }).iloc[60:]
    return X, ret.reindex(X.index)


def test_sparse_jump_model_run():
    """The sparse model runs end to end and down-weights the noise features."""
    X, ret = _toy_features()
    result = run_rolling_jm(X, ret, model="sjm", jump_penalty=10., window=300, min_window=250,
                            n_init=2, verbose=False)
    assert result.model == "sjm"
    assert result.feat_weights is not None
    assert list(result.feat_weights.columns) == list(X.columns)
    assert len(result.feat_weights) == len(result.schedule)
    assert (result.feat_weights.to_numpy() >= 0).all()
    informative = result.feat_weights[["ret_20", "DD_10"]].mean().mean()
    noise = result.feat_weights[["noise_a", "noise_b"]].mean().mean()
    assert informative > .5 > noise, (informative, noise)      # the noise columns are dropped
    # centroids are reported for every feature, in the original units
    assert all(f"center_{col}" in result.params.columns for col in X.columns)
    assert result.params.stay_prob.between(0., 1.).all()
    assert result.params.freq.gt(0.).all()      # both regimes present in every window


def test_solve_lasso_pinned():
    """Pinning exempts a feature from the threshold without disturbing the free ones."""
    from jumpmodels.sparse_jump import solve_lasso
    scores, kappa = np.array([1., .9, .05, .02]), 1.2
    free = np.asarray(solve_lasso(scores, kappa))
    assert np.allclose(solve_lasso_pinned(scores, kappa, np.zeros(4, bool)), free)
    assert (free[2:] == 0.).all()               # the two weak features are dropped...

    pinned = solve_lasso_pinned(scores, kappa, np.array([False, False, True, True]))
    assert (pinned > 0.).all()                  # ... and kept once pinned
    assert abs(np.linalg.norm(pinned) - 1.) < 1e-12
    # the surviving free features keep their relative weights, and the budget only overshoots
    assert abs(pinned[0] / pinned[1] - free[0] / free[1]) < 1e-9
    assert pinned.sum() >= free.sum()
    # a pinned feature that separates the clusters not at all is floored, not dropped
    floored = solve_lasso_pinned(np.array([1., .9, 0.]), kappa, np.array([False, False, True]))
    assert floored[2] > 0.
    try:
        solve_lasso_pinned(scores, kappa, np.zeros(3, bool))
        raise AssertionError("길이가 다른 마스크는 ValueError를 내야 합니다.")
    except ValueError:
        pass


def test_pinned_sparse_jump_model_run():
    """A pinned feature carries a positive weight at every refit, dropped or not without it."""
    X, ret = _toy_features()
    kwargs = dict(model="sjm", jump_penalty=10., window=300, min_window=250, n_init=2,
                  max_feats=1.5, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = run_rolling_jm(X, ret, **kwargs)
        pinned = run_rolling_jm(X, ret, pin_feats=["noise_a"], **kwargs)
    assert free.pinned_features == [] and pinned.pinned_features == ["noise_a"]
    assert (free.feat_weights["noise_a"] == 0.).any()        # dropped when left to the selection
    assert (pinned.feat_weights["noise_a"] > 0.).all()       # never dropped once pinned
    # the informative features are still there, and the unpinned noise column is still droppable
    assert (pinned.feat_weights[["ret_20", "DD_10"]] > 0.).all().all()
    assert pinned.feat_weights["noise_a"].mean() > pinned.feat_weights["noise_b"].mean()
    assert set(pinned.regimes.regime.unique()) <= {0, 1}

    # an unknown feature is a typo, not something to pin silently
    try:
        run_rolling_jm(X, ret, pin_feats=["없는피처"], **kwargs)
        raise AssertionError("존재하지 않는 고정 피처는 KeyError를 내야 합니다.")
    except KeyError:
        pass


def test_pinning_is_ignored_by_the_plain_model():
    """The plain JM has no feature selection to override, so a pin is dropped with a warning."""
    X, ret = _toy_features()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_rolling_jm(X, ret, model="jm", jump_penalty=10., window=300, min_window=250,
                                n_init=2, pin_feats=["noise_a"], verbose=False)
    assert result.pinned_features == [] and result.feat_weights is None
    assert any("sjm" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_continuous_model_reports_probabilities():
    """The default continuous model fills `proba_*` with probabilities, not one-hot rows."""
    X, ret = _toy_features()
    kwargs = dict(jump_penalty=10., window=300, min_window=250, n_init=2, verbose=False)
    for model in ("jm", "sjm"):
        result = run_rolling_jm(X, ret, model=model, **kwargs)
        proba = result.regimes[["proba_0", "proba_1"]].to_numpy()
        assert result.cont and result.grid_size == DEFAULT_GRID_SIZE
        assert np.allclose(proba.sum(axis=1), 1.) and (proba >= 0.).all()
        # genuine probabilities: strictly between 0 and 1 somewhere, and on the simplex grid
        assert ((proba > 0.) & (proba < 1.)).any(), "원핫으로만 나왔습니다"
        assert np.allclose(proba / DEFAULT_GRID_SIZE, np.round(proba / DEFAULT_GRID_SIZE))
        # the reported regime stays the most likely state, so the 0/1 signal is unaffected
        assert (result.regimes.regime.to_numpy() == proba.argmax(axis=1)).all()
        assert set(result.regimes.regime.unique()) <= {0, 1}

    # a coarser mesh coarsens the reported probabilities
    coarse = run_rolling_jm(X, ret, model="jm", grid_size=.25, **kwargs)
    assert coarse.grid_size == .25
    assert set(np.round(coarse.regimes.proba_0.unique(), 10)) <= {0., .25, .5, .75, 1.}


def test_discrete_model_keeps_one_hot_probabilities():
    """`cont=False` restores the discrete model of the article, whose `proba_*` are 0/1."""
    X, ret = _toy_features()
    result = run_rolling_jm(X, ret, model="jm", jump_penalty=10., window=300, min_window=250,
                            n_init=2, cont=False, verbose=False)
    proba = result.regimes[["proba_0", "proba_1"]].to_numpy()
    assert not result.cont
    assert np.isin(proba, [0., 1.]).all() and np.allclose(proba.sum(axis=1), 1.)
    assert (result.regimes.regime.to_numpy() == proba.argmax(axis=1)).all()


def test_jump_model_run_has_no_feature_weights():
    """The plain model produces the same shape of output, without feature weights."""
    X, ret = _toy_features()
    result = run_rolling_jm(X, ret, model="jm", jump_penalty=10., window=300, min_window=250,
                            n_init=2, verbose=False)
    assert result.model == "jm" and result.feat_weights is None
    assert set(result.regimes.regime.unique()) <= {0, 1}


def _scaler_frame(seed=0):
    """Four columns on wildly different scales, one of them constant."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=400).date
    return pd.DataFrame({
        "small": rng.normal(0., .001, 400),          # a return-sized feature
        "large": rng.normal(300., 40., 400),         # an index-level-sized feature
        "skewed": rng.lognormal(0., 1.5, 400),       # a heavy right tail
        "flat": np.full(400, 7.),                    # no spread at all
    }, index=index)


def test_scaler_names():
    """Only the documented scalers are accepted, and None means the default."""
    assert resolve_scaler(None) == DEFAULT_SCALER == "standard"
    for name in SCALERS:
        assert resolve_scaler(name.upper()) == name
    for bad in ("zscore", "", "quantile"):
        try:
            resolve_scaler(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r}를 거부하지 않았습니다")


def test_standard_scaler_matches_the_library():
    """The default scaler reproduces `StandardScalerPD`, so the article's protocol is intact."""
    from jumpmodels.preprocess import StandardScalerPD
    X = _scaler_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # the constant column
        ours = FeatureScaler("standard").fit_transform(X)
    theirs = StandardScalerPD().fit_transform(X)
    assert np.allclose(ours, theirs)
    assert list(ours.columns) == list(X.columns) and ours.index.equals(X.index)


def test_scalers_put_features_on_one_scale():
    """Every scaler but "none" leaves the columns comparable; "none" leaves them as they are."""
    X = _scaler_frame()
    spreads = {}
    for name in SCALERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")         # the constant column
            scaled = FeatureScaler(name).fit_transform(X)
        spreads[name] = scaled.drop(columns="flat").std(ddof=0)
        # the constant column survives as a constant instead of dividing by zero
        assert np.isfinite(scaled["flat"]).all() and scaled["flat"].std(ddof=0) == 0.

    for name in ("standard", "robust", "minmax"):
        assert spreads[name].max() / spreads[name].min() < 5., (name, spreads[name])
    # untouched, the three columns are orders of magnitude apart
    assert spreads["none"].max() / spreads["none"].min() > 1e3


def test_robust_scaler_uses_the_median_and_iqr():
    """"robust" divides by the Gaussian-equivalent IQR, so a normal feature comes out unit-scale."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(5., 2., 20000)})
    scaler = FeatureScaler("robust").fit(X)
    q25, q75 = np.percentile(X.x, [25, 75])
    assert abs(scaler.center_[0] - np.median(X.x)) < 1e-12
    assert abs(scaler.scale_[0] - (q75 - q25) / IQR_TO_STD) < 1e-12
    assert abs(scaler.scale_[0] - 2.) < .05                    # ... which is the standard deviation

    # a column whose IQR is degenerate falls back to the standard deviation rather than to 1
    mostly_zero = pd.DataFrame({"dummy": np.where(np.arange(400) % 10 == 0, 1., 0.)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")                          # no "constant column" warning
        fallback = FeatureScaler("robust").fit(mostly_zero)
    assert abs(fallback.scale_[0] - mostly_zero.dummy.std(ddof=0)) < 1e-12


def test_scaling_is_fitted_on_the_training_window_only():
    """`transform` reuses the fitted statistics, so later rows cannot change earlier ones."""
    X = _scaler_frame().drop(columns="flat")
    train, later = X.iloc[:200], X.iloc[200:]
    for name in SCALERS:
        scaler = FeatureScaler(name).fit(train)
        head = scaler.transform(X).iloc[:200]
        assert np.allclose(head, scaler.transform(train))       # the tail changed nothing
        # and the fit really was out of sample: `later` is not renormalized to the same spread
        assert np.allclose(scaler.transform(later), (later - scaler.center_) / scaler.scale_)


def test_scaler_inverse_transform_round_trip():
    """Centroids are written out in the original units under every scaler."""
    X = _scaler_frame().drop(columns="flat")
    rows = np.asarray(X.iloc[:5])
    for name in SCALERS:
        scaler = FeatureScaler(name).fit(X)
        assert np.allclose(scaler.inverse_transform(scaler.transform(X).iloc[:5]), rows)


def test_scaler_rejects_a_different_feature_matrix():
    """A matrix that does not match the fitted columns is an error, not a silent misalignment."""
    X = _scaler_frame().drop(columns="flat")
    scaler = FeatureScaler("standard").fit(X)
    for bad in (X.drop(columns="small"), X.rename(columns={"small": "other"})):
        try:
            scaler.transform(bad)
        except ValueError:
            continue
        raise AssertionError("피처 구성이 달라도 통과했습니다")


def test_rolling_run_accepts_every_scaler():
    """The rolling fit runs under each scaler and records the one it used."""
    X, ret = _toy_features()
    kwargs = dict(model="jm", jump_penalty=10., window=300, min_window=250, n_init=2,
                  verbose=False)
    for name in SCALERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")         # "none" warns about the mixed units
            result = run_rolling_jm(X, ret, scaler=name, **kwargs)
        assert result.scaler == name
        assert set(result.regimes.regime.unique()) <= {0, 1}
        # a state missing from a training window is recorded as NaN; the rest are un-scaled
        # back into the feature units, so they stay inside the range of the feature
        centers = result.params[[f"center_{col}" for col in X.columns]].dropna()
        assert len(centers) > 0
        for col in X.columns:
            assert X[col].min() <= centers[f"center_{col}"].min()
            assert centers[f"center_{col}"].max() <= X[col].max()

    try:
        run_rolling_jm(X, ret, scaler="zscore", **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("알 수 없는 스케일러를 거부하지 않았습니다")


def test_default_scaler_leaves_the_run_unchanged():
    """Not passing `scaler` reproduces the run as it was before the option existed."""
    X, ret = _toy_features()
    kwargs = dict(model="sjm", jump_penalty=10., window=300, min_window=250, n_init=2,
                  verbose=False)
    before = run_rolling_jm(X, ret, **kwargs)
    after = run_rolling_jm(X, ret, scaler="standard", **kwargs)
    assert before.scaler == "standard"
    assert before.regimes.equals(after.regimes)
    assert np.allclose(before.feat_weights, after.feat_weights)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)}개 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
