#!/usr/bin/env python
"""
Checks for the pieces of the pipeline that are easy to get subtly wrong: the causality of
the feature transforms, the trading delay, the transaction cost accounting, the robustness
table layout and the median filter of the HMM benchmark.

Run directly (`python test_pipeline.py`) or through `pytest test_pipeline.py`.
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import (build_weights, delay_robustness_table, performance_table,
                      resolve_backtest_basis, resolve_cash_limits, resolve_cost_bps,
                      run_0_1_strategy)
from data_io import (attach_benchmark, load_benchmark_series, load_market_data,
                     relative_return, resolve_signal_return)
from features import (EXAMPLE_HLS, EXTRA_DD_HLS, EXTRA_WINDOWS, FEATURE_SETS, apply_transform,
                      build_extra_features, build_features, compute_ewm_DD, feature_engineer,
                      feature_series, feature_series_name, feature_set_columns, parse_extra_spec,
                      resolve_pinned_features, resolve_removed_features)
from hmm_benchmark import smooth_states
from regime_episodes import (align_paths, compare_episode_paths, episode_metrics,
                             extract_episodes, length_scenarios, normalized_path,
                             rank_similar_episodes, resolve_target_episode, select_episodes)
from rolling import (DEFAULT_GRID_SIZE, init_model, refit_schedule, resolve_grid_size,
                     resolve_max_feats, run_rolling_jm, semiannual_anchors, state_losses)
from run_pipeline import load_data_with_extras, market_columns, prepare_inputs, run_inference
from sparse_pin import PinnedSparseJumpModel, solve_lasso_pinned
from weights import (CUSTOM_TYPE, GROUPINGS, NO_HORIZON, feature_group_map, feature_group_name,
                     feature_horizon, feature_type_name, group_feature_weights,
                     weight_group_summary)

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


############################################
## 벤치마크 차감 (섹터 지수)
############################################

BENCH_DATES = pd.bdate_range("2021-01-04", periods=8).date


def _sector_and_benchmark(own):
    """
    A benchmark and a sector index that is the benchmark times a known own-return path.

    `own` is what the sector does *beyond* the market, so `(1 + bench_ret) * (1 + own) - 1`
    is the sector return and "ratio" subtraction has to give `own` back exactly.
    """
    bench_ret = np.array([0., .03, -.05, .02, .04, -.06, .01, .02])
    bench_close = pd.Series(100. * np.cumprod(1. + bench_ret), index=BENCH_DATES)
    sector_ret = (1. + bench_ret) * (1. + np.asarray(own)) - 1.
    sector_close = pd.Series(50. * np.cumprod(1. + sector_ret), index=BENCH_DATES)
    return sector_close, bench_close


def test_relative_return_methods():
    """The three subtractions are the three formulas they claim to be, and nothing else."""
    ret = pd.Series([.02, -.03, .005], index=BENCH_DATES[:3])
    bench = pd.Series([.01, -.04, .005], index=BENCH_DATES[:3])

    assert np.allclose(relative_return(ret, bench, "diff"), ret - bench)
    assert np.allclose(relative_return(ret, bench, "log"), np.log1p(ret) - np.log1p(bench))
    assert np.allclose(relative_return(ret, bench, "ratio"), (1. + ret) / (1. + bench) - 1.)
    # a day the asset matched the benchmark is a zero under every one of them
    assert all(abs(float(relative_return(ret, bench, method).iloc[2])) < 1e-15
               for method in ("diff", "log", "ratio"))
    try:
        relative_return(ret, bench, "subtract")
    except ValueError:
        pass
    else:
        raise AssertionError("알 수 없는 rel_method 를 거부해야 합니다")


def test_attach_benchmark_leaves_only_the_sector_signal():
    """Subtracting the benchmark recovers exactly what the sector did on its own."""
    own = np.array([0., .01, .01, -.02, 0., -.02, 0., .015])
    sector_close, bench_close = _sector_and_benchmark(own)
    data = pd.DataFrame({"close": sector_close, "ret": sector_close.pct_change()})

    out = attach_benchmark(data, bench_close, rel_method="ratio")
    assert list(out.columns) == ["close", "ret", "bench_close", "bench_ret", "rel_ret"]
    assert np.isnan(out.rel_ret.iloc[0])                        # no return on the first row
    # the market's own moves -- including its -6% day -- leave no trace in the relative return
    assert np.allclose(out.rel_ret.iloc[1:], own[1:])
    # "diff" is the same statement to first order, and the exact subtraction it promises
    diff = attach_benchmark(data, bench_close, rel_method="diff")
    assert np.allclose(diff.rel_ret.iloc[1:], (diff.ret - diff.bench_ret).iloc[1:])
    assert np.abs(diff.rel_ret.iloc[1:] - own[1:]).max() < 5e-3


def test_attach_benchmark_aligns_without_looking_ahead():
    """A benchmark on its own calendar is carried forward, never backwards."""
    own = np.zeros(8)
    sector_close, bench_close = _sector_and_benchmark(own)
    data = pd.DataFrame({"close": sector_close, "ret": sector_close.pct_change()})
    # the benchmark did not trade on the fourth day, and starts a day late
    gapped = bench_close.drop(index=[BENCH_DATES[0], BENCH_DATES[3]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = attach_benchmark(data, gapped, rel_method="diff")
    assert np.isnan(out.bench_close.iloc[0])                    # nothing is filled backwards
    assert out.bench_close.iloc[3] == bench_close.iloc[2]       # the holiday holds its last close
    assert abs(float(out.bench_ret.iloc[3])) < 1e-15            # so its return that day is zero
    # and the whole thing is causal: the tail cannot change a value computed earlier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prefix = attach_benchmark(data.iloc[:5], gapped.loc[:BENCH_DATES[4]], rel_method="diff")
    assert np.allclose(out.rel_ret.iloc[1:5], prefix.rel_ret.iloc[1:], equal_nan=True)


def test_attach_benchmark_rejects_a_benchmark_that_is_the_asset():
    """The same column twice means a relative return of zero; say so rather than fit it."""
    own = np.zeros(8)
    sector_close, _ = _sector_and_benchmark(own)
    data = pd.DataFrame({"close": sector_close, "ret": sector_close.pct_change()})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = attach_benchmark(data, sector_close, rel_method="diff")
    assert np.allclose(out.rel_ret.iloc[1:], 0.)
    assert any("상대수익률" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_load_market_data_reads_a_benchmark_column():
    """A benchmark column of the main file becomes `bench_ret` and `rel_ret`."""
    own = np.array([0., .01, -.02, .03, 0., -.01, .02, 0.])
    sector_close, bench_close = _sector_and_benchmark(own)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sector.csv")
        pd.DataFrame({"날짜": BENCH_DATES, "종가": sector_close.to_numpy(),
                      "코스피": bench_close.to_numpy(), "무위험금리": 3.}).to_csv(path, index=False)

        plain = load_market_data(path)
        data = load_market_data(path, bench_col="코스피", rel_method="ratio")
        assert "rel_ret" not in plain.columns             # nothing changes without a benchmark
        assert list(data.columns) == ["close", "rf_raw", "rf", "ret", "excess_ret",
                                      "bench_close", "bench_ret", "rel_ret"]
        # the benchmark is attached before the first (return-less) row is dropped, so the
        # relative return starts on the same day the asset return does
        assert data.index[0] == plain.index[0] and not data.rel_ret.isna().any()
        assert np.allclose(data.rel_ret, own[1:])
        assert np.allclose(data.close, plain.close) and np.allclose(data.ret, plain.ret)

        # the asset cannot be its own benchmark by accident
        try:
            load_market_data(path, close_col="종가", bench_col="종가")
        except ValueError:
            pass
        else:
            raise AssertionError("종가 열과 벤치마크 열이 같으면 거부해야 합니다")


def test_benchmark_from_a_file_matches_a_benchmark_column():
    """Both routes -- a column of the main file, a file of its own -- give the same series."""
    own = np.array([0., .01, -.02, .03, 0., -.01, .02, 0.])
    sector_close, bench_close = _sector_and_benchmark(own)
    with tempfile.TemporaryDirectory() as folder:
        main = os.path.join(folder, "sector.csv")
        side = os.path.join(folder, "kospi.csv")
        pd.DataFrame({"날짜": BENCH_DATES, "종가": sector_close.to_numpy(),
                      "코스피": bench_close.to_numpy(), "무위험금리": 3.}).to_csv(main, index=False)
        pd.DataFrame({"date": BENCH_DATES,
                      "close": bench_close.to_numpy()}).to_csv(side, index=False)

        assert np.allclose(load_benchmark_series(side), bench_close)
        in_file, _ = load_data_with_extras(main, bench_col="코스피", rel_method="diff")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # the first day has no benchmark return
            from_file, _ = load_data_with_extras(main, bench_file=side, rel_method="diff")
        # the separate file cannot see the day before the data starts, so its first return is
        # missing; everywhere else the two are the same number
        assert np.isnan(from_file.bench_ret.iloc[0])
        assert np.allclose(in_file.rel_ret.iloc[1:], from_file.rel_ret.iloc[1:])

        try:
            load_data_with_extras(main, bench_col="코스피", bench_file=side)
        except ValueError:
            pass
        else:
            raise AssertionError("--bench-col 과 --bench-file 을 함께 주면 거부해야 합니다")


def test_resolve_signal_return():
    """"auto" takes the benchmark out whenever there is one; the other two are explicit."""
    plain = pd.DataFrame({"excess_ret": [.01, .02]})
    with_bench = pd.DataFrame({"excess_ret": [.01, .02], "rel_ret": [.003, -.001]})

    assert resolve_signal_return(plain) == "excess_ret"
    assert resolve_signal_return(with_bench) == "rel_ret"
    assert resolve_signal_return(with_bench, "excess") == "excess_ret"
    assert resolve_signal_return(with_bench, "relative") == "rel_ret"
    for bad, data in (("relative", plain), ("resid", with_bench)):
        try:
            resolve_signal_return(data, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"signal_ret={bad!r} 은 거부해야 합니다")


def test_resolve_backtest_basis():
    """The two bases hand the strategy the two pairs of series they promise."""
    own = np.array([0., .01, -.02, .03, 0., -.01, .02, 0.])
    sector_close, bench_close = _sector_and_benchmark(own)
    plain = pd.DataFrame({"close": sector_close, "ret": sector_close.pct_change(), "rf": .0001})
    data = attach_benchmark(plain, bench_close, rel_method="diff")

    ret, rf, name = resolve_backtest_basis(data, "asset")
    assert np.allclose(ret, data.ret, equal_nan=True) and np.allclose(rf, data.rf)
    assert name == "자산"
    # the active position earns the relative return, and being out of it earns nothing extra
    rel, rel_rf, rel_name = resolve_backtest_basis(data, "relative")
    assert np.allclose(rel, data.rel_ret, equal_nan=True) and np.allclose(rel_rf, 0.)
    assert rel_rf.index.equals(data.index) and rel_name == "상대"

    # a benchmark-free run has no relative position to score, and says so rather than guess
    try:
        resolve_backtest_basis(plain, "relative")
    except ValueError:
        pass
    else:
        raise AssertionError("벤치마크 없이 backtest_ret='relative' 는 거부해야 합니다")
    try:
        resolve_backtest_basis(data, "active")
    except ValueError:
        pass
    else:
        raise AssertionError("알 수 없는 backtest_ret 은 거부해야 합니다")


def test_the_relative_backtest_scores_the_active_position():
    """On the relative basis every number is an over-the-benchmark number."""
    own = np.array([0., .01, -.02, .03, 0., -.01, .02, 0.])
    sector_close, bench_close = _sector_and_benchmark(own)
    plain = pd.DataFrame({"close": sector_close, "ret": sector_close.pct_change(), "rf": .0001})
    data = attach_benchmark(plain, bench_close, rel_method="ratio").iloc[1:]   # drop the NaN day
    regimes = pd.Series([0, 0, 0, 1, 1, 0, 0], index=data.index, name="regime")

    asset = run_0_1_strategy(regimes, *resolve_backtest_basis(data, "asset")[:2],
                             delay=1, cost_bps=10.)
    rel = run_0_1_strategy(regimes, *resolve_backtest_basis(data, "relative")[:2],
                           delay=1, cost_bps=10.)

    # same signal, same trades -- only what the position earns differs
    assert np.allclose(asset.weight, rel.weight) and np.allclose(asset.traded, rel.traded)
    assert np.allclose(asset.bh, data.ret) and np.allclose(rel.bh, data.rel_ret)
    assert np.allclose(rel.rf, 0.)
    assert np.allclose(rel.jm, rel.weight * data.rel_ret - rel.cost)
    # and the sector's own return is no longer in it: a day the sector merely followed the
    # market is a zero, however far the market itself moved
    assert abs(float(rel.bh.iloc[3])) < 1e-12 and abs(float(data.ret.iloc[3])) > 1e-2

    # out of the active position is flat against the benchmark, not in cash
    all_bear = run_0_1_strategy(pd.Series(1, index=data.index),
                                *resolve_backtest_basis(data, "relative")[:2],
                                delay=1, cost_bps=10.)
    assert np.allclose(all_bear.weight.iloc[2:], 0.) and np.allclose(all_bear.jm.iloc[3:], 0.)

    # with no risk-free leg the Sharpe ratio of the table is an information ratio
    table = performance_table(rel, label="JM 0/1 (상대)")
    ir = float(rel.jm.mean() / rel.jm.std(ddof=1) * np.sqrt(252))
    assert abs(table.loc["Sharpe", "JM 0/1 (상대)"] - ir) < 1e-9


############################################
## 변수 유형별 가중치
############################################

def test_feature_grouping_names():
    """Every column of a feature set lands in a type, and only a custom variable in "custom"."""
    assert feature_type_name("sortino_20") == "return"
    assert feature_type_name("std_5") == feature_type_name("ret-abs") == "realized-vol"
    assert feature_type_name("vol-ratio_5-20") == "log-vol"
    assert feature_type_name("DD_10") == feature_type_name("DD-log_5") == "downside"
    assert feature_type_name("VIX_ewm20") == CUSTOM_TYPE
    # the whole "extra" set is covered by the four named types, custom left for the user
    columns = feature_set_columns("extra")
    assert CUSTOM_TYPE not in {feature_type_name(col) for col in columns}

    assert feature_horizon("std_20") == "20"
    assert feature_horizon("vol-ratio_5-20") == "5-20"
    # a name with no horizon, and a custom variable whose suffix is a transform, have none
    assert feature_horizon("ret-cumlog") == feature_horizon("VIX_ewm20") == NO_HORIZON

    assert feature_group_name("std_5", "type-horizon") == "realized-vol_5"
    assert feature_group_name("std_5", "series") == "std"
    assert feature_group_name("std_5", "horizon") == "5"


def test_feature_group_map_covers_every_column_once():
    """Each grouping partitions the columns: no column is lost and none is counted twice."""
    columns = feature_set_columns("extra") + ["VIX_ewm20"]
    for grouping in GROUPINGS:
        mapping = feature_group_map(columns, grouping)
        grouped = [col for cols in mapping.values() for col in cols]
        assert sorted(grouped) == sorted(columns), grouping
        assert all(cols for cols in mapping.values()), grouping     # no empty group is kept
    # the "type" grouping reports its groups in a fixed order, whatever the column order
    forward = list(feature_group_map(columns, "type"))
    assert forward == list(feature_group_map(columns[::-1], "type"))


def test_group_feature_weights_shares():
    """The shares of a refit add up to one, and `total` keeps the L1 norm they came from."""
    columns = ["ret_20", "sortino_20", "std_5", "vol-log_20", "DD_10"]
    dates = pd.to_datetime(["2020-01-02", "2020-07-01"]).date
    feat_weights = pd.DataFrame([[.5, .5, 1., 2., 1.], [0., 0., 1., 1., 0.]],
                                index=dates, columns=columns)
    shares = group_feature_weights(feat_weights, grouping="type")
    groups = [col for col in shares.columns if col != "total"]
    assert np.allclose(shares[groups].sum(axis=1), 1.)
    assert np.allclose(shares["total"], feat_weights.sum(axis=1))
    assert abs(shares.loc[dates[0], "return"] - 1. / 5.) < 1e-12        # (.5 + .5) / 5
    assert abs(shares.loc[dates[1], "log-vol"] - .5) < 1e-12            # 1 / 2, the rest dropped
    assert shares.loc[dates[1], "return"] == 0.

    summary = weight_group_summary(shares, feat_weights)
    assert summary.index[0] == "log-vol"                               # the largest mean share
    assert abs(summary.loc["log-vol", "mean_share"] - .45) < 1e-12     # (2/5 + 1/2) / 2
    assert summary.loc["return", "n_features"] == 2
    assert summary.loc["return", "mean_kept"] == 1.                    # 2 kept, then 0


def test_group_feature_weights_survives_an_all_zero_refit():
    """A refit whose weights are all zero gives NaN shares rather than a division by zero."""
    feat_weights = pd.DataFrame([[1., 1.], [0., 0.]], columns=["std_5", "DD_10"],
                                index=pd.to_datetime(["2020-01-02", "2020-07-01"]).date)
    shares = group_feature_weights(feat_weights)
    assert shares.iloc[0].notna().all()
    assert shares.iloc[1][["realized-vol", "downside"]].isna().all()
    assert shares.iloc[1]["total"] == 0.


############################################
## 유사 국면 탐색
############################################

def _episode_frame():
    """A regime path with three bear episodes, the last one still running at the end."""
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n).date
    regime = pd.Series(0, index=dates)
    regime.iloc[5:10] = 1          # 5 days, a fall
    regime.iloc[20:23] = 1         # 3 days, a rise -> a false signal
    regime.iloc[37:] = 1           # 3 days, still running at the end of the sample
    ret = pd.Series(.001, index=dates)
    ret.iloc[5:10] = -.02
    ret.iloc[20:23] = .01
    ret.iloc[37:] = -.005
    return regime, ret, dates


def test_extract_episodes():
    """Episodes are the maximal runs of the state, measured on the benchmark return."""
    regime, ret, dates = _episode_frame()
    episodes = extract_episodes(regime, ret)
    assert list(episodes["length"]) == [5, 3, 3]
    assert list(episodes.index) == [dates[5], dates[20], dates[37]]
    assert list(episodes["end"]) == [dates[9], dates[22], dates[39]]
    assert list(episodes["ongoing"]) == [False, False, True]
    # a rising episode is flagged as a false signal, a falling one is not
    assert list(episodes["false_signal"]) == [False, True, False]
    assert abs(episodes["period_ret"].iloc[0] - ((1 - .02) ** 5 - 1)) < 1e-12
    # the drawdown is measured from the level in force at entry, so a straight fall is
    # exactly the episode return
    assert abs(episodes["mdd"].iloc[0] - episodes["period_ret"].iloc[0]) < 1e-12
    assert episodes["mdd"].iloc[1] == 0.               # a straight rise draws down nothing


def test_extract_episodes_handles_the_edges():
    """An episode may start on the first day of the sample and end on the last."""
    dates = pd.bdate_range("2020-01-01", periods=6).date
    regime = pd.Series([1, 1, 0, 0, 1, 1], index=dates)
    episodes = extract_episodes(regime, pd.Series(-.01, index=dates))
    assert list(episodes["length"]) == [2, 2]
    assert list(episodes.index) == [dates[0], dates[4]]
    assert list(episodes["ongoing"]) == [False, True]
    # a state that never occurs gives an empty table rather than an error
    assert extract_episodes(pd.Series(0, index=dates), pd.Series(0., index=dates)).empty


def test_select_episodes_filters_are_opt_in():
    """No filter narrows the table unless it is asked for, except the running episode."""
    regime, ret, _ = _episode_frame()
    episodes = extract_episodes(regime, ret)
    assert len(select_episodes(episodes, drop_ongoing=False)) == 3
    assert len(select_episodes(episodes)) == 2                     # the running one goes
    assert len(select_episodes(episodes, drop_false_signal=True)) == 1
    assert len(select_episodes(episodes, min_length=4)) == 1
    assert len(select_episodes(episodes, drop_ongoing=False,
                               exclude=[episodes.index[0]])) == 2
    # `before` is what keeps a comparison backward-looking: only what had already ended
    assert len(select_episodes(episodes, drop_ongoing=False, before=episodes.index[2])) == 2
    assert len(select_episodes(episodes, drop_ongoing=False, before=episodes.index[0])) == 0


def test_episode_metrics_split_the_length_into_its_two_causes():
    """Distance days and penalty days partition the episode, and follow the loss columns."""
    regime, ret, dates = _episode_frame()
    close = pd.Series(100. * (1. + ret).cumprod().to_numpy(), index=dates)
    data = pd.DataFrame({"close": close, "ret": ret}, index=dates)
    regimes = pd.DataFrame({"regime": regime}, index=dates)
    # the bull centroid is closer on the last day of the first episode: only the jump
    # penalty is keeping the model in the bear state there
    regimes["loss_0"] = 1.
    regimes["loss_1"] = np.where(regime.to_numpy() == 1, .5, 2.)
    regimes.loc[dates[9], "loss_1"] = 1.5
    regimes["refit_date"] = dates[0]

    metrics = episode_metrics(extract_episodes(regime, ret), regimes, data, horizon=4)
    assert (metrics["distance_days"] + metrics["penalty_days"] == metrics["length"]).all()
    assert list(metrics["penalty_days"]) == [1., 0., 0.]
    assert abs(metrics["penalty_share"].iloc[0] - 1. / 5.) < 1e-12
    assert metrics["loss_diff_mean"].iloc[1] == -.5              # (.5 - 1.) on every day
    # the forward window looks past the episode and is cut at the end of the sample
    assert list(metrics["fwd_days"]) == [4., 4., 3.]
    assert abs(metrics["ret_fwd"].iloc[0] - ((1 - .02) ** 4 - 1)) < 1e-12
    # no refit parameters, so the two columns that need them are missing rather than wrong
    assert metrics["vol_ratio_state"].isna().all()


def test_episode_metrics_without_loss_columns():
    """A run whose regimes carry no `loss_*` still yields every other metric."""
    regime, ret, dates = _episode_frame()
    close = pd.Series(100. * (1. + ret).cumprod().to_numpy(), index=dates)
    data = pd.DataFrame({"close": close, "ret": ret}, index=dates)
    regimes = pd.DataFrame({"regime": regime, "refit_date": dates[0]}, index=dates)
    metrics = episode_metrics(extract_episodes(regime, ret), regimes, data, horizon=4)
    assert metrics[["loss_diff_mean", "penalty_days", "penalty_share"]].isna().all().all()
    assert metrics[["vol_5_mean", "dd_at_entry", "ret_fwd"]].notna().all().all()


def test_resolve_target_episode():
    """A target may be named by its start date, by a day inside it, or left to default."""
    regime, ret, dates = _episode_frame()
    episodes = extract_episodes(regime, ret)
    assert resolve_target_episode(episodes) == dates[37]             # the last one
    assert resolve_target_episode(episodes, dates[5]) == dates[5]    # its own start
    assert resolve_target_episode(episodes, dates[7]) == dates[5]    # a day inside it
    assert resolve_target_episode(episodes, dates[9]) == dates[5]    # its last day
    try:
        resolve_target_episode(episodes, dates[15])                  # a bull day
    except KeyError as exc:
        assert "국면 안의 날짜" in str(exc)
    else:
        raise AssertionError("국면 밖의 날짜인데 오류가 나지 않았습니다.")


def test_rank_similar_episodes():
    """The ranking is by standardized distance, and a missing metric costs nothing."""
    index = pd.Index(["a", "b", "c", "target"], name="start")
    metrics = pd.DataFrame({"vol_5_mean": [.10, .30, .11, .10],
                            "dd_at_entry": [-.01, -.30, -.02, -.01],
                            "state_flip": [1., 1., 1., 1.]}, index=index)
    ranking = rank_similar_episodes(metrics, target="target",
                                    columns=["vol_5_mean", "dd_at_entry", "state_flip"])
    assert list(ranking.index) == ["a", "c", "b"]
    assert list(ranking["rank"]) == [1, 2, 3]
    assert ranking.loc["a", "distance"] == 0.
    # a metric that never varies carries no information and is dropped from the average
    assert (ranking["n_metrics"] == 2).all()

    # a metric missing for one episode is skipped for that episode only
    metrics.loc["b", "vol_5_mean"] = np.nan
    ranking = rank_similar_episodes(metrics, target="target",
                                    columns=["vol_5_mean", "dd_at_entry"])
    assert ranking.loc["b", "n_metrics"] == 1
    assert ranking.loc["a", "n_metrics"] == 2
    # the default set is narrowed to the metrics the table holds instead of raising...
    assert len(rank_similar_episodes(metrics, target="target")) == 3
    # ...but a metric named explicitly has to be there
    try:
        rank_similar_episodes(metrics, target="target", columns=["nope"])
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("없는 지표를 지정했는데 오류가 나지 않았습니다.")


def test_align_paths_recovers_a_known_lag():
    """A path compared against a shifted copy of itself recovers the shift."""
    shape = 1. + np.sin(np.linspace(0., 3., 100)) * .1
    lag = 7
    # both paths start on their own day 1, but the match starts `lag` days further into the
    # same shape -- so the match is the one running ahead
    target = pd.Series(shape[:60], index=np.arange(1, 61))
    match = pd.Series(shape[lag:lag + 60], index=np.arange(1, 61))

    result = align_paths(target, match, max_lag=20)
    assert result["lag"] == -lag                        # the target is running `lag` days behind
    assert result["rmse"] < 1e-9
    assert set(result["aligned"].columns) == {"target", "match"}
    assert result["n_overlap"] == 60 - lag

    # comparing them day by day, with no lag allowed, is strictly worse
    raw = align_paths(target, match, max_lag=0)
    assert raw["lag"] == 0 and raw["rmse"] > result["rmse"]


def test_normalized_path_and_comparison():
    """Both paths start at 1 on their own entry day, whatever the price level there."""
    dates = pd.bdate_range("2020-01-01", periods=200).date
    close = pd.Series(np.linspace(100., 300., 200), index=dates)
    path = normalized_path(close, dates[50], 30)
    assert len(path) == 30 and path.iloc[0] == 1. and list(path.index[:2]) == [1, 2]
    comparison = compare_episode_paths(close, dates[150], dates[50], horizon=30, max_lag=10)
    assert list(comparison["paths"].columns) == ["target", "match"]
    assert comparison["paths"].iloc[0].tolist() == [1., 1.]
    assert comparison["target_start"] == dates[150]
    # a path cut short by the end of the data still compares, on the overlap it leaves
    assert len(normalized_path(close, dates[190], 30)) == 10


def test_length_scenarios():
    """Every scenario turns a length into a remaining number of days and a date."""
    regime, ret, dates = _episode_frame()
    episodes = extract_episodes(regime, ret)
    similar = rank_similar_episodes(
        episode_metrics(episodes, pd.DataFrame({"regime": regime}, index=dates),
                        pd.DataFrame({"close": 100. * (1. + ret).cumprod(), "ret": ret},
                                     index=dates), horizon=4))
    scenarios = length_scenarios(episodes, pd.Index(dates), similar=similar, top=2)
    # restricted to what had ended before the second episode, only the first one is left
    earlier = length_scenarios(episodes, pd.Index(dates), target=episodes.index[1],
                               before=episodes.index[1])
    assert earlier.loc["p50", "length_days"] == 5.
    assert "p50" in scenarios.index and "similar-1" in scenarios.index
    assert (scenarios["elapsed_days"] == 3).all()               # the running episode
    assert np.allclose(scenarios["remaining_days"], scenarios["length_days"] - 3)
    # the median of the two closed episodes, of 5 and 3 days
    assert scenarios.loc["p50", "length_days"] == 4.
    # the projection runs past the end of the sample, so the date is extrapolated
    assert scenarios.loc["p50", "projected_end"] > dates[-1]


def test_state_losses_explain_the_labels():
    """The recorded losses are the model's own, and disagree with the label only under the penalty."""
    X, ret = _toy_features()
    result = run_rolling_jm(X, ret, model="sjm", jump_penalty=10., window=300, min_window=250,
                            n_init=2, verbose=False)
    losses = result.regimes[["loss_0", "loss_1"]]
    assert losses.notna().all().all() and (losses >= 0.).all().all()
    nearest = losses.to_numpy().argmin(axis=1)
    # most days sit in the state whose centroid is closest; the rest are what the jump
    # penalty is holding, which is exactly what `penalty_days` counts
    agreement = float((nearest == result.regimes.regime.to_numpy()).mean())
    assert agreement > .5, agreement

    episodes = extract_episodes(result.regimes.regime, ret.reindex(result.regimes.index))
    data = pd.DataFrame({"close": (1. + ret).cumprod(), "ret": ret}).reindex(result.regimes.index)
    metrics = episode_metrics(episodes, result.regimes, data, result.params)
    penalty_days = int(((result.regimes.regime == 1) &
                        (result.regimes.loss_1 > result.regimes.loss_0)).sum())
    assert int(metrics["penalty_days"].sum()) == penalty_days
    assert int(metrics["distance_days"].sum() + metrics["penalty_days"].sum()) == int(metrics["length"].sum())


############################################
## 추론 전용 모드
############################################

def test_last_refit_only_matches_the_tail_of_the_full_run():
    """One refit gives exactly the last segment of the full walk, day for day."""
    X, ret = _toy_features()
    shared = dict(model="jm", jump_penalty=10., window=300, min_window=250, n_init=2,
                  verbose=False)
    full = run_rolling_jm(X, ret, **shared)
    last = run_rolling_jm(X, ret, last_refit_only=True, **shared)

    assert len(last.schedule) == 1 and last.schedule[0] == full.schedule[-1]
    refit_date = last.schedule[0][0]
    # the inferred half runs from the refit date to the end of the data
    assert last.regimes.index[0] == refit_date
    assert last.regimes.index[-1] == X.index[-1]
    assert (last.regimes["refit_date"] == refit_date).all()

    # and it is the same inference the full run made over those days: the fit sees only the
    # window before the half, and a day of the half sees only that window plus the days up to it
    common = full.regimes.index.intersection(last.regimes.index)
    assert len(common) == len(last.regimes)
    for col in ("regime", "proba_0", "proba_1", "loss_0", "loss_1"):
        assert np.allclose(full.regimes.loc[common, col], last.regimes.loc[common, col]), col
    # only the last refit's parameters are reported
    assert list(last.params["refit_date"].unique()) == [refit_date]


def _write_sample_sector_and_benchmark(folder, n=1400, seed=1):
    """
    A sector index and its benchmark, in the layout the pipeline reads.

    The benchmark has a bull/bear cycle of its own and the sector adds a second, faster one
    on top of it, so the two signals genuinely differ -- which is the point of subtracting
    one from the other.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    market_state = np.tile(np.repeat([0, 1], 120), n // 240 + 1)[:n]
    market = rng.normal(np.where(market_state == 0, .0008, -.001),
                        np.where(market_state == 0, .007, .02))
    own_state = np.tile(np.repeat([0, 1], 70), n // 140 + 1)[:n]
    own = rng.normal(np.where(own_state == 0, .0009, -.0012),
                     np.where(own_state == 0, .005, .012))
    sector = (1. + market) * (1. + own) - 1.

    main = os.path.join(folder, "sector.csv")
    side = os.path.join(folder, "benchmark.csv")
    pd.DataFrame({"날짜": dates.date,
                  "종가": (100. * np.cumprod(1. + sector)).round(4),
                  "무위험금리": 3.}).to_csv(main, index=False)
    pd.DataFrame({"날짜": dates.date,
                  "종가": (2000. * np.cumprod(1. + market)).round(4)}).to_csv(side, index=False)
    return main, side


def test_the_model_is_fitted_on_the_benchmark_subtracted_return():
    """With a benchmark, the features, the state ordering and the tables all follow `rel_ret`."""
    with tempfile.TemporaryDirectory() as folder:
        main, side = _write_sample_sector_and_benchmark(folder)
        shared = dict(feature_set="paper", warmup=60, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # the first day has no benchmark return
            data, X, _ = prepare_inputs(main, bench_file=side, **shared)
            plain_data, plain_X, _ = prepare_inputs(main, **shared)

        # the signal is the relative return, and the features are the ones it produces
        assert np.allclose(data.signal_ret, data.rel_ret, equal_nan=True)
        assert np.allclose(plain_data.signal_ret, plain_data.excess_ret)
        expected = build_features(data.rel_ret, ver="paper", warmup=60)
        assert list(X.columns) == list(expected.columns)
        assert X.index.equals(expected.index) and np.allclose(X, expected)
        # and they are not the ones the raw sector return produces
        common = X.index.intersection(plain_X.index)
        assert len(common) > 100 and not np.allclose(X.loc[common], plain_X.loc[common])

        # forcing the excess return back reproduces the benchmark-free run exactly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forced_data, forced_X, _ = prepare_inputs(main, bench_file=side,
                                                      signal_ret="excess", **shared)
        assert np.allclose(forced_data.signal_ret, forced_data.excess_ret)
        assert np.allclose(forced_X.loc[common], plain_X.loc[common])
        # the benchmark columns still come along for the ride, and get written out
        assert market_columns(forced_data) == ["close", "ret", "rf", "excess_ret",
                                               "bench_close", "bench_ret", "rel_ret"]
        assert market_columns(plain_data) == ["close", "ret", "rf", "excess_ret"]


def test_run_inference_on_a_sector_against_its_benchmark():
    """The whole inference run goes through with a benchmark, and reports the relative call."""
    with tempfile.TemporaryDirectory() as folder:
        main, side = _write_sample_sector_and_benchmark(folder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = run_inference(main, outdir=folder, bench_file=side, feature_set="paper",
                                jump_penalty=10., window=300, min_window=250, warmup=60,
                                n_init=2, plot=False, verbose=False)

        regimes = out["regimes"]
        # the regimes are those of the relative return, and the table says what they were read on
        assert np.allclose(out["data"].signal_ret, out["data"].rel_ret, equal_nan=True)
        for col in ("bench_close", "bench_ret", "rel_ret"):
            assert col in regimes.columns, col
        # the strategy side is untouched: the weight still maps the regime of the asset itself
        assert set(regimes.regime.unique()) <= {0, 1}
        assert out["summary"]["regime_name"] in ("bull", "bear")
        assert regimes["weight"].isin([0., 1.]).all()
        assert np.allclose(regimes.close, out["data"].close.reindex(regimes.index))


def _write_sample_input(folder, n=1400, seed=0):
    """A small price file in the layout the pipeline reads, written into `folder`."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    state = np.tile(np.repeat([0, 1], 120), n // 240 + 1)[:n]
    ret = rng.normal(np.where(state == 0, .0008, -.001), np.where(state == 0, .007, .02))
    path = os.path.join(folder, "sample.csv")
    pd.DataFrame({"날짜": dates.date,
                  "종가": (100. * np.cumprod(1. + ret)).round(4),
                  "무위험금리": 3.}).to_csv(path, index=False)
    return path


def test_run_inference_covers_the_current_half_only():
    """The inference mode fits once before the current half and reports where it stands now."""
    with tempfile.TemporaryDirectory() as folder:
        out = run_inference(_write_sample_input(folder), outdir=folder, feature_set="paper",
                            jump_penalty=10., window=300, min_window=250, warmup=60,
                            n_init=2, plot=False, verbose=False)

        result, regimes, summary = out["result"], out["regimes"], out["summary"]
        assert len(result.schedule) == 1
        refit_date = result.schedule[0][0]
        # the training window ends the day before the half starts: no day of the half is fitted on
        assert summary["train_end"] < refit_date <= summary["inference_start"]
        assert summary["n_train"] == 300
        assert summary["inference_start"] == regimes.index[0] == refit_date
        assert summary["asof"] == regimes.index[-1] == out["X"].index[-1]
        assert summary["n_inference"] == len(regimes)

        # the headline call is the last day's, and the run length reaches back to its start
        assert summary["regime"] == regimes["regime"].iloc[-1]
        assert summary["regime_name"] in ("bull", "bear")
        assert 1 <= summary["run_length"] <= len(regimes)
        assert (regimes["regime"].iloc[-summary["run_length"]:] == summary["regime"]).all()
        assert summary["run_start"] == regimes.index[len(regimes) - summary["run_length"]]
        # the recommended weight is the signal mapped through the cash limits, nothing more
        assert summary["weight"] in (0., 1.)

        # every file is prefixed, so an inference run cannot overwrite a backtest run
        names = [os.path.basename(path) for path in out["written"]]
        assert all(name.startswith("inference_") for name in names), names
        assert "inference_regimes.csv" in names and "inference_summary.csv" in names
        # the plain model has no feature weights, so no weight tables are written
        assert out["weight_groups"] is None
        assert not [name for name in names if "weight" in name]


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)}개 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
