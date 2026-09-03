#!/usr/bin/env python
"""
Checks for the pieces of FactorGCL that are easy to get subtly wrong: the hypergraph
convolution against the formula it implements, the cascading residual decomposition, the
InfoNCE loss, the label and normalisation conventions of the supplement, the rolling split
layout and its embargo, the metrics and the TopK backtest accounting.

Run directly (`python test_factorgcl.py`) or through `pytest test_factorgcl.py`.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import daily_returns, resolve_costs, run_topk_backtest
from baselines import BASELINES, BaselineConfig
from data import (FEATURE_FIELDS, PanelData, PreprocessConfig, build_prior_exposures,
                  compute_raw_labels, pivot_panel, standardize_labels)
from loss import factorgcl_loss, info_nce_loss, multi_period_mse
from metrics import (daily_ic, f1_score, ic_icir, investment_metrics, max_drawdown, mse)
from model import FactorGCL, HyperGCNLayer, ProjectionHead
from synthetic import make_synthetic_panel
from train import (ModelConfig, TrainConfig, evaluate, indices_in_range, predict,
                   rolling_splits, set_seed, train_model)

SEED = 0


def _small_panel(n_stocks=24, n_days=180, seq_len=15, future_len=8, periods=(1, 5)):
    panel_df, industry = make_synthetic_panel(n_stocks=n_stocks, n_days=n_days, n_industries=4,
                                              n_hidden_factors=2, seed=SEED)
    config = PreprocessConfig(seq_len=seq_len, future_len=future_len, periods=periods)
    return PanelData(panel_df, industry, config)


# --------------------------------------------------------------------------- the hypergraph

def test_hypergcn_matches_the_formula():
    """Eq. 1: sigma(Dn^-1/2 H W De^-1 H^T Dn^-1/2 e w), built densely for comparison."""
    torch.manual_seed(SEED)
    layer = HyperGCNLayer(5, 3)
    e = torch.randn(9, 5)
    incidence = (torch.rand(9, 6) > 0.4).float()
    weight = torch.rand(6) + 0.5

    dn = torch.diag((incidence * weight).sum(1).pow(-0.5))
    de = torch.diag(incidence.sum(0).reciprocal())
    propagation = dn @ incidence @ torch.diag(weight) @ de @ incidence.T @ dn
    expected = torch.nn.functional.leaky_relu(propagation @ layer.linear(e), 0.01)
    assert torch.allclose(layer(e, incidence, weight), expected, atol=1e-5)


def test_hypergcn_survives_isolated_nodes_and_edges():
    """A stock in no industry, or an industry with no stock, must not produce nan."""
    torch.manual_seed(SEED)
    layer = HyperGCNLayer(4, 4)
    incidence = torch.zeros(5, 3)
    incidence[0, 0] = incidence[1, 0] = incidence[2, 1] = 1.0   # stocks 3, 4 and edge 2 idle
    out = layer(torch.randn(5, 4), incidence)
    assert torch.isfinite(out).all()
    # an isolated node receives no message at all, so its output is the activation of zero
    assert torch.allclose(out[3], torch.zeros(4))


def test_hypergcn_accepts_soft_incidence():
    """The hidden beta module feeds exposures in (0, 1); degrees are then weighted sums."""
    torch.manual_seed(SEED)
    layer = HyperGCNLayer(4, 4)
    soft = torch.rand(7, 5)
    out = layer(torch.randn(7, 4), soft)
    assert out.shape == (7, 4) and torch.isfinite(out).all()


# ------------------------------------------------------------------- the cascade of Eq. 5-8

def _model(n_prior=4, n_hidden=6, hidden=8, periods=3, **kwargs):
    torch.manual_seed(SEED)
    return FactorGCL(input_size=len(FEATURE_FIELDS), num_prior_factors=n_prior,
                     hidden_size=hidden, num_hidden_factors=n_hidden, num_rnn_layers=2,
                     num_periods=periods, **kwargs)


def _inputs(n_stocks=11, seq_len=12, n_prior=4):
    torch.manual_seed(SEED + 1)
    x = torch.randn(n_stocks, seq_len, len(FEATURE_FIELDS))
    beta = torch.zeros(n_stocks, n_prior)
    beta[torch.arange(n_stocks), torch.arange(n_stocks) % n_prior] = 1.0
    return x, beta


def test_cascading_residuals_follow_the_article():
    """e_r = e_s - e_p, then e_alpha = LeakyReLU(w(e_s - e_p - e_h) + b) (Eq. 7)."""
    model = _model().eval()
    x, beta = _inputs()
    out = model(x, beta)
    residual_after_prior = out["stock_embeddings"] - out["prior_beta"]
    expected_hidden, _ = model.hidden_beta(residual_after_prior, out["beta_hidden"])
    assert torch.allclose(out["hidden_beta"], expected_hidden, atol=1e-6)
    expected_alpha = model.individual_alpha(residual_after_prior - out["hidden_beta"])
    assert torch.allclose(out["alpha"], expected_alpha, atol=1e-6)


def test_prediction_is_the_sum_of_the_three_components():
    """Eq. 8: y = w_o1 e_p + w_o2 e_h + w_o3 e_alpha + b_o."""
    model = _model().eval()
    x, beta = _inputs()
    out = model(x, beta)
    expected = (model.out_prior(out["prior_beta"]) + model.out_hidden(out["hidden_beta"])
                + model.out_alpha(out["alpha"]) + model.out_bias)
    assert torch.allclose(out["y_hat"], expected, atol=1e-6)
    assert out["y_hat"].shape == (x.shape[0], model.num_periods)


def test_hidden_exposures_are_soft_hyperedges():
    """beta_h = Sigmoid(e_r . c^T), of shape (N, M) and strictly inside (0, 1)."""
    model = _model(n_hidden=6).eval()
    x, beta = _inputs()
    beta_hidden = model(x, beta)["beta_hidden"]
    assert beta_hidden.shape == (x.shape[0], 6)
    assert (beta_hidden > 0).all() and (beta_hidden < 1).all()


def test_ablation_switches_drop_their_module():
    """The four variants of Table 2 must run and keep the prediction shape."""
    x, beta = _inputs()
    for flags in ({"use_prior": False}, {"use_hidden": False}, {"use_alpha": False}):
        model = _model(**flags).eval()
        out = model(x, beta)
        assert out["y_hat"].shape == (x.shape[0], model.num_periods)
        dropped = [name for name, keep in
                   (("prior_beta", model.use_prior), ("hidden_beta", model.use_hidden),
                    ("alpha", model.use_alpha)) if not keep]
        for name in dropped:
            assert out[name] is None


def test_future_branch_shares_the_factor_modules_and_reuses_exposures():
    """Eq. 9-10: phi'_prior and phi'_hidden are phi_prior and phi_hidden, and the exposures
    come from the historical branch."""
    model = _model().eval()
    x, beta = _inputs()
    out = model(x, beta)
    x_future = torch.randn(x.shape[0], 7, len(FEATURE_FIELDS))
    future = model.forward_future(x_future, beta, out["beta_hidden"])
    e_s = model.future_feature_extractor(x_future)
    prior = model.prior_beta(e_s, beta)                       # shared parameters
    hidden, used = model.hidden_beta(e_s - prior, out["beta_hidden"])
    assert torch.allclose(used, out["beta_hidden"])           # exposures reused, not re-mined
    # Eq. 10 leaves the future alpha as the bare residual
    assert torch.allclose(future["alpha"], e_s - prior - hidden, atol=1e-6)
    assert model.feature_extractor is not model.future_feature_extractor


def test_shared_future_encoder_flag():
    model = _model(share_future_feature_extractor=True)
    assert model.feature_extractor is model.future_feature_extractor


def test_future_alpha_module_flag_applies_equation_7():
    model = _model(future_alpha_module=True).eval()
    x, beta = _inputs()
    out = model(x, beta)
    x_future = torch.randn(x.shape[0], 7, len(FEATURE_FIELDS))
    future = model.forward_future(x_future, beta, out["beta_hidden"])
    e_s = model.future_feature_extractor(x_future)
    residual = e_s - model.prior_beta(e_s, beta)
    residual = residual - model.hidden_beta(residual, out["beta_hidden"])[0]
    assert torch.allclose(future["alpha"], model.individual_alpha(residual), atol=1e-6)


def test_model_rejects_an_empty_cascade():
    try:
        _model(use_prior=False, use_hidden=False, use_alpha=False)
    except ValueError:
        return
    raise AssertionError("a model with no module at all should not be constructible")


# ----------------------------------------------------------------------------- the losses

def test_info_nce_matches_its_definition():
    """Eq. 11, recomputed by hand from cosine similarities."""
    torch.manual_seed(SEED)
    past, future = torch.randn(6, 5), torch.randn(6, 5)
    tau = 0.1
    past_n = past / past.norm(dim=1, keepdim=True)
    future_n = future / future.norm(dim=1, keepdim=True)
    similarity = past_n @ future_n.T / tau
    expected = -torch.mean(torch.diag(similarity) - torch.logsumexp(similarity, dim=1))
    assert torch.allclose(info_nce_loss(past, future, temperature=tau), expected, atol=1e-6)


def test_info_nce_prefers_temporally_consistent_alphas():
    """Matching past and future embeddings must score better than shuffled ones."""
    torch.manual_seed(SEED)
    alpha = torch.randn(8, 5)
    aligned = info_nce_loss(alpha, alpha + 0.01 * torch.randn(8, 5))
    shuffled = info_nce_loss(alpha, (alpha + 0.01 * torch.randn(8, 5))[torch.randperm(8)])
    assert float(aligned) < float(shuffled)
    # identical past and future embeddings sit at the floor of the loss for this batch
    perfect = float(info_nce_loss(alpha, alpha.clone(), temperature=0.1))
    assert perfect <= float(aligned) + 1e-6 and perfect < 0.1


def test_info_nce_uses_the_projection_head():
    torch.manual_seed(SEED)
    head = ProjectionHead(5)
    past, future = torch.randn(4, 5), torch.randn(4, 5)
    assert torch.allclose(info_nce_loss(past, future, head),
                          info_nce_loss(head(past), head(future)), atol=1e-6)


def test_multi_period_mse_honours_the_label_mask():
    y_hat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    mask = torch.tensor([[True, False], [True, True]])
    assert np.isclose(float(multi_period_mse(y_hat, y)), (1 + 4 + 9 + 16) / 4)
    assert np.isclose(float(multi_period_mse(y_hat, y, mask)), (1 + 9 + 16) / 3)


def test_objective_is_mse_plus_gamma_times_contrastive():
    """Eq. 13, and the '-wo CL' ablation that switches the second term off."""
    torch.manual_seed(SEED)
    y_hat, y = torch.randn(5, 3), torch.randn(5, 3)
    past, future = torch.randn(5, 4), torch.randn(5, 4)
    total, parts = factorgcl_loss(y_hat, y, past, future, gamma=0.25, temperature=0.1)
    assert torch.allclose(total, parts["mse"] + 0.25 * parts["contrastive"], atol=1e-6)
    without, parts0 = factorgcl_loss(y_hat, y, past, future, gamma=0.0)
    assert torch.allclose(without, parts0["mse"]) and float(parts0["contrastive"]) == 0.0


# ------------------------------------------------------------------------ the data pipeline

def test_labels_follow_the_supplement_formula():
    """y~_t = (price_{t+dt+1} - price_{t+1}) / price_{t+1}."""
    price = np.arange(1, 11, dtype=float).reshape(10, 1) ** 1.5
    labels = compute_raw_labels(price, (1, 3))
    for t in range(10):
        for k, dt in enumerate((1, 3)):
            if t + dt + 1 <= 9:
                expected = (price[t + dt + 1, 0] - price[t + 1, 0]) / price[t + 1, 0]
                assert np.isclose(labels[t, 0, k], expected)
            else:
                assert np.isnan(labels[t, 0, k])


def test_labels_are_cross_sectionally_standardised():
    raw = np.random.default_rng(SEED).normal(size=(4, 30, 2)) * 0.03 + 0.01
    standardized = standardize_labels(raw, mad_clip=None)
    for t in range(4):
        for k in range(2):
            row = standardized[t, :, k]
            assert np.isclose(np.nanmean(row), 0.0, atol=1e-9)
            assert np.isclose(np.nanstd(row), 1.0, atol=1e-9)


def test_label_winsorisation_tames_an_outlier():
    raw = np.zeros((1, 21, 1))
    raw[0, :, 0] = np.linspace(-0.02, 0.02, 21)
    raw[0, 20, 0] = 50.0                                   # an absurd return
    clipped = standardize_labels(raw, mad_clip=5.0)[0, :, 0]
    unclipped = standardize_labels(raw, mad_clip=None)[0, :, 0]
    assert clipped[20] < 20 and unclipped[20] > 4          # the outlier no longer dominates
    assert np.isclose(np.corrcoef(clipped[:20], raw[0, :20, 0])[0, 1], 1.0)


def test_windows_are_dimensionless():
    """Prices are divided by the current close, volume by the window's average volume."""
    panel = _small_panel()
    index = panel.usable_date_indices(need_future=True)[3]
    sample = panel.cross_section(index, with_future=True)
    close_channel = FEATURE_FIELDS.index("close")
    volume_channel = FEATURE_FIELDS.index("volume")
    assert np.allclose(sample.x[:, -1, close_channel], 1.0, atol=1e-5)
    assert np.allclose(sample.x[:, :, volume_channel].mean(axis=1), 1.0, atol=1e-3)
    assert sample.x.shape == (len(sample.stocks), panel.config.seq_len, len(FEATURE_FIELDS))
    assert sample.x_future.shape[1] == panel.config.future_len


def test_missing_values_are_dropped_then_filled():
    """A stock whose window is mostly missing leaves the cross-section; the rest is filled."""
    panel_df, industry = make_synthetic_panel(n_stocks=8, n_days=60, n_industries=2, seed=SEED)
    victim = panel_df["stock"].unique()[0]
    hole = (panel_df["stock"] == victim) & (panel_df["date"] >= panel_df["date"].unique()[20])
    panel_df.loc[hole, list(FEATURE_FIELDS)] = np.nan
    panel = PanelData(panel_df, industry, PreprocessConfig(seq_len=10, future_len=5,
                                                           periods=(1, 2)))
    index = panel.usable_date_indices(need_future=True)[-1]
    sample = panel.cross_section(index, with_future=True)
    assert victim not in set(sample.stocks)
    assert np.isfinite(sample.x).all()


def test_features_are_clipped():
    panel_df, industry = make_synthetic_panel(n_stocks=6, n_days=60, n_industries=2, seed=SEED)
    panel_df.loc[panel_df.index[-1], "volume"] = 1e18       # one absurd volume print
    config = PreprocessConfig(seq_len=10, future_len=5, periods=(1, 2), feature_clip=3.0)
    panel = PanelData(panel_df, industry, config)
    index = panel.usable_date_indices(need_future=True)[-1]
    sample = panel.cross_section(index, with_future=True)
    assert np.abs(sample.x).max() <= 3.0 + 1e-6


def test_prior_exposures_are_industry_indicators():
    beta, factors = build_prior_exposures({"A": "bank", "B": "tech", "C": "bank"},
                                          ["A", "B", "C", "D"])
    assert list(factors) == ["bank", "tech"]
    assert np.array_equal(beta, np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32))
    # a fixed factor space keeps every rolling window on the same hyperedge set
    beta2, factors2 = build_prior_exposures({"A": "bank"}, ["A"], factors=["bank", "tech", "auto"])
    assert beta2.shape == (1, 3) and list(factors2) == ["bank", "tech", "auto"]


def test_usable_days_need_a_full_window_on_both_sides():
    panel = _small_panel(n_days=120, seq_len=15, future_len=8, periods=(1, 5))
    n_dates = len(panel.dates)
    with_future = panel.usable_date_indices(need_future=True)
    assert with_future[0] == panel.config.seq_len - 1               # T days of history
    assert with_future[-1] <= n_dates - 1 - panel.config.future_len  # T' days of future
    assert with_future[-1] <= n_dates - 1 - (max(panel.config.periods) + 1)  # and a label
    assert panel.usable_date_indices(need_future=False, need_label=False)[-1] == n_dates - 1


def test_pivot_panel_keeps_gaps_as_nan():
    panel_df, industry = make_synthetic_panel(n_stocks=3, n_days=6, n_industries=1, seed=SEED)
    panel_df = panel_df.iloc[1:]                                     # drop one stock-day
    dates, stocks, matrices = pivot_panel(panel_df)
    assert matrices["close"].shape == (6, 3)
    assert np.isnan(matrices["close"]).sum() == 1


# ------------------------------------------------------------------- the rolling protocol

def test_rolling_splits_have_the_article_layout():
    """5 years train : 1 year validation : 2 years test, tiling the test period."""
    dates = pd.bdate_range("2014-01-01", "2023-06-30")
    splits = rolling_splits(dates, 5, 1, 2, test_start="2020-01-01", test_end="2023-06-30")
    assert len(splits) == 2
    first = splits[0]
    assert first.test[0] == pd.Timestamp("2020-01-01")
    assert first.test[1] == pd.Timestamp("2021-12-31")
    assert first.valid == (pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"))
    assert first.train[1] == pd.Timestamp("2018-12-31")
    assert first.train[0] == pd.Timestamp("2014-01-01")
    second = splits[1]
    assert second.test == (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-30"))
    assert second.valid == (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))
    assert second.train == (pd.Timestamp("2016-01-01"), pd.Timestamp("2020-12-31"))
    # the windows are contiguous and never overlap
    assert first.test[1] + pd.Timedelta(days=1) == second.test[0]
    for split in splits:
        assert split.train[1] < split.valid[0] < split.valid[1] < split.test[0]


def test_embargo_keeps_training_labels_out_of_the_validation_window():
    panel = _small_panel(n_days=200, seq_len=10, future_len=5, periods=(1, 5))
    dates = pd.DatetimeIndex(pd.to_datetime(panel.dates))
    start, end = dates[20], dates[120]
    embargo = max(max(panel.config.periods) + 1, panel.config.future_len)
    plain = indices_in_range(panel, start, end, need_future=True, embargo=0)
    embargoed = indices_in_range(panel, start, end, need_future=True, embargo=embargo)
    assert max(plain) == 120
    assert max(embargoed) == 120 - embargo
    # every embargoed day's label and future window stay inside the range
    assert dates[max(embargoed) + max(panel.config.periods) + 1] <= end


# ---------------------------------------------------------------------------- the metrics

def test_daily_ic_is_the_cross_sectional_correlation():
    rng = np.random.default_rng(SEED)
    y, y_hat = rng.normal(size=50), rng.normal(size=50)
    assert np.isclose(daily_ic(y, y_hat), np.corrcoef(y, y_hat)[0, 1])
    assert np.isnan(daily_ic([1.0], [1.0]))
    assert np.isnan(daily_ic([1.0, 1.0], [0.5, 0.5]))       # no dispersion


def test_ic_and_icir_over_days():
    frame = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 4,
                          "label": [1.0, 2, 3, 4, 4, 3, 2, 1],
                          "pred": [1.0, 2, 3, 4, 1, 2, 3, 4]})
    ic, icir = ic_icir(frame)
    assert np.isclose(ic, 0.0, atol=1e-12)                  # +1 on one day, -1 on the other
    assert np.isclose(icir, 0.0, atol=1e-12)
    frame.loc[frame["date"] == "d2", "pred"] = [4.0, 3, 2, 1]
    ic, icir = ic_icir(frame)
    assert np.isclose(ic, 1.0) and np.isinf(icir) or np.isnan(icir) or icir > 1e6


def test_f1_uses_the_supplement_counts():
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    y_hat = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    # TP = 2, FP = 1, FN = 1 -> precision = 2/3, recall = 2/3
    assert np.isclose(f1_score(y, y_hat), 2 / 3)
    assert f1_score(np.array([1.0]), np.array([-1.0])) == 0.0


def test_mse_ignores_missing_entries():
    assert np.isclose(mse([1.0, 2.0, np.nan], [0.0, 0.0, 5.0]), 2.5)


def test_max_drawdown_of_a_cumulative_curve():
    assert np.isclose(max_drawdown([0.1, 0.3, 0.05, 0.2]), 0.25)
    assert np.isclose(max_drawdown([-0.1, -0.2]), 0.2)      # measured from the initial peak


def test_investment_metrics_match_their_definitions():
    rng = np.random.default_rng(SEED)
    strategy = pd.Series(rng.normal(0.001, 0.01, size=252))
    benchmark = pd.Series(rng.normal(0.0005, 0.01, size=252))
    stats = investment_metrics(strategy, benchmark)
    excess = strategy - benchmark
    assert np.isclose(stats["AR"], excess.sum() / 252 * 252)
    assert np.isclose(stats["IR"], excess.mean() / excess.std(ddof=1) * np.sqrt(252))
    assert np.isclose(stats["RoMaD"], stats["AR"] / max_drawdown(excess.cumsum()))
    assert np.isclose(stats["CER"], excess.sum())


# --------------------------------------------------------------------------- the backtest

def _flat_market(n_days=40, n_stocks=4):
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    stocks = [f"S{i}" for i in range(n_stocks)]
    # stock i compounds at a constant daily rate, so every horizon's return is known exactly
    rates = np.array([0.02, 0.01, 0.0, -0.01])[:n_stocks]
    prices = np.outer(np.ones(n_days), np.ones(n_stocks)) * np.cumprod(
        np.tile(1 + rates, (n_days, 1)), axis=0)
    return pd.DataFrame(prices, index=dates, columns=stocks)


def _one_signal_day(prices, scores, day=0):
    """A prediction frame with a single signal day, so exactly one tranche is ever alive."""
    return pd.DataFrame({"date": [prices.index[day]] * len(prices.columns),
                         "stock": list(prices.columns), "pred_5": list(scores)})


def test_topk_tranche_return_equals_the_label_it_is_trained_on():
    """With no cost, a tranche's compounded daily returns reproduce
    price_{t+dt+1} / price_{t+1} - 1, the label of the anchor day."""
    prices = _flat_market()
    predictions = _one_signal_day(prices, [3.0, 2.0, 1.0, 0.0])
    frame = run_topk_backtest(predictions, prices, "pred_5", topk=1, holding=5, cost=0.0)
    assert len(frame) == 5                                  # one tranche, held five days
    best = prices.columns[0]                                # the 2%-a-day stock
    label = prices[best].iloc[0 + 5 + 1] / prices[best].iloc[0 + 1] - 1
    assert np.isclose((1 + frame["strategy"]).prod() - 1, label, atol=1e-9)
    assert np.isclose(label, 1.02 ** 5 - 1)


def test_transaction_cost_reduces_the_tranche_return():
    """The cost is charged once on the way in and once on the way out of a tranche."""
    prices = _flat_market()
    predictions = _one_signal_day(prices, [3.0, 2.0, 1.0, 0.0])
    free = run_topk_backtest(predictions, prices, "pred_5", topk=1, holding=5, cost=0.0)
    charged = run_topk_backtest(predictions, prices, "pred_5", topk=1, holding=5, cost=0.003,
                                cost_mode="per_side")
    gross = (1 + free["strategy"]).prod()
    net = (1 + charged["strategy"]).prod()
    assert np.isclose(net, gross * (1 - 0.003) ** 2, atol=1e-12)
    half = run_topk_backtest(predictions, prices, "pred_5", topk=1, holding=5, cost=0.003,
                             cost_mode="round_trip")
    assert np.isclose((1 + half["strategy"]).prod(), gross * (1 - 0.0015) ** 2, atol=1e-12)
    assert resolve_costs(0.003, "round_trip") == (0.0015, 0.0015)
    assert resolve_costs(0.003, "per_side") == (0.003, 0.003)


def test_topk_selects_the_highest_scores_and_benchmarks_equally():
    prices = _flat_market()
    predictions = pd.DataFrame({"date": np.repeat(prices.index[:20], 4),
                                "stock": list(prices.columns) * 20,
                                "pred_5": list(np.array([0.0, 1.0, 2.0, 3.0])) * 20})
    frame = run_topk_backtest(predictions, prices, "pred_5", topk=1, holding=5, cost=0.0)
    # the top-scored stock is the one that loses 1% a day
    assert np.isclose((1 + frame["strategy"]).iloc[:5].prod() - 1, 0.99 ** 5 - 1, atol=1e-9)
    returns = daily_returns(prices)
    assert np.allclose(frame["benchmark"], returns.mean(axis=1).reindex(frame.index))


# ------------------------------------------------------------- training and the baselines

def test_training_reduces_the_loss_and_restores_the_best_state():
    panel = _small_panel(n_stocks=20, n_days=140, seq_len=10, future_len=5, periods=(1, 5))
    indices = panel.usable_date_indices(need_future=True)
    train_indices, valid_indices = indices[:50], indices[60:80]
    model_config = ModelConfig(hidden_size=8, num_hidden_factors=4, num_rnn_layers=1)
    train_config = TrainConfig(epochs=3, patience=3, verbose=False, gamma=0.1)
    model, history = train_model(panel, train_indices, valid_indices, model_config, train_config)
    assert len(history) == 3
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    assert history["contrastive"].iloc[-1] < history["contrastive"].iloc[0]
    # the returned model is the one of the best validation step
    best_epoch = int(history["valid_score"].idxmax())
    set_seed(train_config.seed)
    frame = predict(model, panel, indices[80:95])
    assert not frame.empty and np.isfinite(frame["pred_1"]).all()
    assert 0 <= best_epoch < len(history)


def test_contrastive_loss_requires_the_alpha_module():
    panel = _small_panel(n_stocks=12, n_days=100, seq_len=8, future_len=4, periods=(1,))
    indices = panel.usable_date_indices(need_future=True)
    try:
        train_model(panel, indices[:5], indices[5:10],
                    ModelConfig(hidden_size=8, use_alpha=False),
                    TrainConfig(epochs=1, gamma=0.1, verbose=False))
    except ValueError:
        return
    raise AssertionError("gamma > 0 without the alpha module should be rejected")


def test_evaluate_reports_every_period():
    panel = _small_panel(n_stocks=16, n_days=120, seq_len=8, future_len=4, periods=(1, 5))
    indices = panel.usable_date_indices(need_future=True)
    model_config = ModelConfig(hidden_size=8, num_hidden_factors=4, num_rnn_layers=1)
    model, _ = train_model(panel, indices[:20], indices[20:30], model_config,
                           TrainConfig(epochs=1, verbose=False))
    metrics = evaluate(predict(model, panel, indices[30:50]), panel.config.periods)
    assert list(metrics.index) == [1, 5]
    assert set(metrics.columns) == {"IC", "ICIR", "MSE", "F1"}


def test_every_baseline_produces_multi_period_predictions():
    x = torch.randn(9, 12, len(FEATURE_FIELDS))
    beta = torch.eye(9)[:, :3]
    for name in BASELINES:
        model = BaselineConfig(name=name, hidden_size=8, seq_len=12).build(
            len(FEATURE_FIELDS), 3, 4)
        assert model(x, beta)["y_hat"].shape == (9, 4)
        assert model.use_alpha is False


def _run_all():
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"  ok   {name}")
        except Exception as error:                            # noqa: BLE001 - a test runner
            failures += 1
            print(f"  FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
