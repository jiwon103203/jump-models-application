#!/usr/bin/env python
"""
Command-line entry point: prepare the panel, run the rolling protocol, score the
predictions with the article's metrics and run the TopK investment simulation.

    # end to end on generated data, to check the installation
    python run_factorgcl.py --synthetic --outdir out

    # the article's settings on a real panel
    python run_factorgcl.py --input panel.csv --industry industry.csv \
        --test-start 2020-01-01 --test-end 2023-06-30 --outdir out

    # the ablation study of Table 2
    python run_factorgcl.py --synthetic --ablation --outdir out

Written outputs: ``predictions.csv`` (the out-of-sample scores), ``metrics.csv`` (IC, ICIR,
MSE and F1 per forward period), ``backtest.csv`` and ``backtest_metrics.csv`` (the TopK
simulation), ``splits.csv`` (the rolling windows) and ``history_window*.csv`` (the training
curves).
"""

import argparse
import os

import pandas as pd

from backtest import (DEFAULT_COST, DEFAULT_HOLDING, DEFAULT_TOPK, backtest_metrics,
                      price_frame_from_panel, run_topk_backtest)
from baselines import BASELINES, BaselineConfig
from data import PanelData, PreprocessConfig
from model import DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_HIDDEN_FACTORS, DEFAULT_NUM_RNN_LAYERS
from train import (DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE, DEFAULT_SEED,
                   ModelConfig, TrainConfig, evaluate, rolling_splits, run_rolling)

# the four variants of the ablation study (Table 2), as overrides of ModelConfig/TrainConfig
ABLATIONS = {
    "FactorGCL": {},
    "-wo Prior": {"use_prior": False},
    "-wo Hidden": {"use_hidden": False},
    "-wo Alpha&CL": {"use_alpha": False, "gamma": 0.0},
    "-wo CL": {"gamma": 0.0},
}


def read_table(path, **kwargs):
    """Read a csv or parquet file."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, **kwargs)


def load_inputs(args):
    """The long-format panel and the industry map, generated or read from disk."""
    if args.synthetic:
        from synthetic import make_synthetic_panel
        panel, industry = make_synthetic_panel(
            n_stocks=args.synthetic_stocks, n_days=args.synthetic_days,
            n_industries=args.synthetic_industries, seed=args.seed)
        return panel, industry
    if not args.input:
        raise SystemExit("pass --input (a price-volume panel) or --synthetic")
    panel = read_table(args.input)
    panel["date"] = pd.to_datetime(panel["date"])
    if args.industry:
        industry = read_table(args.industry)
        if "industry" not in industry.columns or "stock" not in industry.columns:
            raise SystemExit("the industry file needs the columns 'stock' and 'industry'")
        industry = industry.set_index("stock")["industry"]
    elif "industry" in panel.columns:
        industry = panel.drop_duplicates("stock").set_index("stock")["industry"]
    else:
        raise SystemExit("pass --industry, or add an 'industry' column to the panel")
    return panel, industry


def build_parser():
    parser = argparse.ArgumentParser(
        description="FactorGCL (Duan, Wang and Li, AAAI-25) on a price-volume panel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    source = parser.add_argument_group("data")
    source.add_argument("--input", help="long-format panel: date, stock, open/high/low/close/vwap/volume")
    source.add_argument("--industry", help="prior factor map: stock, industry")
    source.add_argument("--synthetic", action="store_true", help="generate a panel instead")
    source.add_argument("--synthetic-stocks", type=int, default=120)
    source.add_argument("--synthetic-days", type=int, default=900)
    source.add_argument("--synthetic-industries", type=int, default=8)
    source.add_argument("--outdir", default="out")

    prep = parser.add_argument_group("preprocessing")
    prep.add_argument("--seq-len", type=int, default=60, help="T, the historical window")
    prep.add_argument("--future-len", type=int, default=20, help="T', the contrastive window")
    prep.add_argument("--periods", type=int, nargs="+", default=[1, 5, 10, 20],
                      help="the forward prediction periods")
    prep.add_argument("--price-field", default="vwap", help="the price the labels use")
    prep.add_argument("--max-missing-ratio", type=float, default=0.2)
    prep.add_argument("--feature-clip", type=float, default=10.0)
    prep.add_argument("--label-mad-clip", type=float, default=5.0)
    prep.add_argument("--future-norm", choices=("anchor", "window"), default="anchor")

    arch = parser.add_argument_group("model")
    arch.add_argument("--model", default="factorgcl", choices=("factorgcl",) + BASELINES)
    arch.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help="H")
    arch.add_argument("--num-hidden-factors", type=int, default=DEFAULT_NUM_HIDDEN_FACTORS,
                      help="M")
    arch.add_argument("--rnn-layers", type=int, default=DEFAULT_NUM_RNN_LAYERS)
    arch.add_argument("--dropout", type=float, default=0.0)
    arch.add_argument("--bn-position", default="input", choices=("input", "output", "both", "none"))
    arch.add_argument("--no-prior", action="store_true", help="ablation: drop the prior beta module")
    arch.add_argument("--no-hidden", action="store_true", help="ablation: drop the hidden beta module")
    arch.add_argument("--no-alpha", action="store_true",
                      help="ablation: drop the individual alpha module (forces --gamma 0)")
    arch.add_argument("--share-future-encoder", action="store_true",
                      help="let the future branch reuse the historical feature extractor")
    arch.add_argument("--future-alpha-module", action="store_true",
                      help="push the future residual through the alpha module of Eq. 7 too")
    arch.add_argument("--ablation", action="store_true", help="run every variant of Table 2")

    opt = parser.add_argument_group("training")
    opt.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    opt.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    opt.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    opt.add_argument("--seed", type=int, default=DEFAULT_SEED)
    opt.add_argument("--gamma", type=float, default=0.1, help="the weight of the contrastive loss")
    opt.add_argument("--temperature", type=float, default=0.1, help="tau")
    opt.add_argument("--grad-clip", type=float, default=None)
    opt.add_argument("--device", default="cpu")
    opt.add_argument("--early-stop-metric", default="ic", choices=("ic", "loss"))
    opt.add_argument("--quiet", action="store_true")

    roll = parser.add_argument_group("rolling protocol")
    roll.add_argument("--train-years", type=int, default=5)
    roll.add_argument("--valid-years", type=int, default=1)
    roll.add_argument("--test-years", type=int, default=2)
    roll.add_argument("--test-start", default=None)
    roll.add_argument("--test-end", default=None)

    sim = parser.add_argument_group("investment simulation")
    sim.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    sim.add_argument("--holding", type=int, default=DEFAULT_HOLDING,
                     help="the holding period, the article's Delta t = 10")
    sim.add_argument("--cost", type=float, default=DEFAULT_COST)
    sim.add_argument("--cost-mode", default="per_side", choices=("per_side", "round_trip"))
    sim.add_argument("--no-backtest", action="store_true")
    return parser


def configs_from_args(args, overrides=None):
    """The `ModelConfig`/`BaselineConfig` and `TrainConfig` an argument namespace asks for."""
    overrides = dict(overrides or {})
    gamma = overrides.pop("gamma", args.gamma)
    if args.model == "factorgcl":
        model_config = ModelConfig(
            hidden_size=args.hidden_size, num_hidden_factors=args.num_hidden_factors,
            num_rnn_layers=args.rnn_layers, dropout=args.dropout, bn_position=args.bn_position,
            use_prior=not args.no_prior, use_hidden=not args.no_hidden,
            use_alpha=not args.no_alpha,
            share_future_feature_extractor=args.share_future_encoder,
            future_alpha_module=args.future_alpha_module)
        for key, value in overrides.items():
            setattr(model_config, key, value)
        if not model_config.use_alpha:
            gamma = 0.0
    else:
        model_config = BaselineConfig(name=args.model, hidden_size=args.hidden_size,
                                      num_layers=args.rnn_layers, dropout=args.dropout,
                                      bn_position=args.bn_position, seq_len=args.seq_len)
        gamma = 0.0
    train_config = TrainConfig(learning_rate=args.lr, epochs=args.epochs, patience=args.patience,
                               seed=args.seed, gamma=gamma, temperature=args.temperature,
                               grad_clip=args.grad_clip, device=args.device,
                               early_stop_metric=args.early_stop_metric, verbose=not args.quiet)
    return model_config, train_config


def run_one(panel, splits, model_config, train_config, args, outdir, tag=""):
    """One full rolling run: train, predict, score and (optionally) backtest."""
    predictions, records = run_rolling(panel, splits, model_config, train_config)
    if predictions.empty:
        print(f"no window produced predictions{' for ' + tag if tag else ''}")
        return None
    suffix = f"_{tag}" if tag else ""
    predictions.to_csv(os.path.join(outdir, f"predictions{suffix}.csv"), index=False)
    metrics = evaluate(predictions, panel.config.periods)
    metrics.to_csv(os.path.join(outdir, f"metrics{suffix}.csv"))
    print(f"\n=== prediction metrics{' (' + tag + ')' if tag else ''} ===")
    print(metrics.to_string(float_format=lambda v: f"{v:.4f}"))

    for number, record in enumerate(records):
        history = record.pop("history", None)
        if history is not None:
            history.to_csv(os.path.join(outdir, f"history{suffix}_window{number}.csv"), index=False)
    pd.DataFrame(records).to_csv(os.path.join(outdir, f"splits{suffix}.csv"), index=False)

    if args.no_backtest:
        return metrics
    pred_col = f"pred_{args.holding}"
    if pred_col not in predictions.columns:
        pred_col = f"pred_{panel.config.periods[0]}"
        print(f"no prediction for a {args.holding}-day horizon; backtesting {pred_col}")
    frame = run_topk_backtest(predictions, price_frame_from_panel(panel), pred_col,
                              topk=args.topk, holding=args.holding, cost=args.cost,
                              cost_mode=args.cost_mode)
    frame.to_csv(os.path.join(outdir, f"backtest{suffix}.csv"))
    stats = backtest_metrics(frame)
    pd.Series(stats).to_csv(os.path.join(outdir, f"backtest_metrics{suffix}.csv"))
    print(f"=== TopK{args.topk} investment simulation, holding {args.holding} days"
          f"{' (' + tag + ')' if tag else ''} ===")
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in stats.items()))
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    raw_panel, industry = load_inputs(args)
    config = PreprocessConfig(seq_len=args.seq_len, future_len=args.future_len,
                              periods=tuple(args.periods), price_field=args.price_field,
                              max_missing_ratio=args.max_missing_ratio,
                              feature_clip=args.feature_clip,
                              label_mad_clip=args.label_mad_clip, future_norm=args.future_norm)
    panel = PanelData(raw_panel, industry, config)
    print(f"panel: {len(panel.dates)} trading days, {len(panel.stocks)} stocks, "
          f"{panel.num_factors} prior factors, periods {panel.config.periods}")

    splits = rolling_splits(panel.dates, train_years=args.train_years,
                            valid_years=args.valid_years, test_years=args.test_years,
                            test_start=args.test_start, test_end=args.test_end)
    if not splits:
        raise SystemExit("the panel is too short for the requested rolling protocol")
    for number, split in enumerate(splits):
        print(f"  window {number}: train {split.train[0].date()}~{split.train[1].date()}, "
              f"valid {split.valid[0].date()}~{split.valid[1].date()}, "
              f"test {split.test[0].date()}~{split.test[1].date()}")

    if args.ablation:
        if args.model != "factorgcl":
            raise SystemExit("--ablation applies to the FactorGCL model")
        summary = {}
        for name, overrides in ABLATIONS.items():
            print(f"\n########## {name} ##########")
            model_config, train_config = configs_from_args(args, overrides)
            metrics = run_one(panel, splits, model_config, train_config, args, args.outdir,
                              tag=name.replace(" ", "").replace("&", "-"))
            if metrics is not None:
                summary[name] = metrics["IC"]
        table = pd.DataFrame(summary).T.rename_axis("variant")
        table.columns = [f"IC_dt={c}" for c in table.columns]
        table.to_csv(os.path.join(args.outdir, "ablation.csv"))
        print("\n=== ablation study (IC) ===")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        return table

    model_config, train_config = configs_from_args(args)
    return run_one(panel, splits, model_config, train_config, args, args.outdir)


if __name__ == "__main__":
    main()
