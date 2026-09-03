"""
FactorGCL -- a hypergraph-based factor model with temporal residual contrastive learning.

An implementation of Duan, Wang and Li (2025), "FactorGCL: A Hypergraph-Based Factor Model
with Temporal Residual Contrastive Learning for Stock Returns Prediction", AAAI-25,
pages 173-181.

    from factorgcl import PanelData, PreprocessConfig, ModelConfig, TrainConfig
    from factorgcl import rolling_splits, run_rolling, evaluate, run_topk_backtest

See ``README.md`` for the mapping from the article's equations to the modules.

The modules import each other by plain name, as the scripts of this repository do, so that
``python run_factorgcl.py`` works from inside this directory; the line below puts the
directory on the import path so that ``import factorgcl`` works from the repository root
too.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import backtest_metrics, price_frame_from_panel, run_topk_backtest
from baselines import BaselineConfig
from data import PanelData, PreprocessConfig
from loss import factorgcl_loss, info_nce_loss, multi_period_mse
from metrics import ic_icir, investment_metrics, prediction_metrics
from model import (FactorGCL, FeatureExtractor, HiddenBetaModule, HyperGCNLayer,
                   IndividualAlphaModule, PriorBetaModule, ProjectionHead)
from train import (ModelConfig, TrainConfig, evaluate, predict, rolling_splits, run_rolling,
                   train_model)

__all__ = [
    "FactorGCL", "HyperGCNLayer", "FeatureExtractor", "PriorBetaModule", "HiddenBetaModule",
    "IndividualAlphaModule", "ProjectionHead",
    "info_nce_loss", "multi_period_mse", "factorgcl_loss",
    "PanelData", "PreprocessConfig",
    "ModelConfig", "TrainConfig", "train_model", "predict", "evaluate", "rolling_splits",
    "run_rolling", "BaselineConfig",
    "ic_icir", "prediction_metrics", "investment_metrics",
    "run_topk_backtest", "backtest_metrics", "price_frame_from_panel",
]
