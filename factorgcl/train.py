"""
Training, prediction and the rolling protocol of the experiments.

The article trains one model per rolling window: "we follow the temporal order to split the
dataset into training set, validation set and test set, where the time length is
5 years : 1 year : 2 years, and adopt a rolling method for training and testing", with the
overall test period running from 01/01/2020 to 06/30/2023. `rolling_splits` reproduces that
schedule; `train_model` trains one window with the settings of the supplement -- Adam at a
learning rate of 1e-3, 100 epochs, early stopping after 20 steps, seed 0.

One optimisation step is one trading day: the hypergraph is rebuilt for every cross-section,
so a "batch" is the whole cross-section of a day and the days are shuffled within an epoch.
"""

from dataclasses import dataclass
import copy
import random

import numpy as np
import pandas as pd
import torch

from loss import DEFAULT_GAMMA, DEFAULT_TEMPERATURE, factorgcl_loss
from metrics import ic_icir, prediction_metrics
from model import (DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_HIDDEN_FACTORS, DEFAULT_NUM_RNN_LAYERS,
                   FactorGCL)

# the settings of the supplementary material ("Implementation Details")
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 20
DEFAULT_SEED = 0

EARLY_STOP_METRICS = ("ic", "loss")


@dataclass
class ModelConfig:
    """The architecture hyper-parameters, with the article's values as defaults."""

    hidden_size: int = DEFAULT_HIDDEN_SIZE                    # H
    num_hidden_factors: int = DEFAULT_NUM_HIDDEN_FACTORS      # M
    num_rnn_layers: int = DEFAULT_NUM_RNN_LAYERS              # L of the GRU
    dropout: float = 0.0
    bn_position: str = "input"
    use_prior: bool = True
    use_hidden: bool = True
    use_alpha: bool = True
    share_future_feature_extractor: bool = False
    future_alpha_module: bool = False

    def build(self, input_size, num_prior_factors, num_periods):
        return FactorGCL(input_size=input_size, num_prior_factors=num_prior_factors,
                         hidden_size=self.hidden_size,
                         num_hidden_factors=self.num_hidden_factors,
                         num_rnn_layers=self.num_rnn_layers, num_periods=num_periods,
                         dropout=self.dropout, bn_position=self.bn_position,
                         use_prior=self.use_prior, use_hidden=self.use_hidden,
                         use_alpha=self.use_alpha,
                         share_future_feature_extractor=self.share_future_feature_extractor,
                         future_alpha_module=self.future_alpha_module)


@dataclass
class TrainConfig:
    """The optimisation hyper-parameters, with the supplement's values as defaults."""

    learning_rate: float = DEFAULT_LEARNING_RATE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    seed: int = DEFAULT_SEED
    gamma: float = DEFAULT_GAMMA              # the weight of the contrastive loss
    temperature: float = DEFAULT_TEMPERATURE  # tau
    weight_decay: float = 0.0
    grad_clip: float = None
    device: str = "cpu"
    early_stop_metric: str = "ic"             # the validation quantity to watch
    verbose: bool = True

    def __post_init__(self):
        if self.early_stop_metric not in EARLY_STOP_METRICS:
            raise ValueError(f"early_stop_metric must be one of {EARLY_STOP_METRICS}")


@dataclass
class Split:
    """One rolling window, as half-open date ranges ``[start, end]`` (both inclusive)."""

    train: tuple
    valid: tuple
    test: tuple

    def as_dict(self):
        return {"train_start": self.train[0], "train_end": self.train[1],
                "valid_start": self.valid[0], "valid_end": self.valid[1],
                "test_start": self.test[0], "test_end": self.test[1]}


def set_seed(seed=DEFAULT_SEED):
    """Seed every source of randomness the training loop draws on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rolling_splits(dates, train_years=5, valid_years=1, test_years=2,
                   test_start=None, test_end=None, step_years=None):
    """The rolling train/validation/test schedule of the experiments.

    Windows are laid out on the calendar rather than on trading-day counts, and the test
    windows tile the requested test period without overlapping: the first one starts at
    ``test_start`` (by default as early as the data allows) and each next one starts where
    the previous ended, so the last window is truncated at ``test_end`` -- which is how the
    article's 3.5-year test period is covered by 2-year windows.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).sort_values()
    if len(dates) == 0:
        return []
    day = pd.Timedelta(days=1)
    step_years = test_years if step_years is None else step_years
    first_possible = dates[0] + pd.DateOffset(years=train_years + valid_years)
    test_start = first_possible if test_start is None else pd.Timestamp(test_start)
    test_start = max(test_start, first_possible)
    test_end = dates[-1] if test_end is None else pd.Timestamp(test_end)
    test_end = min(test_end, dates[-1])

    splits = []
    current = test_start
    while current <= test_end:
        window_end = min(current + pd.DateOffset(years=step_years) - day, test_end)
        valid_end = current - day
        valid_start = valid_end - pd.DateOffset(years=valid_years) + day
        train_end = valid_start - day
        train_start = train_end - pd.DateOffset(years=train_years) + day
        if train_start < dates[0]:
            train_start = dates[0]
        splits.append(Split(train=(train_start, train_end), valid=(valid_start, valid_end),
                            test=(current, window_end)))
        current = window_end + day
    return splits


def indices_in_range(panel, start, end, need_future=False, need_label=True, embargo=0):
    """The anchor-day indices of ``panel`` that fall in ``[start, end]`` and carry a full
    window.

    ``embargo`` drops the last few days of the range. A training day looks ``max(period) + 1``
    days into the future for its labels and ``T'`` days for its contrastive branch, so
    without an embargo the tail of the training range would overlap the validation range;
    `run_rolling` sets it from the panel's configuration.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(panel.dates)))
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    usable = panel.usable_date_indices(need_future=need_future, need_label=need_label)
    selected = [i for i in usable if start <= dates[i] <= end]
    if embargo > 0 and selected:
        # the last position of the range, minus the days whose labels would reach past it
        end_position = int(np.searchsorted(dates.values, np.datetime64(end), side="right")) - 1
        selected = [i for i in selected if i <= end_position - embargo]
    return selected


def _to_tensor(array, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)


def _run_day(model, sample, device, with_future):
    """Forward one cross-section; returns the historical output dict and the future one."""
    x = _to_tensor(sample.x, device)
    beta = _to_tensor(sample.beta, device)
    out = model(x, beta)
    future_out = None
    if with_future and sample.x_future is not None and model.use_alpha:
        future_out = model.forward_future(_to_tensor(sample.x_future, device), beta,
                                          out["beta_hidden"])
    return out, future_out


@torch.no_grad()
def predict(model, panel, indices, device="cpu", periods=None):
    """Predictions over the given anchor days, as a long-format frame.

    Columns: ``date``, ``stock``, ``pred_{dt}``, ``label_{dt}`` and ``raw_label_{dt}`` for
    every forward period ``dt``.
    """
    model.eval()
    periods = panel.config.periods if periods is None else periods
    frames = []
    for index in indices:
        sample = panel.cross_section(index, with_future=False, need_label=False)
        if sample is None:
            continue
        out, _ = _run_day(model, sample, device, with_future=False)
        y_hat = out["y_hat"].detach().cpu().numpy()
        frame = {"date": sample.date, "stock": sample.stocks}
        for k, dt in enumerate(periods):
            frame[f"pred_{dt}"] = y_hat[:, k]
            label = np.where(sample.label_mask[:, k], sample.label[:, k], np.nan)
            frame[f"label_{dt}"] = label
            frame[f"raw_label_{dt}"] = sample.raw_label[:, k]
        frames.append(pd.DataFrame(frame))
    if not frames:
        return pd.DataFrame(columns=["date", "stock"])
    return pd.concat(frames, ignore_index=True)


def evaluate(prediction_frame, periods):
    """``IC``, ``ICIR``, ``MSE`` and ``F1`` per forward prediction period."""
    rows = {}
    for dt in periods:
        columns = ["date", f"label_{dt}", f"pred_{dt}"]
        if not set(columns) <= set(prediction_frame.columns):
            continue
        frame = prediction_frame[columns].dropna()
        frame = frame.rename(columns={f"label_{dt}": "label", f"pred_{dt}": "pred"})
        rows[dt] = prediction_metrics(frame)
    return pd.DataFrame(rows).T.rename_axis("period")


@torch.no_grad()
def _validation_score(model, panel, indices, device, train_config, periods):
    """The quantity early stopping watches: the mean validation IC over the forward periods
    (higher is better), or the negated validation loss."""
    model.eval()
    if train_config.early_stop_metric == "loss":
        total, count = 0.0, 0
        for index in indices:
            sample = panel.cross_section(index, with_future=False, need_label=True)
            if sample is None:
                continue
            out, _ = _run_day(model, sample, device, with_future=False)
            loss, _ = factorgcl_loss(out["y_hat"], _to_tensor(sample.label, device), gamma=0.0,
                                     mask=_to_tensor(sample.label_mask, device, torch.bool))
            total += float(loss.detach())
            count += 1
        return float("nan") if count == 0 else -total / count
    frame = predict(model, panel, indices, device, periods)
    if frame.empty:
        return float("nan")
    scores = []
    for dt in periods:
        sub = frame[["date", f"label_{dt}", f"pred_{dt}"]].dropna()
        sub = sub.rename(columns={f"label_{dt}": "label", f"pred_{dt}": "pred"})
        ic, _ = ic_icir(sub)
        if np.isfinite(ic):
            scores.append(ic)
    return float(np.mean(scores)) if scores else float("nan")


def train_model(panel, train_indices, valid_indices, model_config=None, train_config=None,
                model=None):
    """Train one model on ``train_indices`` and early-stop on ``valid_indices``.

    Returns ``(model, history)`` with the model restored to the parameters of the best
    validation step, as early stopping requires.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    if train_config.gamma != 0 and not model_config.use_alpha:
        raise ValueError("the contrastive loss needs the alpha embeddings: set gamma to 0 "
                         "together with use_alpha=False (the article's '-wo Alpha&CL')")
    set_seed(train_config.seed)
    device = torch.device(train_config.device)
    if model is None:
        model = model_config.build(panel.input_size, panel.num_factors, panel.num_periods)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate,
                                 weight_decay=train_config.weight_decay)
    use_contrastive = train_config.gamma != 0 and model.use_alpha

    best_score, best_state, best_epoch, history = -np.inf, None, -1, []
    train_indices = list(train_indices)
    for epoch in range(train_config.epochs):
        model.train()
        order = list(train_indices)
        random.shuffle(order)
        epoch_loss, epoch_mse, epoch_cl, steps = 0.0, 0.0, 0.0, 0
        for index in order:
            sample = panel.cross_section(index, with_future=use_contrastive, need_label=True)
            if sample is None:
                continue
            out, future_out = _run_day(model, sample, device, with_future=use_contrastive)
            loss, parts = factorgcl_loss(
                out["y_hat"], _to_tensor(sample.label, device),
                alpha_past=out["alpha"],
                alpha_future=None if future_out is None else future_out["alpha"],
                projection=model.projection, gamma=train_config.gamma,
                temperature=train_config.temperature,
                mask=_to_tensor(sample.label_mask, device, torch.bool))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_mse += float(parts["mse"].detach())
            epoch_cl += float(parts["contrastive"].detach())
            steps += 1
        if steps == 0:
            raise ValueError("no usable training day in this window")
        score = _validation_score(model, panel, valid_indices, device, train_config,
                                  panel.config.periods)
        history.append({"epoch": epoch, "loss": epoch_loss / steps, "mse": epoch_mse / steps,
                        "contrastive": epoch_cl / steps, "valid_score": score})
        if train_config.verbose:
            print(f"  epoch {epoch:3d}  loss {epoch_loss / steps:.5f}  "
                  f"mse {epoch_mse / steps:.5f}  cl {epoch_cl / steps:.5f}  "
                  f"valid {score:.5f}", flush=True)
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= train_config.patience:
            if train_config.verbose:
                print(f"  early stopping at epoch {epoch} (best {best_epoch})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def run_rolling(panel, splits, model_config=None, train_config=None, embargo=None):
    """Train and predict over every rolling window, concatenating the out-of-sample frames.

    Returns ``(predictions, records)``: the long-format prediction frame over the whole test
    period, and one record per window with its dates, its training history and the number of
    days it used.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainConfig()
    if embargo is None:
        embargo = max(max(panel.config.periods) + 1, panel.config.future_len)
    frames, records = [], []
    for number, split in enumerate(splits):
        train_indices = indices_in_range(panel, *split.train, need_future=True, need_label=True,
                                         embargo=embargo)
        valid_indices = indices_in_range(panel, *split.valid, need_future=False, need_label=True,
                                         embargo=embargo)
        test_indices = indices_in_range(panel, *split.test, need_future=False, need_label=False)
        if not train_indices or not valid_indices or not test_indices:
            records.append({**split.as_dict(), "skipped": True, "n_train": len(train_indices),
                            "n_valid": len(valid_indices), "n_test": len(test_indices)})
            continue
        if train_config.verbose:
            print(f"[window {number}] train {split.train[0].date()}~{split.train[1].date()} "
                  f"({len(train_indices)} days), valid {split.valid[0].date()}~"
                  f"{split.valid[1].date()} ({len(valid_indices)}), test "
                  f"{split.test[0].date()}~{split.test[1].date()} ({len(test_indices)})",
                  flush=True)
        model, history = train_model(panel, train_indices, valid_indices, model_config,
                                     train_config)
        frame = predict(model, panel, test_indices, train_config.device, panel.config.periods)
        frames.append(frame)
        records.append({**split.as_dict(), "skipped": False, "n_train": len(train_indices),
                        "n_valid": len(valid_indices), "n_test": len(test_indices),
                        "best_valid_score": float(history["valid_score"].max()),
                        "epochs_run": int(len(history)), "history": history})
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return predictions, records
