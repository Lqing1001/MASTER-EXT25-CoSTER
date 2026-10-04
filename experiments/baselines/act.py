"""ACT adaptation: dynamic relations with temporal decomposition and isolated fluctuation/shock branches."""

from __future__ import annotations

import argparse

import copy

import json

import math

import os

import random

import time

from datetime import datetime

from pathlib import Path

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from coster.runtime import (
    DATASET_ROOT,
    ROOT, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VALID_END, VALID_START,
    UniverseStore, drop_extreme_and_zscore, evaluate, metrics_by_period,
    summarize_daily, write_daily_csv,
)

def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)

def causal_average(x: torch.Tensor, window: int) -> torch.Tensor:
    outputs = []
    for t in range(x.shape[1]):
        outputs.append(x[:, max(0, t - window + 1):t + 1].mean(dim=1))
    return torch.stack(outputs, dim=1)

class ACTCompat(nn.Module):
    """Dynamic-relation ACT screen; static industry/region PSPE is unavailable."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.trend_proj = nn.Sequential(nn.Linear(158, d_model), nn.LayerNorm(d_model))
        self.dynamic_graph = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.backward = nn.Linear(d_model, d_model)
        self.trend_fuse = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LeakyReLU())
        self.fluct_tcn = nn.Sequential(
            nn.Conv1d(158, d_model, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.GELU(),
        )
        self.shock_proj = nn.Sequential(
            nn.Linear(158 * 2, d_model), nn.LeakyReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.component_score = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock = x[:, :, :158]
        trend = causal_average(stock, 4)
        detrended = stock - trend
        fluct = causal_average(detrended, 2)
        shock = stock - trend - fluct

        base = self.trend_proj(trend[:, -1])
        dynamic, _ = self.dynamic_graph(base.unsqueeze(0), base.unsqueeze(0), base.unsqueeze(0), need_weights=False)
        dynamic = dynamic.squeeze(0)
        purified = base - self.backward(dynamic)
        trend_z = self.trend_fuse(torch.cat([dynamic, purified], dim=-1))

        fluct_z = self.fluct_tcn(fluct.transpose(1, 2))[:, :, -1]
        shock_cf = causal_average(shock, 3)[:, -1]
        shock_z = self.shock_proj(torch.cat([shock[:, -1], shock_cf], dim=-1))
        components = torch.stack([trend_z, fluct_z, shock_z], dim=1)
        weights = torch.softmax(self.component_score(components).squeeze(-1), dim=1)
        return self.head((weights.unsqueeze(-1) * components).sum(1)).squeeze(-1)

def model_loss(model_name, pred, target):
    return pearson_loss(pred, target) + 0.1 * F.mse_loss(pred, target)

def validation_score(model: nn.Module, store: UniverseStore, device: torch.device, max_eval_days: int | None):
    rows, _ = evaluate(model, store, VALID_START, VALID_END, device, max_eval_days)
    metrics = summarize_daily(rows)
    return metrics, 0.5 * (metrics["IC"] + metrics["RankIC"])

def train_epoch(model_name, model, store, dates, optimizer, device, max_train_days):
    model.train()
    losses = []
    if max_train_days is not None:
        dates = dates[:max_train_days]
    for number, day in enumerate(dates, 1):
        x, y, _ = store.batch(int(day), training=True)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        keep, target = drop_extreme_and_zscore(y)
        pred = model(x[keep])
        loss = model_loss(model_name, pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if number % 250 == 0 or number == len(dates):
            log(f"{model_name} train {number}/{len(dates)} loss={np.mean(losses):.6f}")
    return float(np.mean(losses))

def build_model(name, device):
    return ACTCompat().to(device), None

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('act',), default='act')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--stage1-epochs', type=int, default=4)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'neural_baselines')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    model, spatial = build_model(args.model, device)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    rng = np.random.RandomState(args.seed)
    stage1_history = []
    if spatial is not None:
        pass
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lr = 0.0001
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0001)
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        order = train_dates.copy()
        rng.shuffle(order)
        train_loss = train_epoch(args.model, model, store, order, optimizer, device, args.max_train_days)
        valid, score = validation_score(model, store, device, args.max_eval_days)
        improved = score > best_score
        if improved:
            best_score, best_epoch, stale = (score, epoch, 0)
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation': valid, 'selection_score': score, 'best': improved})
        log(f"{args.model} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    prefix = f'{args.model}_{args.universe}_seed{args.seed}'
    torch.save(model.state_dict(), args.output_dir / f'{prefix}_best.pt')
    rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    result = {'source': 'local compatible implementation; not official-code reproduction', 'model': args.model, 'universe': args.universe, 'seed': args.seed, 'best_epoch': best_epoch, 'stage1_history': stage1_history, 'history': history, 'metrics': metrics_by_period(rows), 'parameters': sum((p.numel() for p in model.parameters())), 'trainable_parameters_stage2': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'adaptations': 'Dynamic relations; industry and region graphs are not used', 'config': vars(args) | {'output_dir': str(args.output_dir.resolve())}}
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_daily_csv(args.output_dir / f'{prefix}_daily.csv', rows)
    np.savez_compressed(args.output_dir / f'{prefix}_predictions.npz', **predictions)
    log(json.dumps({'model': args.model, 'best_epoch': best_epoch, 'metrics': result['metrics']}, ensure_ascii=False))

if __name__ == "__main__":
    main()
