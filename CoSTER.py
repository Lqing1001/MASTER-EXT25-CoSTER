"""CoSTER: continuous and sparse temporal expert ranker (standalone source).

独立文件：只需 Python 3.10+、NumPy、PyTorch（论文环境 PyTorch 2.8）。
时序排序器含8个可学习位置向量。
检查点文件名与加载接口保持一致。
包含 MASTER、连续编码器、混合排序器、时序残差、Top-1残差、训练与推断。
不包含 CSMAR 原始数据、数据构建工具、已训练权重和组合回测。

快速检查（CPU；随机输入/随机权重，不代表论文性能）：
    python CoSTER.py demo

Python 接口（每次输入是一个交易日的完整股票横截面）：
    model = CoSTER(universe="csi300").eval()
    scores = model(x)  # x: float32 Tensor[N, 8, 221]
    # 前158维为Alpha158，后63维为Market63；必须已按训练期统计量预处理。
    # 不得将不同交易日混在N维，也不得按未来标签有效性筛选推断股票。
    # 新建模型权重随机；真实预测请使用 CoSTER.from_checkpoints(...)。

训练（原始训练入口要求CUDA；每阶段独立运行，同一seed）：
    python CoSTER.py train-master --dataset-root DATA --universe csi300 --seed 0 --output-dir RUN/master
    python CoSTER.py train-continuous --dataset-root DATA --universe csi300 --seed 0 --output-dir RUN/continuous_prism
    python CoSTER.py train-sparse --dataset-root DATA --universe csi300 --seed 0 --continuous-dir RUN/continuous_prism --output-dir RUN/sparse_top1

数据目录：DATA/master_input/csi300/by_year/year=YYYY/{features,labels,dates,instruments}.npy
features为[M,221]预处理特征；labels为[M]原始标签；dates为YYYYMMDD整数；
instruments按日排序，日记录连续。CSI800目录结构相同。保留最终数据日历校验3886天。
训练20100104--20191224；验证20200102--20211224；测试20220104--20251231。
MASTER训练40轮；连续编码器4轮、基础排序器12轮、时序残差10轮、稀疏残差10轮。
固定训练期预处理常数必须由外部数据准备流程提供；本文件不会重新拟合它们。

整套测试/验证推断（--run-dir目录包含上述三个阶段输出目录）：
    python CoSTER.py infer --dataset-root DATA --run-dir RUN --universe csi300 --seed 0 --split test --output-dir OUTPUT

本版需包含time_position的最终检查点；如各阶段来自不同目录，请在Python中使用
CoSTER.from_checkpoints(master_path, base_path, temporal_path, sparse_path, universe).
模型不支持对组合输出直接端到端重训：请使用上述分阶段训练入口。
保持原程序冻结和dropout模式行为，不为重构而改变训练方法。
稀疏Top-1无直接任务梯度传给硬选择打分器；未使用代码本量化。
编码器中保留未参与推断的codebook/重建头，以兼容checkpoint及论文参数量。
标准化使用原实现max(population_std, 1e-12)，在float64中执行，固定等权。
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

ROOT = Path(__file__).resolve().parent / "CoSTER_runs"
DATASET_ROOT = Path(os.environ.get("MASTER_EXT_DATASET_ROOT", "datasets/master_ext_strict_20191224_v1")).resolve()
TRAIN_START, TRAIN_END = 20100104, 20191224
VALID_START, VALID_END = 20200102, 20211224
TEST_START, TEST_END = 20220104, 20251231
PAPER_TEST_END, EXTENSION_START = 20231231, 20240101
STEP_LEN, FEATURES = 8, 221



# MIT License
# 
# Copyright (c) 2026 MASTER-EXT25-DynaFuse contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# MIT License
# 
# Copyright (c) 2025 Data Management Technology and AI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# MASTER components adapted from https://github.com/SJTU-DMTai/MASTER


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)
        self.norm1 = LayerNorm(d_model, eps=1e-05)
        self.norm2 = LayerNorm(d_model, eps=1e-05)
        self.ffn = nn.Sequential(Linear(d_model, d_model), nn.ReLU(), Dropout(p=dropout), Linear(d_model, d_model), Dropout(p=dropout))

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output


class TAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)
        self.norm1 = LayerNorm(d_model, eps=1e-05)
        self.norm2 = LayerNorm(d_model, eps=1e-05)
        self.ffn = nn.Sequential(Linear(d_model, d_model), nn.ReLU(), Dropout(p=dropout), Linear(d_model, d_model), Dropout(p=dropout))

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)
        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)
        return att_output


class Gate(nn.Module):

    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)
        return output


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)


class MasterTemporalResidual(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(158, d_model)
        self.temporal = TAttention(d_model=d_model, nhead=2, dropout=dropout)
        self.pool = TemporalAttention(d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, stock: torch.Tensor) -> torch.Tensor:
        z = self.temporal(self.input_proj(stock))
        return self.head(self.pool(z)).squeeze(-1)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class DayLocation:
    year: int
    start: int
    end: int


class UniverseStore:
    """Memory-mapped daily cross sections with Qlib-equivalent TS padding."""

    def __init__(self, dataset_root: Path, universe: str):
        self.universe = universe
        self.base = dataset_root / 'master_input' / universe / 'by_year'
        self.arrays: dict[int, dict[str, np.ndarray]] = {}
        self.day_locations: dict[int, DayLocation] = {}
        for year_dir in sorted(self.base.glob('year=*')):
            year = int(year_dir.name.split('=')[1])
            arrays = {'features': np.load(year_dir / 'features.npy', mmap_mode='r'), 'labels': np.load(year_dir / 'labels.npy', mmap_mode='r'), 'dates': np.load(year_dir / 'dates.npy', mmap_mode='r'), 'instruments': np.load(year_dir / 'instruments.npy', mmap_mode='r')}
            if arrays['features'].shape[1] != FEATURES:
                raise ValueError(f'{year}: expected {FEATURES} features')
            self.arrays[year] = arrays
            dates = arrays['dates']
            unique, starts, counts = np.unique(dates, return_index=True, return_counts=True)
            for day, start, count in zip(unique, starts, counts):
                self.day_locations[int(day)] = DayLocation(year, int(start), int(start + count))
        self.calendar = np.array(sorted(self.day_locations), dtype=np.int32)
        self.calendar_pos = {int(day): pos for pos, day in enumerate(self.calendar)}
        if len(self.calendar) != 3886:
            raise ValueError(f'unexpected calendar length: {len(self.calendar)}')

    def dates_between(self, start: int, end: int) -> np.ndarray:
        mask = (self.calendar >= start) & (self.calendar <= end)
        return self.calendar[mask]

    def _day_arrays(self, day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        location = self.day_locations[int(day)]
        arrays = self.arrays[location.year]
        slc = slice(location.start, location.end)
        return (arrays['features'][slc], arrays['labels'][slc], arrays['instruments'][slc])

    def batch(self, day: int, training: bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current_x, current_y, current_inst = self._day_arrays(day)
        if training:
            current_keep = np.isfinite(current_y)
            current_y = np.asarray(current_y[current_keep], dtype=np.float32)
            current_inst = np.asarray(current_inst[current_keep])
        else:
            current_y = np.asarray(current_y, dtype=np.float32)
            current_inst = np.asarray(current_inst)
        n = len(current_inst)
        sequence = np.empty((n, STEP_LEN, FEATURES), dtype=np.float32)
        sequence.fill(np.nan)
        valid = np.zeros((n, STEP_LEN), dtype=bool)
        current_pos = self.calendar_pos[int(day)]
        first_pos = max(0, current_pos - STEP_LEN + 1)
        history = self.calendar[first_pos:current_pos + 1]
        offset = STEP_LEN - len(history)
        for slot, history_day in enumerate(history, start=offset):
            day_x, day_y, day_inst = self._day_arrays(int(history_day))
            if training:
                source_keep = np.isfinite(day_y)
                source_inst = np.asarray(day_inst[source_keep])
                source_x = day_x[source_keep]
            else:
                source_inst = np.asarray(day_inst)
                source_x = day_x
            positions = np.searchsorted(source_inst, current_inst)
            bounded = positions < len(source_inst)
            matched = np.zeros(n, dtype=bool)
            matched[bounded] = source_inst[positions[bounded]] == current_inst[bounded]
            if matched.any():
                sequence[matched, slot, :] = source_x[positions[matched]]
                valid[matched, slot] = True
        for slot in range(1, STEP_LEN):
            missing = ~valid[:, slot] & valid[:, slot - 1]
            sequence[missing, slot, :] = sequence[missing, slot - 1, :]
            valid[missing, slot] = True
        for slot in range(STEP_LEN - 2, -1, -1):
            missing = ~valid[:, slot] & valid[:, slot + 1]
            sequence[missing, slot, :] = sequence[missing, slot + 1, :]
            valid[missing, slot] = True
        if not valid.all() or not np.isfinite(sequence).all():
            raise ValueError(f'non-finite sequence for {self.universe} {day}')
        return (sequence, current_y, current_inst)


def drop_extreme_and_zscore(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Official DropNA -> DropExtremeLabel -> CSZScoreNorm sequence."""
    finite = torch.isfinite(labels)
    original_indices = torch.nonzero(finite, as_tuple=False).squeeze(1)
    labels = labels[finite]
    count = int(0.025 * labels.shape[0])
    if count > 0:
        order = torch.argsort(labels)
        keep_local = order[count:-count]
        original_indices = original_indices[keep_local]
        labels = labels[keep_local]
    labels = (labels - labels.mean()) / labels.std()
    return (original_indices, labels)


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(pred: np.ndarray, label: np.ndarray, rank: bool=False) -> float:
    mask = np.isfinite(pred) & np.isfinite(label)
    if mask.sum() < 3:
        return math.nan
    x = pred[mask].astype(np.float64)
    y = label[mask].astype(np.float64)
    if rank:
        x = rank_average(x)
        y = rank_average(y)
    x -= x.mean()
    y -= y.mean()
    denom = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / denom) if denom > 0 else math.nan


def summarize_daily(rows: list[dict]) -> dict:
    ic = np.array([row['IC'] for row in rows], dtype=np.float64)
    ric = np.array([row['RankIC'] for row in rows], dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    ric = ric[np.isfinite(ric)]
    return {'days': len(rows), 'valid_ic_days': int(len(ic)), 'IC': float(ic.mean()), 'IC_std': float(ic.std()), 'ICIR': float(ic.mean() / ic.std()), 'RankIC': float(ric.mean()), 'RankIC_std': float(ric.std()), 'RankICIR': float(ric.mean() / ric.std())}


def evaluate(model: torch.nn.Module, store: UniverseStore, start: int, end: int, device: torch.device, max_days: int | None=None) -> tuple[list[dict], dict[str, np.ndarray]]:
    dates = store.dates_between(start, end)
    if max_days is not None:
        dates = dates[:max_days]
    daily_rows: list[dict] = []
    pred_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    date_chunks: list[np.ndarray] = []
    inst_chunks: list[np.ndarray] = []
    model.eval()
    for number, day in enumerate(dates, 1):
        x, y, instruments = store.batch(int(day), training=False)
        with torch.inference_mode():
            pred = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)
        ic = correlation(pred, y)
        rank_ic = correlation(pred, y, rank=True)
        daily_rows.append({'date': int(day), 'n': int(len(y)), 'finite_labels': int(np.isfinite(y).sum()), 'IC': ic, 'RankIC': rank_ic})
        pred_chunks.append(pred.astype(np.float32))
        label_chunks.append(y.astype(np.float32))
        date_chunks.append(np.full(len(y), int(day), dtype=np.int32))
        inst_chunks.append(instruments.astype('S8'))
        if number % 100 == 0 or number == len(dates):
            log(f'evaluate {store.universe}: {number}/{len(dates)} days')
    predictions = {'dates': np.concatenate(date_chunks), 'instruments': np.concatenate(inst_chunks), 'predictions': np.concatenate(pred_chunks), 'labels': np.concatenate(label_chunks)}
    return (daily_rows, predictions)


def write_daily_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['date', 'n', 'finite_labels', 'IC', 'RankIC'])
        writer.writeheader()
        writer.writerows(rows)


def metrics_by_period(daily_rows):
    periods = {'early_test_20220104_20231231':(TEST_START,20231231), 'extension_20240101_20251231':(20240101,TEST_END), 'all_20220104_20251231':(TEST_START,TEST_END)}
    periods.update({str(y):(y*10000+101,y*10000+1231) for y in range(2022,2026)})
    return {k:summarize_daily([r for r in daily_rows if a<=r['date']<=b]) for k,(a,b) in periods.items() if any(a<=r['date']<=b for r in daily_rows)}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class MasterExpert(nn.Module):

    def __init__(self, variant: str, universe: str, d_model: int=256):
        super().__init__()
        self.variant = variant
        self.use_gate = True
        self.use_temporal = True
        self.use_spatial = True
        self.mean_pool = False
        if d_model % 4:
            raise ValueError('d_model must be divisible by the four temporal heads')
        beta = 5 if universe == 'csi300' else 2
        self.feature_gate = Gate(63, 158, beta=beta)
        self.projection = nn.Linear(158, d_model)
        self.position = PositionalEncoding(d_model)
        self.temporal = TAttention(d_model=d_model, nhead=4, dropout=0.5)
        self.spatial = SAttention(d_model=d_model, nhead=2, dropout=0.5)
        if not self.mean_pool:
            self.pool = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stock = x[:, :, :158]
        gate = self.feature_gate(x[:, -1, 158:221])
        stock = stock * gate.unsqueeze(1)
        z = self.position(self.projection(stock))
        z = self.temporal(z)
        z = self.spatial(z)
        z = self.pool(z)
        return self.decoder(z).squeeze(-1)


def train_one(args, store, model, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dates = store.dates_between(TRAIN_START, TRAIN_END)
    if args.max_train_days is not None:
        dates = dates[:args.max_train_days]
    rng = np.random.RandomState(args.seed)
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = dates.copy()
        rng.shuffle(order)
        losses = []
        started = time.time()
        for number, day in enumerate(order, 1):
            x, y, _ = store.batch(int(day), training=True)
            feature = torch.from_numpy(x).to(device)
            labels = torch.from_numpy(np.array(y, copy=True)).to(device)
            keep, normalized = drop_extreme_and_zscore(labels)
            pred = model(feature[keep])
            loss = torch.mean((pred - normalized) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at {day}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(order):
                log(f'{args.variant} epoch={epoch} train {number}/{len(order)} loss={np.mean(losses):.6f}')
        rows, _ = evaluate(model, store, VALID_START, VALID_END, device, args.max_eval_days)
        valid = summarize_daily(rows)
        score = 0.5 * (valid['IC'] + valid['RankIC'])
        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': valid, 'selection_score': score, 'best': improved, 'elapsed_seconds': time.time() - started})
        log(f"{args.variant} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    return (history, best_epoch)


class ContinuousEncoder(nn.Module):

    def __init__(self, variant: str, d_model: int=64, codebook_size: int=512):
        super().__init__()
        self.variant = variant
        self.use_vq = False
        self.use_cross_stock = True
        self.use_market_prior = True
        self.gru = nn.GRU(158, d_model, batch_first=True)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.cross = nn.TransformerEncoder(layer, 2)
        # Registered only for published checkpoint/parameter-count compatibility; never used in forward.
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.normal_(self.codebook.weight, std=0.1)
        self.decoder = nn.Sequential(nn.Linear(d_model + 64, 256), nn.GELU(), nn.Linear(256, 158))
        self.aux = nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1))
        self.prior = nn.Linear(63, 64)

    def encode(self, x: torch.Tensor):
        stock, market = (x[:, :, :158], x[:, :, 158:221])
        mean = stock.mean(1, keepdim=True)
        std = stock.std(1, keepdim=True).clamp_min(1e-05)
        norm = (stock - mean) / std
        h = self.gru(norm)[1][-1]
        z = self.cross(h.unsqueeze(0)).squeeze(0)
        indices = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
        quant = z
        straight = z
        prior = self.prior(market[:, -1])
        return (z, quant, straight, indices, prior)

    def forward(self, x: torch.Tensor):
        z, quant, straight, indices, prior = self.encode(x)
        joined = torch.cat([straight, prior], dim=-1)
        reconstruction = self.decoder(joined)
        aux = self.aux(joined).squeeze(-1)
        return (z, quant, indices, reconstruction, aux)


class MixtureRanker(nn.Module):

    def __init__(self, spatial: ContinuousEncoder, variant: str, d_model: int=64, experts: int=2):
        super().__init__()
        self.spatial = spatial
        self.variant = variant
        self.use_temporal_encoder = True
        self.use_conditioned_gate = True
        self.input_proj = nn.Linear(158, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 2, 0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.gate = nn.Linear(d_model, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model + 64, 128), nn.GELU(), nn.Linear(128, 1)) for _ in range(experts)])
        with torch.random.fork_rng(devices=[]):
            self.time_position = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, state, _, _, prior = self.spatial.encode(x)
        seq = self.input_proj(x[:, :, :158]) + self.time_position[:, :x.shape[1]]
        context = self.temporal(torch.cat([state.unsqueeze(1), seq], dim=1))[:, 0]
        weights = torch.softmax(self.gate(state), dim=-1)
        joined = torch.cat([context, prior], dim=-1)
        outputs = torch.cat([expert(joined) for expert in self.experts], dim=1)
        return (weights * outputs).sum(1)


class TemporalAdapter(nn.Module):

    def __init__(self, base: MixtureRanker):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = MasterTemporalResidual()
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        with torch.no_grad():
            base_pred = self.base(x)
        residual = self.residual(x[:, :, :158])
        return base_pred + torch.tanh(self.residual_scale) * residual


def stage1_loss(spatial: ContinuousEncoder, variant: str, x: torch.Tensor, target: torch.Tensor):
    _, _, _, reconstruction, aux = spatial(x)
    terms = {'reconstruction': F.mse_loss(reconstruction, x[:, -1, :158]), 'aux_return': 0.0001 * F.mse_loss(aux, target)}
    return (sum(terms.values()), {name: float(value.detach().cpu()) for name, value in terms.items()})


def evaluate_for_selection(model, store, device, max_eval_days):
    rows, _ = evaluate(model, store, VALID_START, VALID_END, device, max_eval_days)
    metrics = summarize_daily(rows)
    return (metrics, 0.5 * (metrics['IC'] + metrics['RankIC']))


def train_spatial(spatial, variant, store, dates, optimizer, device, epochs, max_train_days):
    if max_train_days is not None:
        dates = dates[:max_train_days]
    history = []
    for epoch in range(epochs):
        spatial.train()
        losses = []
        term_sums: dict[str, list[float]] = {}
        for number, day in enumerate(dates, 1):
            x, y, _ = store.batch(int(day), training=True)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            keep, target = drop_extreme_and_zscore(y)
            loss, terms = stage1_loss(spatial, variant, x[keep], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(spatial.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in terms.items():
                term_sums.setdefault(name, []).append(value)
            if number % 250 == 0 or number == len(dates):
                log(f'{variant} spatial epoch={epoch} {number}/{len(dates)} loss={np.mean(losses):.6f}')
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'terms': {name: float(np.mean(values)) for name, values in term_sums.items()}})
    for parameter in spatial.parameters():
        parameter.requires_grad_(False)
    return history


def train_predictor(model, label, store, dates, optimizer, device, epochs, patience, seed, max_train_days, max_eval_days):
    rng = np.random.RandomState(seed)
    best_score = -float('inf')
    best_state = None
    best_epoch = -1
    stale = 0
    history = []
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(epochs):
        model.train()
        order = dates.copy()
        rng.shuffle(order)
        if max_train_days is not None:
            order = order[:max_train_days]
        losses = []
        for number, day in enumerate(order, 1):
            x, y, _ = store.batch(int(day), training=True)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            keep, target = drop_extreme_and_zscore(y)
            pred = model(x[keep])
            loss = F.mse_loss(pred, target) + 0.1 * pearson_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if number % 250 == 0 or number == len(order):
                log(f'{label} epoch={epoch} {number}/{len(order)} loss={np.mean(losses):.6f}')
        valid, score = evaluate_for_selection(model, store, device, max_eval_days)
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy({name: value for name, value in model.state_dict().items() if not name.startswith('base.')})
        else:
            stale += 1
        entry = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': valid, 'selection_score': score, 'best': improved}
        if hasattr(model, 'residual_scale'):
            entry['residual_scale'] = float(torch.tanh(model.residual_scale).detach().cpu())
        history.append(entry)
        log(f"{label} epoch={epoch} valid_IC={valid['IC']:.6f} valid_RankIC={valid['RankIC']:.6f} best={best_epoch}")
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f'{label} did not produce a finite validation checkpoint')
    current = model.state_dict()
    current.update(best_state)
    model.load_state_dict(current)
    return (history, best_epoch, best_state)


class DeformableTemporalResidual(nn.Module):

    def __init__(self, history=8, d_model=64, selected=1):
        super().__init__()
        self.selected = selected
        self.input_proj = nn.Linear(158, d_model)
        self.position = nn.Parameter(torch.randn(1, history, d_model) * 0.02)
        self.selector = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, stock):
        sequence = self.input_proj(stock) + self.position[:, :stock.shape[1]]
        scores = self.selector(sequence).squeeze(-1)
        top = torch.topk(scores, k=min(self.selected, scores.shape[1]), dim=1).indices
        mask = torch.full_like(scores, -torch.inf)
        mask.scatter_(1, top, scores.gather(1, top))
        weights = torch.softmax(mask, dim=1)
        pooled = torch.einsum('nt,ntd->nd', weights, self.value(sequence))
        return self.head(pooled).squeeze(-1)


class SparseExpert(nn.Module):

    def __init__(self, anchor, variant, selected=1):
        super().__init__()
        self.base = anchor
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = DeformableTemporalResidual(selected=selected)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        self.base.eval()
        with torch.no_grad():
            anchor_pred = self.base(x)
        return anchor_pred + torch.tanh(self.residual_scale) * self.residual(x[:, :, :158])


def train_master_main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='full')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'master')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    model = MasterExpert(args.variant, args.universe, d_model=args.d_model).to(device)
    started = time.time()
    history, best_epoch = train_one(args, store, model, device)
    width_suffix = '' if args.d_model == 256 else f'_d{args.d_model}'
    prefix = f'master_{args.variant}{width_suffix}_{args.universe}_seed{args.seed}'
    torch.save(model.state_dict(), args.output_dir / f'{prefix}_best.pt')
    rows, predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    result = {'experiment': 'MASTER expert', 'variant': args.variant, 'universe': args.universe, 'seed': args.seed, 'best_epoch': best_epoch, 'history': history, 'metrics': metrics_by_period(rows), 'parameters': sum((p.numel() for p in model.parameters())), 'd_model': args.d_model, 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'output_dir': str(args.output_dir.resolve())}}
    if 'encoder_checkpoint' in result.get('config', {}): result['config']['encoder_checkpoint'] = str(args.encoder_checkpoint) if args.encoder_checkpoint else None
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    write_daily_csv(args.output_dir / f'{prefix}_daily.csv', rows)
    np.savez_compressed(args.output_dir / f'{prefix}_predictions.npz', **predictions)
    log(json.dumps({'variant': args.variant, 'best_epoch': best_epoch, 'metrics': result['metrics']}, ensure_ascii=False))


def train_continuous_main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='no_vq')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage1-epochs', type=int, default=4)
    parser.add_argument('--encoder-checkpoint', type=Path, help='Reuse the frozen four-epoch encoder from a ranker state_dict')
    parser.add_argument('--base-epochs', type=int, default=12)
    parser.add_argument('--adapter-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'continuous_prism')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    prefix = f'{args.variant}_{args.universe}_seed{args.seed}'
    started = time.time()
    spatial = ContinuousEncoder(args.variant).to(device)
    optimizer_spatial = torch.optim.AdamW(spatial.parameters(), lr=args.lr, weight_decay=0.0001)
    if args.encoder_checkpoint:
        encoder_state = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
        spatial.load_state_dict({k[len('spatial.'):]: v for k, v in encoder_state.items() if k.startswith('spatial.')}, strict=True)
        for p in spatial.parameters(): p.requires_grad_(False)
        spatial_history = []
        spatial_source = str(args.encoder_checkpoint.resolve())
    else:
        spatial_history = train_spatial(spatial, args.variant, store, train_dates.copy(), optimizer_spatial, device, args.stage1_epochs, args.max_train_days)
        spatial_source = 'retrained_from_scratch'
    base = MixtureRanker(spatial, args.variant).to(device)
    base_trainable = [p for p in base.parameters() if p.requires_grad]
    optimizer_base = torch.optim.AdamW(base_trainable, lr=args.lr, weight_decay=0.0001)
    base_history, base_best_epoch, base_state = train_predictor(base, f'{args.variant}/base', store, train_dates.copy(), optimizer_base, device, args.base_epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    base_checkpoint = args.output_dir / f'{prefix}_base_best.pt'
    torch.save(base.state_dict(), base_checkpoint)
    base_rows, base_predictions = evaluate(base, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_base_daily.csv', base_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_base_predictions.npz', **base_predictions)
    adapter = TemporalAdapter(base).to(device)
    adapter_trainable = [p for p in adapter.parameters() if p.requires_grad]
    optimizer_adapter = torch.optim.AdamW(adapter_trainable, lr=args.lr, weight_decay=0.0001)
    adapter_history, adapter_best_epoch, adapter_state = train_predictor(adapter, f'{args.variant}/adapter', store, train_dates.copy(), optimizer_adapter, device, args.adapter_epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    torch.save(adapter_state, args.output_dir / f'{prefix}_adapter_best.pt')
    adapter_rows, adapter_predictions = evaluate(adapter, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_adapter_daily.csv', adapter_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_adapter_predictions.npz', **adapter_predictions)
    result = {'experiment': 'Continuous expert: encoder, mixture ranker, temporal residual', 'variant': args.variant, 'what_changed': 'continuous encoder and mixture ranker with temporal residual', 'universe': args.universe, 'seed': args.seed, 'spatial_source': spatial_source, 'spatial_history': spatial_history, 'base': {'best_epoch': base_best_epoch, 'history': base_history, 'metrics': metrics_by_period(base_rows), 'parameters': sum((p.numel() for p in base.parameters())), 'trainable_parameters_stage2': len(base_state) and sum((p.numel() for p in base_trainable)), 'checkpoint': str(base_checkpoint.resolve())}, 'with_temporal_adapter': {'best_epoch': adapter_best_epoch, 'history': adapter_history, 'metrics': metrics_by_period(adapter_rows), 'total_parameters': sum((p.numel() for p in adapter.parameters())), 'adapter_trainable_parameters': sum((p.numel() for p in adapter_trainable)), 'final_residual_scale': float(torch.tanh(adapter.residual_scale).detach().cpu())}, 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'output_dir': str(args.output_dir.resolve())}}
    if 'encoder_checkpoint' in result.get('config', {}): result['config']['encoder_checkpoint'] = str(args.encoder_checkpoint) if args.encoder_checkpoint else None
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    log(json.dumps({'variant': args.variant, 'base': result['base']['metrics'], 'with_temporal_adapter': result['with_temporal_adapter']['metrics'], 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


def train_sparse_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.set_defaults(variant='deformable')
    parser.add_argument('--universe', choices=('csi300', 'csi800'), default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--selected', type=int, default=1, help='number of temporal positions retained by the deformable residual')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max-train-days', type=int)
    parser.add_argument('--max-eval-days', type=int)
    parser.add_argument('--continuous-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    seed_all(args.seed)
    device = torch.device('cuda:0')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory: " + str(args.output_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = UniverseStore(DATASET_ROOT, args.universe)
    train_dates = store.dates_between(TRAIN_START, TRAIN_END)
    prefix_base = f'no_vq_{args.universe}_seed{args.seed}'
    ranker = MixtureRanker(ContinuousEncoder('no_vq'), 'no_vq').to(device)
    ranker.load_state_dict(torch.load(args.continuous_dir / f'{prefix_base}_base_best.pt', map_location=device, weights_only=True))
    anchor = TemporalAdapter(ranker).to(device)
    anchor.load_state_dict(torch.load(args.continuous_dir / f'{prefix_base}_adapter_best.pt', map_location=device, weights_only=True), strict=False)
    if not 1 <= args.selected <= 8:
        raise ValueError('--selected must lie in [1, 8]')
    seed_all(args.seed)
    model = SparseExpert(anchor, args.variant, selected=args.selected).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0001)
    started = time.time()
    history, best_epoch, best_state = train_predictor(model, f'ta_{args.variant}', store, train_dates.copy(), optimizer, device, args.epochs, args.patience, args.seed, args.max_train_days, args.max_eval_days)
    suffix = f'_topk{args.selected}' if args.variant == 'deformable' and args.selected != 4 else ''
    prefix = f'ta_{args.variant}{suffix}_{args.universe}_seed{args.seed}'
    torch.save(best_state, args.output_dir / f'{prefix}_best.pt')
    valid_rows, valid_predictions = evaluate(model, store, VALID_START, VALID_END, device, args.max_eval_days)
    test_rows, test_predictions = evaluate(model, store, TEST_START, TEST_END, device, args.max_eval_days)
    write_daily_csv(args.output_dir / f'{prefix}_validation_daily.csv', valid_rows)
    write_daily_csv(args.output_dir / f'{prefix}_daily.csv', test_rows)
    np.savez_compressed(args.output_dir / f'{prefix}_validation_predictions.npz', **valid_predictions)
    np.savez_compressed(args.output_dir / f'{prefix}_predictions.npz', **test_predictions)
    result = {'experiment': 'Sparse temporal residual on the frozen temporal expert', 'variant': args.variant, 'universe': args.universe, 'seed': args.seed, 'best_epoch': best_epoch, 'history': history, 'validation_metrics': history[best_epoch]['validation'], 'metrics': metrics_by_period(test_rows), 'trainable_parameters': sum((p.numel() for p in trainable)), 'total_parameters': sum((p.numel() for p in model.parameters())), 'final_residual_scale': float(torch.tanh(model.residual_scale).detach().cpu()), 'elapsed_seconds': time.time() - started, 'mode': 'single-process single-threaded', 'config': vars(args) | {'continuous_dir': str(args.continuous_dir.resolve()), 'output_dir': str(args.output_dir.resolve())}}
    if 'encoder_checkpoint' in result.get('config', {}): result['config']['encoder_checkpoint'] = str(args.encoder_checkpoint) if args.encoder_checkpoint else None
    (args.output_dir / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'variant': args.variant, 'best_epoch': best_epoch, 'validation': result['validation_metrics'], 'metrics': result['metrics'], 'trainable_parameters': result['trainable_parameters']}, ensure_ascii=False, indent=2), flush=True)



def daily_zscore(scores: torch.Tensor) -> torch.Tensor:
    """One complete date; match the release's NumPy float64 population z-score."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("Expected a nonempty daily score vector")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite prediction")
    values = scores.to(torch.float64)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-12)


class CoSTER(nn.Module):
    """Full algorithm. Use eval() for inference; train experts with stage CLIs."""
    def __init__(self, universe="csi300"):
        super().__init__()
        if universe not in ("csi300", "csi800"):
            raise ValueError("universe must be csi300 or csi800")
        self.universe = universe
        self.master = MasterExpert("full", universe)
        base = MixtureRanker(ContinuousEncoder("no_vq"), "no_vq")
        self.continuous = SparseExpert(TemporalAdapter(base), "deformable", selected=1)

    def forward(self, x, return_components=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (8, 221) or x.shape[0] < 1:
            raise ValueError("Expected one full daily cross-section [N, 8, 221]")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Inputs must be finite, preprocessed floating-point features")
        master_score = self.master(x)
        expert_score = self.continuous(x)
        scores = 0.5 * daily_zscore(master_score) + 0.5 * daily_zscore(expert_score)
        if return_components:
            return {"scores": scores, "master": master_score, "continuous": expert_score}
        return scores

    @classmethod
    def from_checkpoints(cls, master_path, base_path, temporal_path, sparse_path,
                         universe="csi300", device="cpu"):
        """Load the four original state_dict files; return an evaluation model."""
        model = cls(universe).to(device)
        def state(path):
            return torch.load(Path(path), map_location=device, weights_only=True)
        def residual(module, path):
            missing, unexpected = module.load_state_dict(state(path), strict=False)
            if unexpected or any(not key.startswith("base.") for key in missing):
                raise ValueError(f"Incompatible residual checkpoint: {path}")
        model.master.load_state_dict(state(master_path), strict=True)
        model.continuous.base.base.load_state_dict(state(base_path), strict=True)
        residual(model.continuous.base, temporal_path)
        residual(model.continuous, sparse_path)
        return model.eval()

    @classmethod
    def from_run_dir(cls, run_dir, universe="csi300", seed=0, device="cpu"):
        root = Path(run_dir)
        return cls.from_checkpoints(
            root / "master" / f"master_full_{universe}_seed{seed}_best.pt",
            root / "continuous_prism" / f"no_vq_{universe}_seed{seed}_base_best.pt",
            root / "continuous_prism" / f"no_vq_{universe}_seed{seed}_adapter_best.pt",
            root / "sparse_top1" / f"ta_deformable_topk1_{universe}_seed{seed}_best.pt",
            universe, device)


def infer_main(argv=None):
    p = argparse.ArgumentParser(description="CoSTER full-universe daily inference")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--universe", choices=["csi300", "csi800"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-eval-days", type=int)
    args = p.parse_args(argv)
    if args.max_eval_days is not None and args.max_eval_days < 1:
        p.error("--max-eval-days must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    model = CoSTER.from_run_dir(args.run_dir, args.universe, args.seed, args.device)
    store = UniverseStore(DATASET_ROOT, args.universe)
    start, end = (VALID_START, VALID_END) if args.split == "validation" else (TEST_START, TEST_END)
    rows, predictions = evaluate(model, store, start, end, torch.device(args.device), args.max_eval_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"CoSTER_{args.universe}_seed{args.seed}_{args.split}"
    np.savez_compressed(args.output_dir / f"{prefix}_predictions.npz", **predictions)
    write_daily_csv(args.output_dir / f"{prefix}_daily.csv", rows)
    summary = summarize_daily(rows)
    (args.output_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def demo_main(argv=None):
    p = argparse.ArgumentParser(description="Random-weight CPU shape check, not paper results")
    p.add_argument("--universe", choices=["csi300", "csi800"], default="csi300")
    p.add_argument("--stocks", type=int, default=32)
    args = p.parse_args(argv)
    if args.stocks < 1:
        p.error("--stocks must be positive")
    seed_all(0)
    model = CoSTER(args.universe).eval()
    x = torch.randn(args.stocks, 8, 221)
    x[:, :, 158:] = x[0:1, :, 158:].clone()
    with torch.inference_mode():
        out = model(x, return_components=True)
    print(json.dumps({"mode": "random-weight shape check", "universe": args.universe,
        "input_shape": list(x.shape), "output_shape": list(out["scores"].shape),
        "finite": bool(torch.isfinite(out["scores"]).all()),
        "master_parameters": sum(p.numel() for p in model.master.parameters()),
        "continuous_parameters": sum(p.numel() for p in model.continuous.parameters())}, indent=2))


def main(argv=None):
    global DATASET_ROOT
    p = argparse.ArgumentParser(description="Standalone CoSTER: model, staged training and inference")
    p.add_argument("command", choices=["demo", "train-master", "train-continuous", "train-sparse", "infer"], nargs="?", default="demo")
    p.add_argument("--dataset-root", type=Path, help="Prepared MASTER-EXT25 dataset directory")
    # Stage help is routed to the selected command, keeping this file self-contained.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("demo", "train-master", "train-continuous", "train-sparse", "infer") and "--help" in argv:
        actions = {"demo": demo_main, "train-master": train_master_main, "train-continuous": train_continuous_main, "train-sparse": train_sparse_main, "infer": infer_main}
        actions[argv[0]](["--help"])
        return
    args, remaining = p.parse_known_args(argv)
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root.resolve()
    if args.command != "demo" and args.dataset_root is None and "MASTER_EXT_DATASET_ROOT" not in os.environ:
        p.error("Specify --dataset-root (or MASTER_EXT_DATASET_ROOT)")
    if args.command.startswith("train-"):
        manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8-sig"))
        if manifest.get("status") != "PASS" or manifest.get("temporal_audit", {}).get("training_feature_cutoff") != 20191224:
            raise ValueError("Expected train-only dataset manifest with cutoff 20191224")
        # The original stage entry points configure Torch's thread counts.
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    {"demo": demo_main, "train-master": train_master_main,
     "train-continuous": train_continuous_main, "train-sparse": train_sparse_main,
     "infer": infer_main}[args.command](remaining)


if __name__ == "__main__":
    main()

