from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import subprocess
import warnings
import urllib.request
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore", message="enable_nested_tensor.*")


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ag_news"
RESULT_DIR = ROOT / "after_edit"
CHECKPOINT_DIR = ROOT / "checkpoints"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

AG_NEWS_URLS = {
    "train": [
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "https://github.com/mhjabreel/CharCnn_Keras/raw/master/data/ag_news_csv/train.csv",
    ],
    "test": [
        "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
        "https://github.com/mhjabreel/CharCnn_Keras/raw/master/data/ag_news_csv/test.csv",
    ],
}

## AG News 是一个经典的英文新闻分类数据集，给一条新闻标题/摘要，判断它属于 4 个类别之一
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(LABEL_NAMES)

## 在定义分词器/词表里要用的特殊符号，再配一个正则表达式分词规则
PAD_TOKEN = "[PAD]" # 补齐长度用
UNK_TOKEN = "[UNK]" # 词表里没有的词
CLS_TOKEN = "[CLS]" # 分类任务常放在开头的标记
SEP_TOKEN = "[SEP]" # 句子/片段分隔标记
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretty(value: float) -> str:
    return f"{value:.4f}"


## 设备这一块
def make_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_speed_flags(device: torch.device) -> None:
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True



## 下面都是数据集处理这一块，没什么要讲的，按着数据集解释说明抄的
def download_file(urls: Sequence[str], path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Optional[Exception] = None
    curl = shutil.which("curl.exe") or shutil.which("curl")

    for url in urls:
        temp_path = path.with_name(path.name + ".part")
        try:
            if curl:
                cmd = [
                    curl,
                    "--noproxy",
                    "*",
                    "-L",
                    "--retry",
                    "6",
                    "--retry-delay",
                    "2",
                    "--retry-all-errors",
                    "--continue-at",
                    "-",
                    "-o",
                    str(temp_path),
                    url,
                ]
                subprocess.run(cmd, check=True)
                if temp_path.exists():
                    temp_path.replace(path)
                    return path
            ## codex帮助我验证的client
            req = urllib.request.Request(url, headers={"User-Agent": "Codex"})
            with opener.open(req, timeout=120) as response, temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            temp_path.replace(path)
            return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if temp_path.exists() and temp_path.stat().st_size == 0:
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    raise RuntimeError(f"failed to download {path.name}: {last_error}")


def ensure_dataset_files() -> Tuple[Path, Path]:
    return download_file(AG_NEWS_URLS["train"], TRAIN_CSV), download_file(AG_NEWS_URLS["test"], TEST_CSV)


def read_ag_news_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=["label", "title", "description"])
    frame["label"] = frame["label"].astype(int) - 1
    text = frame["title"].fillna("").astype(str).str.strip() + " " + frame["description"].fillna("").astype(str).str.strip()
    frame["text"] = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return frame[["label", "text"]].reset_index(drop=True)


def summarize_frame(frame: pd.DataFrame, name: str) -> None:
    counts = frame["label"].value_counts().sort_index()
    print(f"{name}: {len(frame)}")
    for idx, label in enumerate(LABEL_NAMES):
        print(f"  {label:8s}: {int(counts.get(idx, 0))}")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


## 分层
def stratified_split(
    frame: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for _, part in frame.groupby(label_col):
        indices = part.index.to_numpy(copy=True)
        rng.shuffle(indices)
        cut = int(round(len(indices) * train_ratio))
        train_parts.append(frame.loc[indices[:cut]])
        val_parts.append(frame.loc[indices[cut:]])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return train_df, val_df


## 抽样
def sample_frame(frame: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or max_rows >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


## 封装词表这个概念
class TextVocab:
    def __init__(self, max_size: int = 50000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def fit(self, texts: Sequence[str]) -> "TextVocab":
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        candidates = [token for token, count in counter.items() if count >= self.min_freq]
        candidates.sort(key=lambda token: (-counter[token], token))
        for token in candidates:
            if len(self.id_to_token) >= self.max_size:
                break
            self.add_token(token)
        return self

    ## 把一句文本变成固定长度的数字序列：分词；加 [CLS] 和 [SEP]；查词表转id；不认识的词用 [UNK]；不够长的用 [PAD] 补齐
    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = tokenize(text)
        if max_len >= 2:
            tokens = [CLS_TOKEN] + tokens[: max_len - 2] + [SEP_TOKEN]
        else:
            tokens = tokens[:max_len]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids[:max_len]


## 表格数据变成pytorch训练时可以直接喂给模型的batch数据
def encode_frame(frame: pd.DataFrame, vocab: TextVocab, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = [vocab.encode(text, max_len) for text in frame["text"].tolist()]
    labels = frame["label"].to_numpy(dtype=np.int64)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


## 把训练集、验证集、测试集都包装成 DataLoader
def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: TextVocab,
    max_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_x, train_y = encode_frame(train_df, vocab, max_len)
    val_x, val_y = encode_frame(val_df, vocab, max_len)
    test_x, test_y = encode_frame(test_df, vocab, max_len)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


## 文本 -> 词向量 + 位置向量 -> Transformer 编码 -> 分类
class TextTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        max_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_norm(self.dropout(x))
        padding_mask = input_ids.eq(self.pad_id)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.head(x[:, 0])


## 文本 -> 词向量 -> 平均池化 -> 分类
class MeanPoolClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        d_model: int = 128,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        mask = input_ids.ne(self.pad_id).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.net(pooled)


@dataclass
class ExperimentConfig:
    name: str              # 实验名称
    kind: str              # 模型类型，例如 transformer / meanpool
    epochs: int            # 训练轮数
    batch_size: int        # 每个 batch 的样本数
    lr: float              # 学习率
    weight_decay: float    # 权重衰减，用于防止过拟合
    patience: int          # 早停等待轮数
    warmup_ratio: float    # 学习率 warmup 占比
    min_lr_ratio: float    # 最小学习率比例
    label_smoothing: float # 标签平滑系数
    d_model: int           # 模型隐藏维度 / 词向量维度
    nhead: int = 4         # Transformer 注意力头数，默认 4
    num_layers: int = 2    # Transformer 编码层数，默认 2
    ff_dim: int = 512      # 前馈网络中间维度
    hidden: int = 256      # 分类器隐藏层维度
    dropout: float = 0.2   # dropout 比例


## 损失函数
def classification_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing <= 0: # 如果 smoothing <= 0，就直接用普通交叉熵
        return F.cross_entropy(logits, targets)
    # 标签平滑：让模型别对某个类别过于自信
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


## 模型工厂函数：根据 spec.kind 决定造哪种模型。
def build_model(spec: ExperimentConfig, vocab: TextVocab, max_len: int, num_classes: int) -> nn.Module:
    if spec.kind == "transformer":
        return TextTransformerClassifier(
            vocab_size=vocab.vocab_size,
            num_classes=num_classes,
            pad_id=vocab.pad_id,
            max_len=max_len,
            d_model=spec.d_model,
            nhead=spec.nhead,
            num_layers=spec.num_layers,
            ff_dim=spec.ff_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "baseline":
        return MeanPoolClassifier(
            vocab_size=vocab.vocab_size,
            num_classes=num_classes,
            pad_id=vocab.pad_id,
            d_model=spec.d_model,
            hidden=spec.hidden,
            dropout=spec.dropout,
        )
    raise ValueError(f"unknown model kind: {spec.kind}")


## 学习率调度器，控制训练过程中学习率怎么变
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(total_steps, 1)
    warmup_steps = max(1, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        # 前期 warmup，一开始学习率从小慢慢升上来
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        # 后期 cosine decay，学习率按余弦曲线慢慢下降，最低不会降到 0，而是降到 min_lr_ratio 对应的比例
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_confusion_matrix(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> Dict[str, object]:
    matrix = matrix.astype(np.float64)
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = matrix.sum(axis=1)
    accuracy = float(tp.sum() / max(matrix.sum(), 1.0))
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.astype(int).tolist(),
        "macro_f1": float(f1.mean() if len(f1) else 0.0),
        "accuracy": accuracy,
    }


def batch_to_device(batch, device: torch.device):
    return tuple(item.to(device) for item in batch)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler,
    device: torch.device,
    smoothing: float,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    use_amp = device.type == "cuda"

    for input_ids, labels in loader:
        input_ids, labels = batch_to_device((input_ids, labels), device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(input_ids)
                loss = classification_loss(logits, labels, smoothing=smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            loss = classification_loss(logits, labels, smoothing=smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        ## 每个 batch 后更新一次学习率
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    smoothing: float = 0.0,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for input_ids, labels in loader:
        input_ids, labels = batch_to_device((input_ids, labels), device)
        logits = model(input_ids)
        loss = classification_loss(logits, labels, smoothing=smoothing)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    confusion = make_confusion_matrix(all_targets, all_predictions, NUM_CLASSES)
    report = metrics_from_confusion(confusion)
    report.update(
        {
            "loss": total_loss / max(total_count, 1),
            "confusion": confusion,
            "targets": all_targets,
            "predictions": all_predictions,
        }
    )
    return report


def train_model(
    spec: ExperimentConfig,
    vocab: TextVocab,
    max_len: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, object]]:
    model = build_model(spec, vocab, max_len, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    total_steps = max(len(train_loader) * spec.epochs, 1)
    warmup_steps = max(int(total_steps * spec.warmup_ratio), 1)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, spec.min_lr_ratio)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    stale = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{spec.name}.pt"

    for epoch in range(1, spec.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            smoothing=spec.label_smoothing,
        )
        val_stats = evaluate_model(model, val_loader, device, smoothing=0.0)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_stats["loss"]))
        history["train_accuracy"].append(float(train_stats["accuracy"]))
        history["val_loss"].append(float(val_stats["loss"]))
        history["val_accuracy"].append(float(val_stats["accuracy"]))
        history["val_macro_f1"].append(float(val_stats["macro_f1"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        current_f1 = float(val_stats["macro_f1"])
        if current_f1 > best_val_f1 + 1e-6:
            best_val_f1 = current_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(
            f"{spec.name} | epoch {epoch:02d} | "
            f"train loss {pretty(train_stats['loss'])} acc {pretty(train_stats['accuracy'])} | "
            f"val loss {pretty(val_stats['loss'])} acc {pretty(val_stats['accuracy'])} "
            f"macro-f1 {pretty(val_stats['macro_f1'])} | lr {pretty(history['lr'][-1])}"
        )

        if stale >= spec.patience:
            print(f"{spec.name} | early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": spec.__dict__,
            "vocab_size": vocab.vocab_size,
            "max_len": max_len,
        },
        checkpoint_path,
    )

    test_stats = evaluate_model(model, test_loader, device, smoothing=0.0)
    test_stats["best_epoch"] = best_epoch
    test_stats["best_val_macro_f1"] = best_val_f1
    test_stats["checkpoint"] = str(checkpoint_path)
    return model, history, test_stats


## 开始画画了，ai写的
def plot_history(history: Dict[str, List[float]], title: str, out_file: Path) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train acc")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="val acc")
    axes[1].plot(epochs, history["val_macro_f1"], marker="o", label="val macro-f1")
    axes[1].set_title("metrics")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_comparison(histories: Dict[str, Dict[str, List[float]]], out_file: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)
    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["val_loss"], marker="o", label=name)
        axes[1].plot(history["epoch"], history["val_macro_f1"], marker="o", label=name)

    axes[0].set_title("validation loss")
    axes[1].set_title("validation macro-f1")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_confusion_matrix(confusion: np.ndarray, out_file: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8), dpi=160)
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(LABEL_NAMES)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)



### 下面的不用看了，让ai帮我写的个简单全面的报告
def format_report(name: str, stats: Dict[str, object]) -> str:
    lines = [f"{name}"]
    lines.append(f"  test loss: {pretty(float(stats['loss']))}")
    lines.append(f"  test acc : {pretty(float(stats['accuracy']))}")
    lines.append(f"  macro F1 : {pretty(float(stats['macro_f1']))}")
    lines.append(f"  best epoch: {stats['best_epoch']}")
    lines.append(f"  best val macro F1: {pretty(float(stats['best_val_macro_f1']))}")
    for idx, label in enumerate(LABEL_NAMES):
        lines.append(
            f"  {label:9s} | p={pretty(float(stats['precision'][idx]))} "
            f"r={pretty(float(stats['recall'][idx]))} f1={pretty(float(stats['f1'][idx]))} "
            f"support={int(stats['support'][idx])}"
        )
    return "\n".join(lines)


def write_results_readme(
    transformer_stats: Dict[str, object],
    baseline_stats: Dict[str, object],
    train_count: int,
    val_count: int,
    test_count: int,
    vocab_size: int,
    max_len: int,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        "# after_edit\n\n"
        "这次换成了真实公开文本数据集 AG News，整体就是一个更适合 Transformer 的新闻分类任务。\n\n"
        "训练设置：\n\n"
        "- encoder-only Transformer 做四分类\n"
        "- learned positional embedding + CLS 分类头\n"
        "- warmup + cosine decay\n"
        "- dropout + label smoothing + early stopping\n\n"
        f"- train / val / test = {train_count} / {val_count} / {test_count}\n"
        f"- vocab size = {vocab_size}\n"
        f"- max length = {max_len}\n\n"
        "主要结果：\n\n"
        f"- transformer_large acc = {pretty(float(transformer_stats['accuracy']))}\n"
        f"- transformer_large macro-F1 = {pretty(float(transformer_stats['macro_f1']))}\n"
        f"- baseline acc = {pretty(float(baseline_stats['accuracy']))}\n"
        f"- baseline macro-F1 = {pretty(float(baseline_stats['macro_f1']))}\n"
    )
    (RESULT_DIR / "README.md").write_text(text, encoding="utf-8")


def run_experiment_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab: TextVocab,
    max_len: int,
    device: torch.device,
    batch_size: int,
    small_epochs: int,
    large_epochs: int,
    baseline_epochs: int,
) -> None:
    train_loader, val_loader, test_loader = build_loaders(train_df, val_df, test_df, vocab, max_len, batch_size)

    small_spec = ExperimentConfig(
        name="transformer_small",
        kind="transformer",
        epochs=small_epochs,
        batch_size=batch_size,
        lr=3e-4,
        weight_decay=1e-2,
        patience=3,
        warmup_ratio=0.08,
        min_lr_ratio=0.08,
        label_smoothing=0.05,
        d_model=128,
        nhead=4,
        num_layers=2,
        ff_dim=384,
        dropout=0.15,
    )
    large_spec = ExperimentConfig(
        name="transformer_large",
        kind="transformer",
        epochs=large_epochs,
        batch_size=batch_size,
        lr=3e-4,
        weight_decay=1e-2,
        patience=3,
        warmup_ratio=0.1,
        min_lr_ratio=0.06,
        label_smoothing=0.05,
        d_model=192,
        nhead=6,
        num_layers=4,
        ff_dim=768,
        dropout=0.2,
    )
    baseline_spec = ExperimentConfig(
        name="mlp_baseline",
        kind="baseline",
        epochs=baseline_epochs,
        batch_size=batch_size,
        lr=4e-4,
        weight_decay=1e-2,
        patience=3,
        warmup_ratio=0.08,
        min_lr_ratio=0.1,
        label_smoothing=0.03,
        d_model=128,
        hidden=256,
        dropout=0.2,
    )

    small_model, small_history, small_stats = train_model(
        small_spec, vocab, max_len, train_loader, val_loader, test_loader, device
    )
    plot_history(small_history, "transformer_small curves", RESULT_DIR / "transformer_small_curves.png")
    plot_confusion_matrix(small_stats["confusion"], RESULT_DIR / "transformer_small_confusion.png", "transformer_small confusion matrix")
    print(format_report("transformer_small", small_stats))
    print()

    large_model, large_history, large_stats = train_model(
        large_spec, vocab, max_len, train_loader, val_loader, test_loader, device
    )
    plot_history(large_history, "transformer_large curves", RESULT_DIR / "transformer_large_curves.png")
    plot_confusion_matrix(large_stats["confusion"], RESULT_DIR / "transformer_large_confusion.png", "transformer_large confusion matrix")
    print(format_report("transformer_large", large_stats))
    print()

    baseline_model, baseline_history, baseline_stats = train_model(
        baseline_spec, vocab, max_len, train_loader, val_loader, test_loader, device
    )
    plot_history(baseline_history, "mlp_baseline curves", RESULT_DIR / "mlp_baseline_curves.png")
    plot_confusion_matrix(baseline_stats["confusion"], RESULT_DIR / "mlp_baseline_confusion.png", "mlp baseline confusion matrix")
    print(format_report("mlp_baseline", baseline_stats))
    print()

    plot_comparison(
        {"transformer_small": small_history, "transformer_large": large_history, "mlp_baseline": baseline_history},
        RESULT_DIR / "transformer_scale_comparison.png",
    )

    write_results_readme(
        transformer_stats=large_stats,
        baseline_stats=baseline_stats,
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        vocab_size=vocab.vocab_size,
        max_len=max_len,
    )

    _ = small_model, large_model, baseline_model  # keeps the trained models alive until the suite ends


def show_dataset_overview(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, vocab: TextVocab) -> None:
    print("dataset overview")
    summarize_frame(train_df, "train")
    summarize_frame(val_df, "val")
    summarize_frame(test_df, "test")
    print("vocab size:", vocab.vocab_size)
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AG News encoder-only Transformer classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--small-epochs", type=int, default=5)
    parser.add_argument("--large-epochs", type=int, default=8)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="run a small smoke test")
    return parser


def apply_fast_mode(args: argparse.Namespace) -> None:
    args.train_samples = args.train_samples or 12000
    args.val_samples = args.val_samples or 2000
    args.test_samples = args.test_samples or 2000
    args.small_epochs = min(args.small_epochs, 3)
    args.large_epochs = min(args.large_epochs, 4)
    args.baseline_epochs = min(args.baseline_epochs, 2)
    args.batch_size = min(args.batch_size, 192)
    args.max_len = min(args.max_len, 96)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fast:
        apply_fast_mode(args)

    set_seed(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    enable_speed_flags(make_device(args.device))

    train_csv, test_csv = ensure_dataset_files()
    full_train = read_ag_news_csv(train_csv)
    test_df = read_ag_news_csv(test_csv)
    train_df, val_df = stratified_split(full_train, train_ratio=0.9, seed=args.seed)

    train_df = sample_frame(train_df, args.train_samples, args.seed)
    val_df = sample_frame(val_df, args.val_samples, args.seed + 1)
    test_df = sample_frame(test_df, args.test_samples, args.seed + 2)

    vocab = TextVocab(max_size=args.max_vocab_size, min_freq=args.min_freq).fit(train_df["text"].tolist())
    show_dataset_overview(train_df, val_df, test_df, vocab)

    device = make_device(args.device)
    print("device:", device)
    print()

    run_experiment_suite(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        vocab=vocab,
        max_len=args.max_len,
        device=device,
        batch_size=args.batch_size,
        small_epochs=args.small_epochs,
        large_epochs=args.large_epochs,
        baseline_epochs=args.baseline_epochs,
    )


if __name__ == "__main__":
    main()
