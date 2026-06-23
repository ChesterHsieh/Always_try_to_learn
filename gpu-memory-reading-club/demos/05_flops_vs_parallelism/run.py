"""FLOPs vs 平行度 demo（讀書會 S5）。

兩個實驗，各自回答一個問題：

  A) 序列相依 vs 全平行：同規模的 LSTM 與 Transformer block 跑同一段序列，
     理論 FLOPs 相近（transformer 甚至更多），wall-clock 與達成 TFLOPS 差多少？
  B) FLOPs ≠ 速度：dense 3×3 conv vs depthwise separable conv，
     紙上 FLOPs ÷8~9，實際時間有沒有 ÷8？

用法：
    python run.py                       # 預設參數、自動選 device（cuda > mps > cpu）
    python run.py --seq-len 1024 --hidden 768
    python run.py --device cpu          # 功能驗證用

說明：FLOPs 採標準理論計數（MAC×2），不含 softmax/norm 等小項。
重點不在絕對數字，而在「FLOPs 相近、速度差一個量級」的相對關係。
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Measurement:
    label: str
    flops: float
    seconds: float

    @property
    def tflops(self) -> float:
        return self.flops / self.seconds / 1e12

    @property
    def ms(self) -> float:
        return self.seconds * 1e3


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward(fn, device: torch.device, repeats: int, warmup: int) -> float:
    """回傳單次 forward 的中位數秒數（同步圍住計時區間）。"""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync(device)
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------- 實驗 A

class TransformerBlock(nn.Module):
    """標準 pre-norm block：self-attention + 4× FFN（與 LSTM 同 hidden 規模對照）。"""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


def lstm_flops(b: int, t: int, h: int) -> float:
    # 每步：weight_ih (4H×H) 與 weight_hh (4H×H) 兩個 matvec → 16H² FLOPs
    return float(b * t * 16 * h * h)


def transformer_flops(b: int, t: int, h: int) -> float:
    proj = 8.0 * b * t * h * h          # Q,K,V,O 四個 H×H 投影
    attn = 4.0 * b * t * t * h          # QKᵀ 與 attn·V
    ffn = 16.0 * b * t * h * h          # 兩個 H×4H GEMM
    return proj + attn + ffn


def run_experiment_a(args: argparse.Namespace, device: torch.device) -> list[Measurement]:
    b, t, h = args.batch, args.seq_len, args.hidden
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.randn(b, t, h, device=device, dtype=dtype)

    lstm = nn.LSTM(h, h, batch_first=True).to(device=device, dtype=dtype)
    block = TransformerBlock(h, args.heads).to(device=device, dtype=dtype)

    sec_lstm = time_forward(lambda: lstm(x), device, args.repeats, args.warmup)
    sec_tf = time_forward(lambda: block(x), device, args.repeats, args.warmup)
    return [
        Measurement(f"LSTM        (T={t} 步序列相依)", lstm_flops(b, t, h), sec_lstm),
        Measurement(f"Transformer (T={t} 全平行)", transformer_flops(b, t, h), sec_tf),
    ]


# ---------------------------------------------------------------- 實驗 B

def conv_flops(b: int, c: int, hw: int, kind: str) -> float:
    if kind == "dense":          # 3×3、C→C
        return 2.0 * b * hw * hw * 9 * c * c
    if kind == "depthwise_sep":  # 3×3 depthwise + 1×1 pointwise
        return 2.0 * b * hw * hw * (9 * c + c * c)
    raise ValueError(kind)


def run_experiment_b(args: argparse.Namespace, device: torch.device) -> list[Measurement]:
    b, c, hw = args.conv_batch, args.channels, args.spatial
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.randn(b, c, hw, hw, device=device, dtype=dtype)

    dense = nn.Conv2d(c, c, 3, padding=1).to(device=device, dtype=dtype)
    depthwise_sep = nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1, groups=c), nn.Conv2d(c, c, 1)
    ).to(device=device, dtype=dtype)

    sec_dense = time_forward(lambda: dense(x), device, args.repeats, args.warmup)
    sec_dw = time_forward(lambda: depthwise_sep(x), device, args.repeats, args.warmup)
    return [
        Measurement("dense 3×3 conv", conv_flops(b, c, hw, "dense"), sec_dense),
        Measurement("depthwise separable", conv_flops(b, c, hw, "depthwise_sep"), sec_dw),
    ]


# ---------------------------------------------------------------- 輸出

def print_pair(title: str, pair: list[Measurement], punchline: str) -> None:
    base, other = pair
    print(f"\n=== {title} ===")
    header = f"{'model':<34}{'GFLOPs':>10}{'ms':>10}{'達成 TFLOPS':>14}"
    print(header)
    print("-" * len(header))
    for m in pair:
        print(f"{m.label:<34}{m.flops / 1e9:>10.1f}{m.ms:>10.2f}{m.tflops:>14.2f}")
    flops_ratio = other.flops / base.flops
    time_ratio = base.seconds / other.seconds
    print(f"FLOPs 比（後/前）= {flops_ratio:.2f}×；時間比（前/後）= {time_ratio:.2f}×")
    print(punchline)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    print(f"device = {device}")
    if device.type == "cpu":
        print("⚠️  CPU 上跑：數字僅供功能驗證，平行度差異請在 GPU（CUDA/MPS）上看。")

    a = run_experiment_a(args, device)
    print_pair(
        "實驗 A：序列相依 vs 全平行（同規模、同序列）", a,
        "讀法：transformer FLOPs 更多卻更快——序列相依鏈（Amdahl 的 1−p）"
        "限制了 LSTM 能暴露的平行度，跟 FLOPs 無關。",
    )

    b = run_experiment_b(args, device)
    print_pair(
        "實驗 B：FLOPs ≠ 速度（dense vs depthwise separable conv）", b,
        "讀法：FLOPs ÷8~9 但時間遠沒有 ÷8——depthwise 的 AI 低、餵不滿運算單元，"
        "省下的 FLOPs 被記憶體頻寬吃掉。紙上 FLOPs 是 CPU 思維。",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLOPs vs 平行度 demo（讀書會 S5）")
    p.add_argument("--device", default=None, help="cuda / mps / cpu；預設自動")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    # 實驗 A
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    # 實驗 B
    p.add_argument("--conv-batch", type=int, default=32)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--spatial", type=int, default=56, help="feature map 邊長")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
