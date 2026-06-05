"""Decode batch sweep（讀書會 S3）。

示範自迴歸 decode 的記憶體頻寬本質：每一步都要把整份權重從 HBM 讀一遍，
所以 batch 小時「單步延遲」幾乎被權重讀取主宰、與 batch 無關；
加大 batch 等於用同一次權重讀取服務更多請求 → 吞吐（tokens/s）近乎線性上升，
直到算力追上（變 compute-bound）才趨緩。

用法：
    python run.py                                  # 預設 fp16、自動選 device
    python run.py --d-model 4096 --layers 32
    python run.py --peak-bw 3.35                   # 給定 HBM 頻寬 → 算理論下限

說明：用一疊 Linear（權重矩陣）模擬「讀一遍模型權重」。AI ≈ batch，
故 batch 跨過 ridge point（H100 約 ~300）後才會由 memory-bound 轉 compute-bound。
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True)
class Row:
    batch: int
    seconds: float
    weight_bytes: float
    d_model: int
    layers: int

    @property
    def step_ms(self) -> float:
        return self.seconds * 1e3

    @property
    def tokens_per_s(self) -> float:
        return self.batch / self.seconds

    @property
    def flops(self) -> float:
        # 每層一個 [B,d]x[d,d] matmul：2*B*d*d，共 layers 層
        return 2.0 * self.batch * self.d_model * self.d_model * self.layers


def build_weights(layers: int, d_model: int, dtype: torch.dtype, device: torch.device):
    # 縮放避免 relu 後數值爆掉；frozen 權重模擬已訓練好的模型
    scale = 1.0 / (d_model ** 0.5)
    return [torch.randn(d_model, d_model, device=device, dtype=dtype) * scale for _ in range(layers)]


def decode_step(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    for w in weights:
        x = torch.relu(x @ w)
    return x


def weight_bytes(layers: int, d_model: int, dtype: torch.dtype) -> float:
    size = torch.finfo(dtype).bits // 8
    return float(layers * d_model * d_model * size)


def time_step(weights, batch, d_model, dtype, device, repeats, warmup) -> float:
    x = torch.randn(batch, d_model, device=device, dtype=dtype)
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            decode_step(x, weights)
        if use_cuda:
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(repeats):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                decode_step(x, weights)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
            else:
                t0 = time.perf_counter()
                decode_step(x, weights)
                times.append(time.perf_counter() - t0)
    return statistics.median(times)


def print_table(rows: list[Row], dtype_name: str, peak_bw: float | None) -> None:
    wb = rows[0].weight_bytes
    print(f"\n=== Decode batch sweep (dtype={dtype_name}, "
          f"weights={rows[0].layers}層 x {rows[0].d_model}² = {wb / 1e9:.2f} GB) ===")
    floor_ms = None
    if peak_bw:
        floor_ms = wb / (peak_bw * 1e12) * 1e3
        print(f"理論單步下限 = 權重 {wb / 1e9:.2f} GB / {peak_bw:.2f} TB/s = {floor_ms:.2f} ms（純讀權重）")
    base = rows[0]
    header = f"{'batch':>7}{'step_ms':>11}{'tokens/s':>13}{'vs batch=1':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ratio = r.tokens_per_s / base.tokens_per_s
        print(f"{r.batch:>7}{r.step_ms:>11.2f}{r.tokens_per_s:>13.0f}{ratio:>11.1f}x")
    print("\n讀法：batch 小時 step_ms 幾乎不變（被權重讀取主宰）→ memory-bound；"
          "tokens/s 隨 batch 近乎線性上升 = 用同一次權重讀取服務更多請求。")
    if floor_ms:
        print(f"      step_ms 逼近但不低於理論下限 {floor_ms:.2f} ms，直到 batch 跨過 ridge "
              "才由算力接手。")


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args: argparse.Namespace) -> None:
    if args.dtype not in DTYPES:
        raise SystemExit(f"未知 dtype：{args.dtype}（可選 {list(DTYPES)}）")
    dtype = DTYPES[args.dtype]
    device = resolve_device(args.device)
    if device.type != "cuda":
        print("⚠️  非 CUDA 裝置：數字僅供功能驗證，記憶體頻寬效應需在 GPU 上才明顯。")

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    weights = build_weights(args.layers, args.d_model, dtype, device)
    wb = weight_bytes(args.layers, args.d_model, dtype)

    rows: list[Row] = []
    for b in batches:
        try:
            sec = time_step(weights, b, args.d_model, dtype, device, args.repeats, args.warmup)
        except RuntimeError as exc:
            print(f"略過 batch={b}：{exc}")
            continue
        rows.append(Row(b, sec, wb, args.d_model, args.layers))

    if rows:
        print_table(rows, args.dtype, args.peak_bw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode batch sweep（讀書會 S3）")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0；預設自動")
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128,256,512")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--peak-bw", type=float, default=None, help="HBM 峰值頻寬 (TB/s)，給定後算理論下限")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
