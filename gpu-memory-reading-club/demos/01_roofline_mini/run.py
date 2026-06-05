"""Roofline mini-demo（讀書會 S1）。

量測不同形狀矩陣乘法的「達成 TFLOPS」與「算術強度（FLOPs/Byte）」，
看它落在 memory-bound（瘦長 / GEMV）還是 compute-bound（大方陣）區。

用法：
    python run.py                                   # 預設 fp16、自動選 device
    python run.py --dtype bf16
    python run.py --peak-tflops 990 --peak-bw 3.35  # 給定硬體峰值 → 標 ridge point 與利用率

說明：bytes 採「每個運算元各讀一次、C 各寫一次」的理想下界模型
（FLOPs = 2·M·K·N；Bytes = dtype_size·(M·K + K·N + M·N)）。
真實 kernel 因 tiling 可能重讀資料，這裡用標準模型建立 AI 直覺即可。
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class Shape:
    """一個矩陣乘法的形狀：C[M,N] = A[M,K] @ B[K,N]。"""

    label: str
    m: int
    k: int
    n: int


# 從「GEMV / 瘦長」到「大方陣」——涵蓋 memory-bound 到 compute-bound 的光譜。
DEFAULT_SHAPES = (
    Shape("GEMV   (1x4096*4096)", 1, 4096, 4096),
    Shape("skinny (8x4096*4096)", 8, 4096, 4096),
    Shape("skinny (64x4096*4096)", 64, 4096, 4096),
    Shape("square 512", 512, 512, 512),
    Shape("square 1024", 1024, 1024, 1024),
    Shape("square 2048", 2048, 2048, 2048),
    Shape("square 4096", 4096, 4096, 4096),
    Shape("square 8192", 8192, 8192, 8192),
)


@dataclass(frozen=True)
class Result:
    shape: Shape
    seconds: float
    flops: float
    bytes_moved: float

    @property
    def tflops(self) -> float:
        return self.flops / self.seconds / 1e12

    @property
    def gbps(self) -> float:
        return self.bytes_moved / self.seconds / 1e9

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes_moved


def flops_of(shape: Shape) -> float:
    return 2.0 * shape.m * shape.k * shape.n


def bytes_of(shape: Shape, dtype: torch.dtype) -> float:
    size = torch.finfo(dtype).bits // 8
    elems = shape.m * shape.k + shape.k * shape.n + shape.m * shape.n
    return float(size * elems)


def time_matmul(
    shape: Shape,
    dtype: torch.dtype,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> float:
    """回傳單次 matmul 的中位數秒數。CUDA 用 cuda.Event、CPU 用 perf_counter。"""
    a = torch.randn(shape.m, shape.k, device=device, dtype=dtype)
    b = torch.randn(shape.k, shape.n, device=device, dtype=dtype)
    use_cuda = device.type == "cuda"

    for _ in range(warmup):
        _ = a @ b
    if use_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeats):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = a @ b
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)  # ms -> s
        else:
            t0 = time.perf_counter()
            _ = a @ b
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def ridge_point(peak_tflops: float | None, peak_bw: float | None) -> float | None:
    """ridge point = 峰值算力 / 峰值頻寬（FLOPs/Byte）。peak_bw 單位 TB/s。"""
    if not peak_tflops or not peak_bw:
        return None
    return peak_tflops / peak_bw


def classify(ai: float, ridge: float | None) -> str:
    if ridge is None:
        return "-"
    return "memory" if ai < ridge else "compute"


def print_table(results: list[Result], dtype_name: str, ridge: float | None,
                peak_tflops: float | None, peak_bw: float | None) -> None:
    print(f"\n=== Roofline mini-demo (dtype={dtype_name}) ===")
    if ridge is not None:
        print(f"ridge point = {peak_tflops:.0f} TFLOPS / {peak_bw:.2f} TB/s "
              f"= {ridge:.0f} FLOPs/Byte")
    header = f"{'shape':<22}{'AI':>10}{'TFLOPS':>10}{'GB/s':>10}{'bound':>9}"
    if ridge is not None:
        header += f"{'util':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        bound = classify(r.arithmetic_intensity, ridge)
        line = (f"{r.shape.label:<22}{r.arithmetic_intensity:>10.1f}"
                f"{r.tflops:>10.1f}{r.gbps:>10.0f}{bound:>9}")
        if ridge is not None:
            # compute-bound 看算力利用率、memory-bound 看頻寬利用率
            if bound == "compute" and peak_tflops:
                util = r.tflops / peak_tflops
            elif peak_bw:
                util = r.gbps / (peak_bw * 1000.0)
            else:
                util = 0.0
            line += f"{util * 100:>7.0f}%"
        print(line)
    print("\n讀法：AI 小 → memory-bound（瘦長/GEMV，達成 TFLOPS 低）；"
          "AI 大 → compute-bound（大方陣，逼近峰值算力）。")


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
        print("⚠️  非 CUDA 裝置：數字僅供功能驗證，roofline 對照請在 GPU 上跑。")

    results: list[Result] = []
    for shape in DEFAULT_SHAPES:
        try:
            sec = time_matmul(shape, dtype, device, args.repeats, args.warmup)
        except RuntimeError as exc:  # 多半是 OOM：跳過該形狀，繼續其餘
            print(f"略過 {shape.label}：{exc}")
            continue
        results.append(Result(shape, sec, flops_of(shape), bytes_of(shape, dtype)))

    ridge = ridge_point(args.peak_tflops, args.peak_bw)
    print_table(results, args.dtype, ridge, args.peak_tflops, args.peak_bw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roofline mini-demo（讀書會 S1）")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0；預設自動")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="硬體峰值算力 (TFLOPS)，給定後標 ridge point 與利用率")
    p.add_argument("--peak-bw", type=float, default=None,
                   help="硬體峰值記憶體頻寬 (TB/s)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
