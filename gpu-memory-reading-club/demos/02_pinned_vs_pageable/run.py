"""H2D 頻寬：pinned vs pageable（讀書會 S2）。

量「主機 → GPU」(Host-to-Device) 傳輸頻寬，比較：
  - pageable（一般 host 記憶體）：OS 可換頁，DMA 不能直接搬 → 要先經 pinned bounce buffer
  - pinned（page-locked）：DMA 直達 → 較快、且能 async 與運算重疊

用法：
    python run.py
    python run.py --sizes-mb 1,4,16,64,256 --repeats 30

注意：pinned 記憶體與 H2D 傳輸都需要 CUDA。非 CUDA 裝置會直接提示並結束。
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Row:
    size_mb: float
    pageable_s: float
    pinned_s: float
    nbytes: int

    def gbps(self, seconds: float) -> float:
        return self.nbytes / seconds / 1e9

    @property
    def speedup(self) -> float:
        return self.pageable_s / self.pinned_s


def bench_h2d(n_elems: int, pinned: bool, device: torch.device,
              repeats: int, warmup: int, dtype=torch.float32) -> float:
    src = torch.empty(n_elems, dtype=dtype, pin_memory=pinned)
    src.uniform_()
    dst = torch.empty(n_elems, dtype=dtype, device=device)

    for _ in range(warmup):
        dst.copy_(src, non_blocking=pinned)
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=pinned)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def print_table(rows: list[Row]) -> None:
    print("\n=== H2D 頻寬：pinned vs pageable ===")
    header = f"{'size':>9}{'pageable':>13}{'pinned':>13}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r.size_mb:>7.0f}MB{r.gbps(r.pageable_s):>11.1f}GB/s"
              f"{r.gbps(r.pinned_s):>11.1f}GB/s{r.speedup:>9.2f}x")
    print("\n讀法：pinned 通常快 ~1.5–2×；更重要的是 pinned 才能 async、與運算重疊（見 S4 壓軸 demo）。")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("⚠️  此 demo 需要 CUDA GPU（pinned 記憶體與 H2D 傳輸皆需 CUDA）。"
              "請在 GPU 環境（如 RunPod）執行。")
        return
    device = torch.device("cuda")
    sizes = [float(x) for x in args.sizes_mb.split(",") if x.strip()]
    itemsize = torch.empty(0, dtype=torch.float32).element_size()

    rows: list[Row] = []
    for mb in sizes:
        n = int(mb * 1e6 / itemsize)
        try:
            pa = bench_h2d(n, False, device, args.repeats, args.warmup)
            pi = bench_h2d(n, True, device, args.repeats, args.warmup)
        except RuntimeError as exc:
            print(f"略過 {mb}MB：{exc}")
            continue
        rows.append(Row(mb, pa, pi, n * itemsize))
    if rows:
        print_table(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2D 頻寬 pinned vs pageable（讀書會 S2）")
    p.add_argument("--sizes-mb", default="1,4,16,64,256")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
