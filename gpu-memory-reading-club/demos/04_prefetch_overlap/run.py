"""H2D 與運算重疊：prefetch / overlap（讀書會 S4，壓軸 demo）。

同樣一批批資料要「搬到 GPU 再算」，比較兩種做法：
  - naive：搬完這批才算這批，搬與算在同一條 stream 上序列進行
  - overlapped：用第二條 copy stream 預先把後續批次搬進來（prefetch），
                compute stream 一邊算、copy stream 一邊搬 → 把搬運延遲藏在運算後面

這正是「預先把資料搬到對的地方」能省多少時間的量化：當搬運與運算時間相近時，
overlap 理想上可逼近 ~2× 吞吐。

用法：
    python run.py
    python run.py --num 16 --rows 4096 --d 4096 --iters 6

注意：需要 CUDA GPU 與 pinned 記憶體；非 CUDA 裝置會直接提示並結束。
"""
from __future__ import annotations

import argparse
import time

import torch


def make_host_batches(num: int, rows: int, d: int, dtype) -> list[torch.Tensor]:
    # pinned host 記憶體，才能 async 搬
    return [torch.empty(rows, d, dtype=dtype, pin_memory=True).uniform_() for _ in range(num)]


def compute(x: torch.Tensor, w: torch.Tensor, iters: int) -> torch.Tensor:
    y = x
    for _ in range(iters):
        y = torch.relu(y @ w)
    return y


def run_naive(host, w, iters, device) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for h in host:
        dev = h.to(device, non_blocking=False)  # H2D：與運算在同一條 stream 上序列化
        compute(dev, w, iters)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_overlapped(host, w, iters, device) -> float:
    n = len(host)
    copy_s = torch.cuda.Stream()
    comp_s = torch.cuda.Stream()
    bufs = [torch.empty_like(h, device=device) for h in host]  # 每批一個 buffer，免 WAR 衝突
    copy_done = [torch.cuda.Event() for _ in range(n)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # copy stream 盡量往前跑，把每批搬進來並記錄事件
    with torch.cuda.stream(copy_s):
        for i in range(n):
            bufs[i].copy_(host[i], non_blocking=True)
            copy_done[i].record(copy_s)
    # compute stream 等到該批搬完再算 → 與後續批的搬運重疊
    for i in range(n):
        comp_s.wait_event(copy_done[i])
        with torch.cuda.stream(comp_s):
            compute(bufs[i], w, iters)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("⚠️  此壓軸 demo 需要 CUDA GPU（stream overlap 與 pinned async 皆需 CUDA）。"
              "請在 GPU 環境（如 RunPod）執行。")
        return
    device = torch.device("cuda")
    dtype = torch.float16
    host = make_host_batches(args.num, args.rows, args.d, dtype)
    w = torch.randn(args.d, args.d, device=device, dtype=dtype) * (1.0 / args.d ** 0.5)

    # warmup
    run_naive(host[:2], w, args.iters, device)
    run_overlapped(host[:2], w, args.iters, device)

    naive = min(run_naive(host, w, args.iters, device) for _ in range(args.repeats))
    over = min(run_overlapped(host, w, args.iters, device) for _ in range(args.repeats))

    print(f"\n=== Prefetch / overlap (num={args.num}, batch={args.rows}x{args.d}, "
          f"iters={args.iters}, fp16) ===")
    print(f"{'naive（搬完才算）':<22}{naive * 1e3:>9.1f} ms")
    print(f"{'overlapped（邊搬邊算）':<20}{over * 1e3:>9.1f} ms")
    print(f"\n加速 ≈ {naive / over:.2f}x（搬運被藏在運算後面）。")
    print("搬運與運算時間越接近，overlap 的效益越大；理想上限約 ~2×。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch / overlap 壓軸 demo（讀書會 S4）")
    p.add_argument("--num", type=int, default=16, help="批次數")
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--d", type=int, default=4096)
    p.add_argument("--iters", type=int, default=6, help="每批的運算量（matmul 次數）")
    p.add_argument("--repeats", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
