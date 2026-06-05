"""ASR encoder vs decoder 代理示範（讀書會 S3）。

同樣處理長度 T 的序列、同樣的權重與層數、同樣的總 FLOPs：
  - encoder：T 個 frame「一次」平行算（每層一個 [T,d]x[d,d] 大 GEMM）
  - decoder：T 個 token「逐一」算（每步一個 [1,d]x[d,d] 的 GEMV，共 T 步）
結果：decoder 的 wall-clock 遠高於 encoder——不是因為算得多，而是因為
逐 token 的序列相依讓每步都要重讀權重、又吃滿 kernel launch 開銷（memory-bound）。

這正是「為什麼 attention decoder 的 ASR（如 Whisper）比 CTC（如 wav2vec2）慢」的核心：
CTC 的 encoder 一次輸出所有 frame 的字元機率、無自迴歸 decode。

用法：
    python asr_proxy.py                 # 預設 fp16、自動選 device
    python asr_proxy.py --frames 256 --d-model 1024 --layers 12

注意：這是「結構性」代理，不含真實 Whisper 的 cross-attention 與 KV cache；
重點在呈現「平行 vs 序列」造成的延遲差異。
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def build_weights(layers, d_model, dtype, device):
    scale = 1.0 / (d_model ** 0.5)
    return [torch.randn(d_model, d_model, device=device, dtype=dtype) * scale for _ in range(layers)]


def encoder_pass(frames, weights):
    # 一次吃整段：x 形狀 [T, d]
    x = frames
    for w in weights:
        x = torch.relu(x @ w)
    return x


def decoder_pass(frames, weights):
    # 逐 token：T 步，每步處理一列 [1, d]
    t = frames.shape[0]
    outs = []
    for i in range(t):
        x = frames[i: i + 1]
        for w in weights:
            x = torch.relu(x @ w)
        outs.append(x)
    return outs


def _time(fn, use_cuda, repeats, warmup) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if use_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1e3)
            else:
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
    return statistics.median(times)


def resolve_device(name):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args):
    if args.dtype not in DTYPES:
        raise SystemExit(f"未知 dtype：{args.dtype}")
    dtype = DTYPES[args.dtype]
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    if not use_cuda:
        print("⚠️  非 CUDA 裝置：數字僅供功能驗證；序列 vs 平行的差距在 GPU 上更顯著。")

    weights = build_weights(args.layers, args.d_model, dtype, device)
    frames = torch.randn(args.frames, args.d_model, device=device, dtype=dtype)

    enc = _time(lambda: encoder_pass(frames, weights), use_cuda, args.repeats, args.warmup)
    dec = _time(lambda: decoder_pass(frames, weights), use_cuda, max(3, args.repeats // 4), 1)

    print(f"\n=== ASR encoder vs decoder 代理 "
          f"(T={args.frames}, d={args.d_model}, layers={args.layers}, dtype={args.dtype}) ===")
    print(f"{'':<26}{'wall-clock':>12}{'每 frame':>14}")
    print("-" * 52)
    print(f"{'encoder（平行，1 次）':<24}{enc * 1e3:>10.2f} ms{enc / args.frames * 1e6:>11.1f} µs")
    print(f"{'decoder（序列，' + str(args.frames) + ' 步）':<24}{dec * 1e3:>10.2f} ms{dec / args.frames * 1e6:>11.1f} µs")
    print(f"\ndecoder / encoder ≈ {dec / enc:.1f}x 慢（總 FLOPs 相同）。")
    print("結論：決定速度的是「能不能平行」與「記憶體存取型態」，不是 FLOPs 總量。")


def parse_args():
    p = argparse.ArgumentParser(description="ASR encoder vs decoder 代理（讀書會 S3）")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--device", default=None)
    p.add_argument("--frames", type=int, default=256, help="序列長度 T")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
