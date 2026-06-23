# Demo 05 — FLOPs vs 平行度（S5）

一句話：**FLOPs 不是速度。** 兩個實驗各打一個迷思：

- **實驗 A**（序列相依 vs 全平行）：同規模 LSTM 與 Transformer block 跑同一段序列。Transformer 理論 FLOPs 還比較多（~1.75×），但因為 T 軸全平行，wall-clock 反而更快、達成 TFLOPS 高數倍——序列相依鏈（Amdahl 的 1−p）才是瓶頸。
- **實驗 B**（FLOPs ≠ 速度）：dense 3×3 conv vs depthwise separable conv。紙上 FLOPs ÷8~9，實際時間遠沒有 ÷8——depthwise 的算術強度低、餵不滿運算單元，省下的 FLOPs 被記憶體頻寬吃掉（MobileNet 的目標硬體本來就是手機 CPU）。

## 怎麼跑

```bash
python run.py                          # 自動選 device（cuda > mps > cpu）
python run.py --seq-len 1024           # 拉長序列，實驗 A 差距更大
python run.py --device cpu             # 功能驗證
```

CUDA 上用 fp16，MPS/CPU 用 fp32。計時：warmup 後同步圍住計時區間、取中位數。

## 實測（Apple M2 / MPS、fp32、預設參數）

```
=== 實驗 A：序列相依 vs 全平行 ===          GFLOPs      ms    達成 TFLOPS
LSTM        (T=512 步序列相依)               17.2    47.69       0.36
Transformer (T=512 全平行)                   30.1    24.92       1.21
→ FLOPs 多 1.75×，時間反而快 1.9×；達成 TFLOPS 差 3.4×

=== 實驗 B：dense vs depthwise separable ===
dense 3×3 conv                              118.4    40.80       2.90
depthwise separable                          13.6    11.04       1.23
→ FLOPs ÷8.7，時間只 ÷3.7；達成 TFLOPS 掉 2.4×
```

CUDA 卡上差距預期更大：實驗 A 的 LSTM 還要付 T 次 kernel launch 開銷、瘦長 GEMM 餵不滿 tensor core；實驗 B 的 dense conv 經 im2col 是又大又方的 GEMM、可吃 tensor core，depthwise 不行。

## 怎麼解讀（接回 S5 三個硬體問題）

| | 平行度在哪個軸 | AI（FLOPs/Byte） | 序列鏈長 |
|---|---|---|---|
| LSTM | B×H（T 軸是鏈） | 低（瘦長 GEMM） | T 步 |
| Transformer | B×T×H | 高（大 GEMM） | 1 |
| dense conv | B×HW×C，GEMM 方正 | 高 | 1 |
| depthwise | channel 軸被拆散 | 個位數 | 1 |

> FLOPs 相同（甚至更多）的模型可以快好幾倍——決定速度的是**暴露的平行度**與**每 byte 的運算次數**。這就是「模型設計 × 計算機結構」互相影響的量化版。
