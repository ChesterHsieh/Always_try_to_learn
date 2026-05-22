# Learning Beyond Gradients — 參考筆記

## 原文連結
- GitHub: [learning-beyond-gradient.md](https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/learning-beyond-gradient.md)
- 完整專案: [Trinkle23897/learning-beyond-gradients](https://github.com/Trinkle23897/learning-beyond-gradients)

## 核心概念

### Heuristic Learning (HL)
不訓練神經網路、不靠梯度的學習典範：
- **主體是程式碼**：策略用代碼表示，不是神經網路參數
- **迴圈更新**：靠環境反饋與直接修改代碼來改進
- **可解釋性**：代碼策略能轉譯成人可讀的解釋
- **樣本效率高**：單次代碼更新可直接跳到新策略

### 為什麼現在可行？

**LLM Agent 改變了維護成本曲線**

過去手寫複雜規則不經濟，但大語言模型能持續迭代軟體系統，讓曾經不可行的啟發式方法變得可行。

## HL vs. Deep RL

| 維度 | Deep RL | HL |
|------|---------|-----|
| 策略 | 神經網路參數 | 代碼規則 |
| 狀態 | 隱式觀察 | 顯式變數與偵測器 |
| 更新 | 梯度下降 | 直接修改代碼 |
| 記憶 | 經驗重放 | 顯式試驗日誌與重放 |

## 優勢

1. **可解釋性** — 代碼就是解釋
2. **樣本效率** — 跳過梯度累積的過程
3. **迴歸測試** — 舊能力變成可測試的
4. **避免災難性遺忘** — 能力被編入規則與測試，不只是權重

## 實驗結果（原文示例）

- **Atari Breakout**: 387 → 507 → 839 → 864（逐次改進）
- **MuJoCo Ant**: 6000+ 分（解釋性行走與 MPC）
- **Atari57**: 中位數接近深度 RL 基線

## 本專案落地

本 `heuristic-learning/` 專案把 HL 命題落在 **LunarLander-v3** 上：

| controller | mean return | landing rate | 訓練？ |
|-----------|-------------|--------------|---------|
| baseline_v1 | **+264.4** | **68%** | 無（手寫規則） |
| rl_ppo | +245.8 | 58% | 有（200k 步） |

零訓練的 baseline_v1 與梯度訓練的 rl_ppo **同級**，驗證 HL 的核心命題。

## 延伸思考

> "凡能持續迭代者，皆始解決"

未來的系統可結合：
- 淺層、快速的神經網路（System 1）
- 啟發式學習模組（System 1）
- LLM Agent 提供反饋（System 2）

實現線上學習而不陷入災難性遺忘。
