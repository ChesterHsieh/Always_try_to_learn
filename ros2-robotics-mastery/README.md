# ROS 2 Robot Platform Engineer

從 ROS 2 通訊核心 → 執行時平台 → 可靠性除錯 → 可觀測性 → Jetson 真機——一條對齊 **Robot Platform Software Engineer** JD 的技能樹。

**最終目標**：在 Jetson Orin Nano 上運營一套 always-on 機器人執行時（systemd + lifecycle + CAN + 健康監控 + 雲端觀測管線），連續運行 72 小時並通過混沌演練。

> **v2（2026-09-01）**：對齊 Anvil Robotics 類型的 platform JD 重構。原 manipulation / Isaac 主線濃縮為支線；新增執行時平台、可靠性除錯、可觀測性、CAN/硬體介面、C++ 生產力五章，並補上 138 題檢核題庫（quiz gate：任務全勾＋答題 ≥80% 才點亮節點）。v1 見 git 歷史。

---

## 快速開始

```bash
# 1. 打開技能樹（或直接開已發布的 Artifact 連結）
cd skill-tree && python3 -m http.server 8080   # 開 http://localhost:8080/ros2-platform.html

# 2. 建立開發環境
cd docker && docker compose build && docker compose up -d
```

## 目錄

| 路徑 | 內容 |
|---|---|
| [`skill-tree/`](skill-tree/) | ⭐ 互動技能樹 v2（46 節點 / 138 題）。資料源 `tree_data.py` + `quiz_*.py`，`python3 build.py` 產生 `ros2-platform.html` |
| [`projects/platform-lab/`](projects/platform-lab/) | ⭐ **Capstone 規格書**：always-on runtime、混沌演練、72h 運行、postmortem |
| [`projects/can-lab/`](projects/can-lab/) | ⭐ SocketCAN：vcan → Jetson 實體 CAN（含購物清單與除錯速查） |
| [`docs/04-jetson-orin-nano.md`](docs/04-jetson-orin-nano.md) | ⚠️ Jetson 刷機、韌體地雷 |
| [`docs/05-version-matrix.md`](docs/05-version-matrix.md) | 版本相容矩陣——裝東西前先看 |
| [`docs/`](docs/) 其他 | v1 時代的環境/課程/雲端筆記，仍然有效 |
| [`docker/`](docker/) | 開發環境（Dockerfile + compose） |
| [`ros2_ws/src/`](ros2_ws/src/) | ROS 2 package（掛載進容器） |

## 技能樹 v2 結構

46 節點、10 章、225 h（主線 198 h ≈ 13 週 @ 15h/週）。

```
序章 · 立足點            4 節點  12h  Docker 環境、colcon、CLI
第一章 · 通訊核心         7 節點  36h  ⭐ topics/QoS/executor/discovery/tf2
第二章 · 執行時平台       5 節點  23h  ⭐ systemd、lifecycle、bringup 編排、健康監控◆
第三章 · 可靠性除錯       5 節點  23h  ⭐ strace/perf/OOM、混沌演練◆、RT 基礎
第四章 · 可觀測性與交付    6 節點  24h  日誌管線、MCAP、Prometheus、Foxglove、CI◆、雲管線
第五章 · Jetson 部署      4 節點  18h  刷機、容器、Mac↔Jetson 分散式◆
第六章 · 硬體介面         5 節點  24h  ⭐ SocketCAN◆、Jetson CAN、ros2_control、乾淨 API
第七章 · C++ 生產力       4 節點  20h  RAII/rclcpp/CMake/sanitizers◆（可與中段並行）
支線 · 模擬與手臂鳥瞰      3 節點  15h  URDF/Gazebo/MoveIt/Isaac 地圖級理解
終章 · Platform Capstone  3 節點  30h  🏆 72h runtime、postmortem 兩版本、面試演武
```

### 與 JD 的對照

| JD Focus Area | 章節 |
|---|---|
| Robot runtime platform（orchestration/logging/health/lifecycle） | 第二章 + 第四章 |
| Reliability & debugging（24/7、跨層 root-cause） | 第三章 + 終章混沌演練 |
| Hardware interfaces（clean stable APIs） | 第六章（A6-4/A6-5 就是這句話的實作） |
| Tooling（ROS2、C++/Python、containers） | 第一、五、七章 |
| Observability & cloud | 第四章（資料工程背景的主場） |
| 對非技術受眾溝通 | A8-3、A9-2（postmortem 主管版）、A9-3 |

## 維護技能樹

```bash
cd skill-tree
vim tree_data.py     # 節點；題庫在 quiz_a*.py
python3 build.py     # 驗證 + 產生 ros2-platform.html（驗證不過 exit 1）
```

進度存瀏覽器 localStorage（key: `ros2-platform-skilltree-v2`）。定期按頁面右上「進度」把 JSON 複製存檔；換瀏覽器貼回「載入」還原。

## 硬體與成本

| 項目 | 狀態 | 成本 |
|---|---|---|
| MacBook (Apple Silicon) + Jetson Orin Nano | ✅ 已有 | — |
| NVMe SSD（Jetson rootfs，強烈建議） | 建議補 | ~$30 |
| SN65HVD230 CAN transceiver ×2 + 線材 | 第六章需要 | ~$5 |
| 雲端物件儲存（S3/MinIO 自架皆可） | 第四章 | ~$0 |

---

*v2.0.0｜2026-09-01｜ROS 2 Jazzy（容器）＋ JetPack 6.x｜46 節點 / 138 題 / 225 h*
