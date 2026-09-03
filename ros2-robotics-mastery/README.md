# ROS 2 Robotics Mastery

從 ROS 2 通訊核心走到 Isaac Sim，最終操作真實機械手臂 —— 一條可勾選、會解鎖、能量化的技能樹。

**最終目標**：在 Isaac Sim 中訓練策略，部署到 ROS 2，驅動真實手臂完成抓取任務。

---

## 快速開始

```bash
# 1. 打開技能樹（建議用 http server，localStorage 才穩）
cd skill-tree && python3 -m http.server 8080
# 瀏覽器開 http://localhost:8080

# 2. 建立開發環境
cd docker && docker compose build && docker compose up -d
docker compose exec ros2 bash

# 3. 從技能樹的第一個節點開始（點右上角「▶ 下一步」）
```

---

## 目錄

| 路徑 | 內容 |
|---|---|
| [`skill-tree/`](skill-tree/) | ⭐ 互動式技能樹（69 節點）。`index.html` 直接開 |
| [`docs/00-roadmap.md`](docs/00-roadmap.md) | 路線圖、里程碑、與既有專案的接口 |
| [`docs/01-environment-setup.md`](docs/01-environment-setup.md) | Mac (Apple Silicon) + Docker + ROS 2 Jazzy |
| [`docs/02-curriculum.md`](docs/02-curriculum.md) | 課程資源索引（已逐一驗證，含「不要浪費時間」清單） |
| [`docs/03-cloud-isaac-sim.md`](docs/03-cloud-isaac-sim.md) | 雲端 GPU 跑 Isaac Sim + 成本控制 |
| [`docs/04-jetson-orin-nano.md`](docs/04-jetson-orin-nano.md) | ⚠️ Jetson 刷機、韌體地雷、Isaac ROS 版本分歧 |
| [`docs/05-version-matrix.md`](docs/05-version-matrix.md) | ⭐ **版本相容矩陣 — 裝東西前先看這個** |
| [`docker/`](docker/) | 開發環境（Dockerfile + compose） |
| [`ros2_ws/src/`](ros2_ws/src/) | 你的 ROS 2 package 放這裡（掛載進容器） |
| [`projects/`](projects/) | 各章實作專案的規格與筆記 |
| [`projects/P7-pickleball-tracker/`](projects/P7-pickleball-tracker/) | ⭐ **P7 Pickleball Tracker**：單鏡頭球體追蹤 + AI 計分，全 C++，面試展示專案 |

---

## 技能樹

69 個節點、11 章、443 小時（主線 289 h / 支線 154 h）。

```
序章 · 立足點            5 節點   16 h   Docker 環境、colcon、CLI
第一章 · 通訊核心        17 節點   82 h   ⭐ pub/sub、service、action、QoS、tf2、lifecycle、executor
第二章 · 機器人建模       5 節點   20 h   URDF、xacro、RViz、物理屬性
第三章 · 模擬世界         5 節點   25 h   Gazebo Harmonic、ros_gz_bridge、感測器
第四章 · 控制迴路         5 節點   28 h   ⭐ ros2_control（模擬↔真機的抽象層）
第五章 · 感知             7 節點   58 h   相機、點雲、深度、DNN 推論、⭐ P7 Pickleball 追蹤
第六章 · 導航（支線）      4 節點   25 h   SLAM、Nav2、Behavior Tree
第七章 · 機械手臂         6 節點   47 h   ⭐⭐ 運動學理論、MoveIt 2、Pick & Place
第八章 · Isaac 生態       6 節點   49 h   ⭐⭐ Isaac Sim、USD、Isaac Lab RL、sim-to-real
第九章 · Jetson 部署      5 節點   40 h   刷機、Isaac ROS GEM、分散式部署、P7 即時追蹤
終章 · 真實世界           4 節點   53 h   🏆 SO-101、LeRobot、Capstone
```

### 節點類型

| 樣式 | 意義 |
|---|---|
| 小圓 | 一般節點 |
| 大圓 | 關鍵節點（notable） |
| **菱形** | **樞紐節點（keystone）** — 整條路線的轉折點 |
| 虛線邊框 | 選修支線 |

### 樞紐節點（最重要的 7 個）

| 節點 | 為什麼 |
|---|---|
| **A1-8 QoS 策略** | 90% 的「收不到訊息」都是它。也是你資料工程背景的最佳切入點 |
| **A1-14 tf2 座標系統** | ★★ 全樹最重要的單一節點。不通則 URDF/Gazebo/Nav2/MoveIt 全變玄學 |
| **A1-K 通訊系統精通** | 「master ROS 溝通系統」的畢業考 |
| **A3-2 ros_gz_bridge** | 模擬世界與 ROS 世界的唯一通道 |
| **A4-1 ros2_control** | ★★ 模擬與真機的共同抽象層。整條路線的樞紐 |
| **A7-5 Pick and Place** | 模擬階段的畢業考 |
| **A8-5 Isaac Lab RL** | 你 JAX/RL 背景的正面對撞點 |

---

## 版本決策（2026-07-31 查證）

| 項目 | 選擇 | 一句話理由 |
|---|---|---|
| ROS 2 | **Jazzy Jalisco** | LTS 到 2029；Isaac Sim 6.0 官方推薦；生態最完整 |
| 模擬器 | **Gazebo Harmonic** | Jazzy 的官方配對（⚠️ Gazebo Classic 已 EOL） |
| 高保真模擬 | **Isaac Sim 6.0.1**（雲端） | ❌ 不支援 macOS；❌ A100/H100 無 RT Core 不能用 |
| Jetson Orin Nano | **JetPack 6.2.2 + Humble** | ⚠️ Isaac ROS 4.x 已放棄 Orin，只支援 Thor |

> 完整說明見 [docs/05-version-matrix.md](docs/05-version-matrix.md)。
> **踩雷成本最高的三件事**：Gazebo Classic 教學、A100/H100 跑 Isaac Sim、Jetson 韌體版本。

---

## 硬體與成本

| 項目 | 狀態 | 成本 |
|---|---|---|
| MacBook Air (Apple Silicon) | ✅ 已有 | — |
| Jetson Orin Nano | ✅ 已有 | — |
| NVMe SSD 128GB+（Jetson 用） | 建議補 | ~$30 |
| 雲端 GPU（Isaac Sim） | 按需租用 | ~$1/hr，第八章估 $50–100 |
| SO-101 手臂（leader+follower） | 選修，終章 | ~US$230 |

---

## 與倉庫其他專案的關係

| 既有專案 | 接口 |
|---|---|
| `robot-mujoco-control` | A3-5 支線把它接上 `mujoco_ros2_control` |
| `ai-monitor-system` | A1-K 的故障診斷腳本 = probe 框架的機器人版 |
| `heuristic-learning` | A8-5 重做一次「手寫規則 vs 訓練 policy」對照 |
| `learn-jax` / purejaxrl | A8-5 的平行化環境思路直接適用 |
| `lora-image-gen` | A8-1 的 RunPod 遠端 GPU 模式完全複用 |
| `DDIA-in-real` | A1-8 QoS、A1-12 rosbag 的心智模型直接對應 |
| `gpu-memory-reading-club` | A1-11 零拷貝、A9-3 共享記憶體瓶頸分析 |

---

## 維護技能樹

技能樹的單一資料來源是 `skill-tree/tree_data.py`。

```bash
cd skill-tree
vim tree_data.py       # 改節點、加資源、調時數
python3 build.py       # 產生 skill-tree.json 與 index.html（含驗證）
```

`build.py` 會檢查：重複 id、不存在的前置、依賴循環、空任務清單。
**進度以 node id 為 key，重新 build 不會遺失。**

---

## 進度備份

技能樹的進度存在瀏覽器 localStorage。定期按「匯出進度」→
把 `ros2-skilltree-progress.json` 存進 `skill-tree/` 並 commit。

---

*建立日期：2026-07-31｜更新：2026-09-03 加入 P7｜ROS 2 Jazzy Jalisco｜69 節點 / 443 小時*
