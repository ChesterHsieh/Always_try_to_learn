# 雲端跑 Isaac Sim（Mac 使用者的唯一路徑）

> 對應技能樹 A8-1。

---

## 一、為什麼一定要雲端

Isaac Sim 6.0.1 的硬性限制：

| 限制 | 內容 |
|---|---|
| 作業系統 | Ubuntu 22.04 / 24.04、Windows 11。**❌ 不支援 macOS** |
| GPU | **必須有 RT Core**。官方原文：*"GPUs without RT Cores (A100, H100) are not supported."* |
| 最低配置 | RTX 4080 / 16 GB VRAM / 32 GB RAM |
| aarch64 建置 | **僅 DGX Spark** → Jetson 完全不能跑 |

好消息：**Isaac Sim WebRTC Streaming Client 有 macOS `.dmg`**。
你不能在 Mac 上跑 Isaac Sim，但可以從 Mac 流暢地操作雲端的 Isaac Sim。

---

## 二、GPU 選型（最容易犯錯的地方）

**A100 與 H100 是雲端最常見的 GPU，也是唯一絕對不能用的。**
它們是純運算卡，沒有 RT Core，Isaac Sim 的 RTX 渲染跑不起來。

| GPU | RT Core | RunPod 參考價/hr | 評價 |
|---|---|---|---|
| RTX 4090 24GB | ✅ | ~$0.69 | 💰 最便宜可用，24GB 對學習夠了 |
| L40 48GB | ✅ | ~$0.82 | 好 |
| RTX 6000 Ada 48GB | ✅ | ~$0.84 | 很好 |
| **L40S 48GB** | ✅ | **~$0.99** | ⭐ **推薦：AWS 官方驗證機型也是它** |
| RTX 5090 32GB | ✅ | ~$0.99 | 可用 |
| RTX PRO 6000 96GB | ✅ | ~$1.99 | 頂規，學習用不需要 |
| A40 48GB | ✅ (Ampere) | ~$0.44 | 最便宜的 48GB，較舊但可用 |
| L4 24GB | ✅ | ~$0.39 | ⚠️ 渲染太弱，體驗差 |
| **A100 / H100** | ❌ | — | ❌ **不能跑** |

AWS 對應機型：`g6e.2xlarge`（L40S，$2.242/hr）或 `g7e.8xlarge`（RTX PRO 6000）。
比 RunPod 貴約 2 倍，除非有 AWS credit 否則沒必要。

---

## 三、方案 A：RunPod / Vast（推薦，你已經熟悉）

你已經在 lora-image-gen 專案用 RunPod 跑 ComfyUI，同一套心智模型直接搬過來。

```bash
# 在雲端 GPU 機器上
docker pull nvcr.io/nvidia/isaac-sim:6.0.1

docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:6.0.1

# 容器內，啟動 WebRTC 串流模式
./runheadless.sh -v
```

前置需求：Docker + NVIDIA Container Toolkit + NGC 帳號（免費註冊，拉 image 要 API key）。

Mac 端：下載並安裝 **Isaac Sim WebRTC Streaming Client 2.0.0 (.dmg)**，連到雲端機器的 IP。
或直接用 Chromium 系瀏覽器開 web viewer。

> ⚠️ 串流需要 NVENC。這也是 A100 不能用的另一個原因。

---

## 四、方案 B：NVIDIA Brev Isaac Launchable（最省事）

瀏覽器直接開，內建 VS Code + Isaac Sim + Isaac Lab + Kit App Streaming，
不用自己搞 Docker。

- 🔗 https://github.com/isaac-sim/isaac-launchable
- ⚠️ 目前打包的是 **Isaac Sim 5.1 + Isaac Lab 2.3**（落後 6.0.1）
- ⚠️ Kit App Streaming 一樣需要 RT Core；Crusoe 機型明確不相容
- 按小時計費

**適合**：第一次接觸 Isaac Sim、只想快速看看它長什麼樣。
**不適合**：需要最新版本、或要精細控制環境。

---

## 五、成本控制（這件事比想像中重要）

$1/hr 聽起來不多，但忘記關機一個週末 = $48。

### 建議做法

1. **寫啟停腳本**，別靠記憶
   ```bash
   # ~/bin/isaac-up / isaac-down
   # 用 RunPod CLI 或 API 控制 pod 的 start/stop
   ```
2. **設定閒置自動關機**（多數供應商都有這個選項）
3. **本機先做完能做的**：URDF 建模、MoveIt 設定、程式邏輯全部在 Mac 的 Gazebo 上完成，
   只把「需要 Isaac Sim 高保真物理/渲染」的部分放到雲端
4. **用 persistent volume** 存 USD 資產與訓練 checkpoint，這樣關機不會丟東西
5. **記帳**：每次 session 記錄「時數 × 費率 = 成本」與「這次學到什麼」

### 預算估算

| 階段 | 預估雲端時數 | 成本（@$1/hr） |
|---|---|---|
| A8-1 環境建置 | 8 h | $8 |
| A8-2 USD 與資產匯入 | 6 h | $6 |
| A8-3 ROS 2 Bridge | 6 h | $6 |
| A8-4 合成資料與隨機化 | 6 h | $6 |
| A8-5 Isaac Lab RL 訓練 | 15 h + 訓練時間 | $15–60 |
| A8-6 Sim-to-Real | 8 h | $8 |
| **第八章合計** | **~50–100 h** | **$50–100** |

RL 訓練會是主要成本來源（可能要跑很久）。
建議先用小規模環境驗證 pipeline 正確，再放大訓練。

---

## 六、ROS 2 版本對接

Isaac Sim 6.0 的 ROS 2 bridge 支援：

| 平台 | ROS 2 |
|---|---|
| Ubuntu 24.04 | **Jazzy（官方推薦）** |
| Ubuntu 22.04 | Humble、Jazzy |
| Windows 11 | Humble |

- Kilted 與 Lyrical **尚未官方支援**
- ROS 1 完全不支援
- 官方 workspace：https://github.com/isaac-sim/IsaacSim-ros_workspaces（有 `jazzy_ws`）

> 這也是本專案主線選 Jazzy 的原因之一 —
> 你在 Mac 上的 Jazzy 開發環境可以直接對接雲端 Isaac Sim。

---

## 七、Isaac Lab

| 項目 | 版本 |
|---|---|
| 穩定版 | **2.3.2**（對應 Isaac Sim **6.0.0**） |
| 開發版 | 3.0.0-beta2（develop 分支，有 breaking change） |

⚠️ **官方文件有版本不一致**：安裝頁還寫「建議用 Isaac Sim 5.1.0」，
但 release notes 說 2.3.2 對應 6.0.0。以 release notes 為準。

需求：Ubuntu 22.04 / Windows 11、32 GB RAM、**≥16 GB VRAM**、Python 3.11、驅動 ≥580.65.06。

- 🔗 Quickstart：https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html
- 🔗 安裝：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

---

## 八、參考連結

- [Isaac Sim 系統需求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim 容器安裝](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_container.html)
- [Livestream 客戶端（含 macOS .dmg）](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
- [AWS 部署](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_advanced_cloud_setup_aws.html)
- [Isaac Launchable](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_advanced_cloud_setup_launchable.html)
- [NGC Isaac Sim container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)
- [Isaac Sim ROS 2 安裝](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros.html)
- [Isaac Lab Release Notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html)
- [RunPod 定價](https://www.runpod.io/pricing)
