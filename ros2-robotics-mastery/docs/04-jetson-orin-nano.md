# Jetson Orin Nano 部署指南

> 對應技能樹第九章（A9-1 ~ A9-4）。
> **⚠️ 動手刷機前請完整讀完第一節。** 這是整個專案最容易浪費一整個週末的地方。

---

## 0. 快速決策

### ⚠️ 不要看盒子，要看 UEFI

NVIDIA 工程師 dusty_nv 在官方論壇明確說明：

> *"there is not a physical difference in appearance or makeup of the kit,
> it is a software/firmware update that enables increased performance
> via a new 25W clock/power configuration."*

**「Super」與「原版」是同一塊板子**，差別純粹在 QSPI 韌體。
所以盒子上有沒有印 "Super"、通路怎麼標，**都不是可靠訊號**。

唯一可靠的判斷是讀 UEFI 的韌體版本 —— 而且**這個檢查不需要 SD 卡、不需要作業系統**，
兩分鐘就能做完。官方文件對此的用詞是 **"Do not skip this check."**

### 檢查方式（三選一）

| 情境 | 方法 |
|---|---|
| **有螢幕**（最簡單） | 接 HDMI/DP + 鍵盤 → 開機 → NVIDIA 開機畫面出現後**連續狂按 Esc** → 進 UEFI setup menu → 最上方那行就是韌體版本 |
| 無螢幕 | USB-to-TTL 序列線接 Button Header，序列終端裡連按 Esc |
| 已能開機進系統 | `sudo nvbootctrl dump-slots-info` |

### 判讀

```
韌體版本 ≥ 36.0  ──► ✅ 可直接刷 JetPack 6.x（見第 3 節）
韌體版本 <  36.0  ──► ⚠️ 必須先做韌體橋接更新（見第 2 節）
                       否則 JetPack 6 的 SD 卡插進去會「畫面全黑、開不了機」
```

> 💡 **2026 年購買的機器幾乎確定 ≥ 36.0。**
> 2024 年 12 月「Super」發表後，所有出貨的 Devkit 都以 $249 販售且出廠即為 Super-ready。
> 但因為通路可能有舊庫存，而檢查只要兩分鐘，**還是查一下再動手**。

**版本決策：JetPack 6.2.2（Ubuntu 22.04 + ROS 2 Humble）**

不是因為保守，而是因為 **Isaac ROS 4.x 已經放棄 Orin 系列，只支援 Jetson Thor**。
Orin Nano 想用 Isaac ROS，只能停在 3.2 版，而 3.2 要求 JetPack 6.x。
詳見 [05-version-matrix.md](05-version-matrix.md#五jetson-分歧本專案最大的版本陷阱)。

---

## 1. 硬體規格與現實預期

| 項目 | 規格 |
|---|---|
| 算力 | 67 INT8 TOPS（MAXN SUPER 模式）／約 40 TOPS（一般模式） |
| 記憶體 | **8 GB LPDDR5，CPU 與 GPU 共享**，102 GB/s |
| CPU | 6-core Arm Cortex-A78AE |
| GPU | Ampere，1024 CUDA cores + 32 tensor cores |
| 功耗 | 7–25 W |

### 8 GB 共享記憶體的現實

扣掉 OS 與 ROS 2 開銷，實際可用的模型預算大約 **5–6 GB**。

**✅ 跑得動**
- YOLOv8 / v11、DetectNet 物件偵測
- 影像分割、立體深度
- Visual SLAM、nvblox（低解析度）
- 標準 ROS 2 導航堆疊
- 小型 LLM / VLM（≤1.2B 參數較舒服）

**❌ 跑不動**
- **Isaac Sim**（aarch64 建置只給 DGX Spark，Jetson 完全不支援）
- 大型 VLM、GR00T 等基礎模型
- 高解析度 nvblox 重建

參考實測（25W，Orin Nano Super 8GB）：

| 模型 | tok/s |
|---|---|
| SmolLM2-135M | 165 |
| Qwen2.5-0.5B | 93 |
| LFM2.5-1.2B | 54 |
| Qwen3-0.6B / Llama3.2-1B | 40–49 |

7B Q4 是絕對上限，會很吃緊且慢。

---

## 2. ⚠️ 韌體橋接更新（僅韌體 < 36.0 才需要）

官方原文：*"Some developer kits shipped with factory firmware that cannot boot JetPack 6.x."*

**先做第 0 節的 UEFI 檢查。韌體 ≥ 36.0 就跳過本節，直接到第 3 節。**

### 更新流程（順序不可跳）

1. 用 **JetPack 5.1.3** 的 SD 卡映像開機（橋接版本）
2. 系統會排程 bootloader 更新 —— GUI 通知或命令列都會提示
3. 重開機，觀察韌體更新進度跑完
4. 安裝 QSPI updater：
   ```bash
   sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater
   ```
5. 重開機，讓 QSPI 更新完成
6. **關機**，換上目標 JetPack 的 SD 卡
7. 開機、完成首次設定
8. 若還有排程的韌體更新，再重開機一次讓它跑完
9. 這時 `nvpmodel` 才會出現 **MAXN SUPER** 選項

> ### 🔴 韌體更新過程中絕對不能斷電
> 也不要反覆用不相容的 JetPack 映像開機，會弄壞系統狀態。

官方文件：[JetPack 6.x Update Path](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)

---

## 3. 刷 JetPack

### ⚠️ JetPack 6.2.2 沒有自己的 SD 卡映像

這是很容易卡住的地方。正確流程是**兩段式**：

```
① 燒錄 JetPack 6.2.1 的 SD 卡映像   (L4T 36.4.4)
       ↓ 開機、完成首次設定
② sudo apt update && sudo apt full-upgrade  → JetPack 6.2.2 (L4T 36.5)
```

下載檔名：**`jp62-r1-orin-nano-sd-card-image.zip`**
（[JetPack 6.2.1 頁面](https://developer.nvidia.com/embedded/jetpack-sdk-621)）

> 為什麼要升到 6.2.2？因為 L4T 36.4.x 有一個 **IOVA allocator 的 bug**：
> 任何需要 >~1.1 GB 連續 CUDA buffer 的模型載入會失敗。
> **6.2.2（L4T 36.5）已修正**。

> ⚠️ **不要下載 JetPack 7.x** —— 官方明文 *"Starting with JetPack 7.2, SD Card images
> are no longer supported."*，而且 Isaac ROS 4.x 不支援 Orin，你會失去硬體加速感知。

### SD 卡準備（macOS）

1. Disk Utility → 選擇**最上層的實體磁碟**（不是底下的 volume）
2. Erase → Format: **MS-DOS (FAT)**、Scheme: **Master Boot Record**
   - 只選 volume 的話不會出現 Scheme 欄位 → 表示選錯層級了
   - 其實這步只是把卡洗乾淨，Etcher 燒錄時會整張覆蓋
3. 用 **Balena Etcher** 燒錄 `jp62-r1-orin-nano-sd-card-image.zip`

- 卡片規格：64 GB 以上、UHS-I、A2 等級
- 之後跑 Isaac ROS 需要 **128 GB+ NVMe SSD**，SD 卡撐不住（可稍後再補）

### 首次開機 → 升到 6.2.2

```bash
# ① 升級到 JetPack 6.2.2
sudo apt update && sudo apt full-upgrade -y
sudo reboot

# ② 安裝 jtop 監控工具
sudo apt install -y python3-pip
sudo pip3 install -U jetson-stats
sudo reboot

# ③ 驗證
jtop                              # 應顯示 JetPack 6.2.2
cat /etc/nv_tegra_release         # 應顯示 R36 REVISION: 5.x
sudo nvbootctrl dump-slots-info   # 韌體版本
```

### 開啟 MAXN SUPER

```bash
sudo nvpmodel -q            # 查目前模式
sudo nvpmodel -m 2          # 切到 MAXN SUPER（編號依 JetPack 版本，用 -q 確認）
sudo jetson_clocks          # 鎖定最高頻率
```

### 切換到 NVMe SSD（建議）

Isaac ROS 的容器與模型檔案很大，SD 卡的 IOPS 會成為瓶頸。
裝上 NVMe 後把 Docker 的 data-root 指到 SSD：

```bash
sudo systemctl stop docker
sudo mv /var/lib/docker /mnt/nvme/docker
echo '{"data-root": "/mnt/nvme/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl start docker
```

---

## 4. ROS 2 Humble 安裝

JetPack 6.2.2 是 Ubuntu 22.04 → Humble 有官方 aarch64 二進位套件。

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
     | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-ros-base ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

> 在 Jetson 上裝 `ros-humble-ros-base` 而非 `desktop` — 你不需要在 Jetson 上跑 RViz，
> 視覺化交給 Mac 端的 Foxglove。

---

## 5. 跨版本通訊：Jazzy (Mac) ↔ Humble (Jetson)

這是本專案架構上唯一的妥協點。

### 原則

- ROS 2 的 DDS wire protocol 跨 distro **大致相容**，但沒有官方保證
- **介面定義（.msg / .srv / .action）必須完全一致** — 同一份原始碼，兩邊各自編譯
- 標準訊息（sensor_msgs、geometry_msgs 等）在 Humble 與 Jazzy 之間定義相同

### 實務作法

```
ros2_ws/src/
├── my_robot_interfaces/     ← 這個 package 兩邊都要編譯（同一份 source）
├── my_robot_perception/     ← 只在 Jetson 上編譯
└── my_robot_planning/       ← 只在 Mac 上編譯
```

### 連線設定

```bash
# 兩邊都要
export ROS_DOMAIN_ID=42

# 若跨網段或 multicast 不通，改用 Discovery Server
export ROS_DISCOVERY_SERVER=192.168.1.50:11811
```

### 一定要實測，不要假設

```bash
# Jetson
ros2 run demo_nodes_cpp talker
# Mac
ros2 topic echo /chatter
```

若不通，依序檢查：Docker 網路模式 → 防火牆 → ROS_DOMAIN_ID → RMW 實作是否一致。

---

## 6. Isaac ROS 3.2

> ⚠️ **一定要用 3.2 分支，不是 4.x。** 4.x 只支援 Jetson Thor。

需求：JetPack 6.1 / 6.2 / 6.2.2、Ubuntu 22.04、CUDA 12.6、ROS 2 Humble、128 GB+ NVMe。

Orin Nano 上實用的 GEM：

| GEM | 用途 |
|---|---|
| `isaac_ros_nitros` | GPU 零拷貝傳輸（對應技能樹 A1-11 的 intra-process 概念） |
| `isaac_ros_visual_slam` | 硬體加速視覺 SLAM |
| `isaac_ros_yolov8` | 加速物件偵測 |
| `isaac_ros_image_proc` | 影像前處理 |
| `isaac_ros_apriltag` | AprilTag 偵測（手眼標定很好用） |
| `isaac_ros_dnn_stereo_depth` | 立體深度 |
| `isaac_ros_nvblox` | 3D 重建（低解析度） |
| `isaac_ros_mcap_lerobot_converter` | rosbag ↔ LeRobot dataset 轉換 |

### 已知地雷

- **JetPack R36.4.7 的 IOVA allocator 有 bug**：任何需要 >~1.1 GB 連續 CUDA buffer 的
  GGUF 模型載入會失敗。**JetPack 6.2.2 (L4T 36.5) 已修**。如果遇到莫名的記憶體配置失敗，
  先確認你不是在 36.4.7 上。
- 8 GB 共享記憶體下，同時跑多個 GEM 很容易 OOM。**建議實測記錄每個 GEM 的記憶體佔用**。

文件：[Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html)

---

## 7. 建議的系統架構

```
┌─────────────────────────┐        ┌──────────────────────────┐
│  Mac (Docker, Jazzy)    │        │  Jetson Orin Nano        │
│                         │        │  (JetPack 6.2.2, Humble) │
│  • MoveIt 2 規劃         │◄──DDS──►│  • 相機驅動               │
│  • RViz / Foxglove      │        │  • Isaac ROS 感知 GEM     │
│  • rosbag 錄製與分析     │        │  • ros2_control 即時迴路  │
│  • 開發與除錯            │        │  • policy 推論            │
└─────────────────────────┘        └──────────────────────────┘
           │                                    │
           │  (訓練階段)                          │  (執行階段)
           ▼                                    ▼
┌─────────────────────────┐        ┌──────────────────────────┐
│  雲端 GPU (L40S/4090)    │        │  真實手臂 (SO-101 等)      │
│  • Isaac Sim 6.0.1      │        │                          │
│  • Isaac Lab RL 訓練     │        │                          │
└─────────────────────────┘        └──────────────────────────┘
```

**分割原則**：延遲敏感的（控制迴路、感知）放 Jetson；
運算重但可容忍延遲的（規劃、訓練、視覺化）放 Mac 或雲端。

**別忘了斷線降級**：網路斷掉時 Jetson 端要能自主安全停止，不能等 Mac 的指令。

---

## 參考連結

- [JetPack 6.2.1（⭐ SD 卡映像從這裡下載）](https://developer.nvidia.com/embedded/jetpack-sdk-621)
- [JetPack 下載總頁](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Orin Nano Devkit 快速入門（含 UEFI 韌體檢查）](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)
- [JetPack 6.x Update Path（韌體橋接流程）](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)
- [NVIDIA 論壇：Original vs Super 是同一塊板子](https://forums.developer.nvidia.com/t/jetson-orin-nano-development-kit-original-vs-super/325972)
- [Orin Nano Super Devkit 產品頁](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html)
- [Isaac ROS Compute Setup](https://nvidia-isaac-ros.github.io/getting_started/compute/index.html)
- [Isaac ROS 套件總覽](https://nvidia-isaac-ros.github.io/repositories_and_packages/index.html)
- [ROS 2 Humble 安裝](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Fast DDS Discovery Server 教學](https://docs.ros.org/en/jazzy/Tutorials/Advanced/Discovery-Server/Discovery-Server.html)
