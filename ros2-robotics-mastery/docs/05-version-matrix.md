# 版本相容矩陣（2026-07-31 查證）

> 這份文件是整個專案最重要的參考。ROS 2 生態的版本分歧比多數軟體生態嚴重，
> 選錯版本會浪費數十小時。**每次要裝新東西前，先回來看這張表。**

---

## 一、本專案的版本決策

| 場景 | 決策 | 理由 |
|---|---|---|
| **主開發線**（Mac Docker / 雲端） | **ROS 2 Jazzy Jalisco** | LTS 到 2029-05；Isaac Sim 6.0 官方推薦；Gazebo Harmonic 配對；MoveIt/Nav2/ros2_control 皆已成熟 |
| **模擬器**（本機） | **Gazebo Harmonic** | Jazzy 的官方配對，LTS 到 2029-05（與 Jazzy 同期結束，很乾淨） |
| **高保真模擬** | **Isaac Sim 6.0.1** + ROS 2 Jazzy bridge | 官方明文：「ROS 2 Jazzy on Ubuntu 24.04 is recommended」 |
| **RL 訓練** | **Isaac Lab 2.3.2** | 對應 Isaac Sim 6.0.0 |
| **Jetson Orin Nano** | **JetPack 6.2.2 + ROS 2 Humble** | ⚠️ 見下方「Jetson 分歧」 |

---

## 二、ROS 2 發行版現況

| 發行版 | 發行 | EOL | LTS | Tier-1 Ubuntu | 本專案評價 |
|---|---|---|---|---|---|
| **Lyrical Luth** | 2026-05-22 | 2031-05 | ✅ 5 年 | 26.04 | 技術最好，但 **Nav2 尚未釋出**、無 JetPack 支援、第三方套件稀少 → 2027 初再回頭看 |
| Kilted Kaiju | 2025-05-23 | **2026-12** | ❌ | 24.04 | 剩不到 5 個月，不要碰 |
| **Jazzy Jalisco** | 2024-05-23 | **2029-05** | ✅ 5 年 | 24.04 | ⭐ **本專案主線** |
| Humble Hawksbill | 2022-05-23 | **2027-05** | ✅ | 22.04 | 只在 Jetson 線使用；剩約 10 個月 |
| Iron Irwini | 2023-05 | 2024-12 | ❌ | — | 已死 |

> ⚠️ 網路上大量 2022-2024 的教學是 Humble 或 Foxy 的。指令大多相容，但 **模擬器部分完全不能照抄**。

---

## 三、Gazebo 配對（這裡最多人踩雷）

| ROS 2 | 官方 Gazebo | Gazebo EOL |
|---|---|---|
| Humble | Fortress (LTS) | 2027-05 |
| **Jazzy** | **Harmonic (LTS)** | **2029-05** |
| Kilted | Ionic | 2026-12 |
| Lyrical | Jetty (LTS) | 2031-05 |

### ⚠️ Gazebo Classic 已於 2025-01 EOL

- `gazebo_ros_pkgs`、`<gazebo>` 標籤舊寫法、`gazebo` 指令 → **在 Jazzy 上根本裝不起來**
- 新的是 `gz sim`、`ros_gz_bridge`、`gz_ros2_control`
- **判斷法**：教學裡出現 `roslaunch gazebo_ros` 或 `gzserver` → 過時，跳過

---

## 四、核心套件版本（來自 rosdistro，2026-07-31）

| 套件 | Humble | **Jazzy** | Kilted | Lyrical |
|---|---|---|---|---|
| MoveIt | 2.5.9 | **2.12.4** | 2.14.3 | 2.14.1 |
| Nav2 | 1.1.20 | **1.3.12** | 1.4.2 | ❌ **未釋出** |
| ros2_control | 2.54.0 | **4.46.0** | 5.16.0 | 6.8.0 |
| ros2_controllers | 2.53.2 | **4.41.0** | 5.16.0 | 6.8.0 |

> 📌 MoveIt 2 官方文件目前**只有 `main`(Rolling) 與 `humble` 兩個路徑**，
> `moveit.picknik.ai/jazzy/` 會 404。用 `main` 版文件，Jazzy 上的 API 幾乎一致。

---

## 五、⚠️ Jetson 分歧（本專案最大的版本陷阱）

**NVIDIA 在 Isaac ROS 4.0 放棄了 Jetson Orin 系列，只支援 Jetson Thor。**

| | Orin Nano（你的硬體） | Jetson Thor |
|---|---|---|
| Isaac ROS | **3.2（最後支援版）** | 4.x |
| JetPack | **6.1 / 6.2 / 6.2.2** | 7.1+ |
| Ubuntu | **22.04** | 24.04 |
| ROS 2 | **Humble** | Jazzy |
| CUDA | 12.6 | 13.0+ |

### 所以會發生什麼

- 你的 **Mac 主線是 Jazzy**，**Jetson 線是 Humble** — 這是無法避免的分歧
- 好消息：ROS 2 的 DDS wire protocol 跨版本大致可通，**但介面（.msg）定義必須完全一致**
- 實務作法：把自訂 interface package 抽出來，兩邊各自編譯同一份原始碼

### JetPack 7.2 呢？

JetPack 7.2（2026-06 發行，Ubuntu 24.04 + Jazzy）看起來很誘人，但：

1. NVIDIA 官方說法**互相矛盾** — 下載頁列出「Jetson Orin Family」，但 2026-03 論壇版主明說「JetPack 7 還不支援 Orin Nano」
2. 即使能刷，**Isaac ROS 4.x 仍然不支援 Orin** → 你會失去硬體加速感知的全部價值
3. **JetPack 7.x 已無 SD 卡映像** — 官方原文：*"Starting with JetPack 7.2, SD Card images are no longer supported."*

**結論：走 JetPack 6.2.2。** 這不是「保守選擇」，是唯一能用 Isaac ROS 的選擇。

### ⚠️ JetPack 6.2.2 的取得方式（很容易卡住）

**6.2.2 沒有自己的 SD 卡映像。** 正確流程是兩段式：

| 步驟 | 動作 | L4T |
|---|---|---|
| ① | 燒錄 **JetPack 6.2.1** 的 `jp62-r1-orin-nano-sd-card-image.zip` | 36.4.4 |
| ② | 開機後 `sudo apt full-upgrade` | **36.5** |

為什麼一定要升到 6.2.2：L4T 36.4.x 有 **IOVA allocator bug**，
任何需要 >~1.1 GB 連續 CUDA buffer 的模型載入會失敗。36.5 已修正。

### ⚠️「Super」與原版是同一塊板子

NVIDIA 工程師 dusty_nv：*"there is not a physical difference in appearance or makeup of
the kit, it is a software/firmware update."*

所以**盒子標示不是可靠訊號**。唯一判斷是讀 UEFI 韌體版本（≥36.0 才能開 JetPack 6.x），
接螢幕開機後連按 Esc 即可，不需要 SD 卡。官方用詞：**"Do not skip this check."**

---

## 六、Isaac Sim / Isaac Lab

| 項目 | 版本 | 關鍵限制 |
|---|---|---|
| Isaac Sim | **6.0.1 GA** | ❌ 不支援 macOS；❌ **A100 / H100 不能用（無 RT Core）**；需 RTX 4080↑ / 16GB VRAM |
| Isaac Lab | **2.3.2** | 對應 Isaac Sim 6.0.0；3.0-beta 在 develop 分支 |
| ROS 2 bridge | — | Ubuntu 24.04 → **Jazzy（推薦）**；Ubuntu 22.04 → Humble/Jazzy；Kilted/Lyrical 未支援 |
| aarch64 建置 | 僅 DGX Spark | **Jetson 完全不能跑 Isaac Sim** |

### 雲端 GPU 選型（必須有 RT Core）

| GPU | RT Core | RunPod 參考價 | 建議 |
|---|---|---|---|
| RTX 4090 24GB | ✅ | ~$0.69/hr | 💰 最便宜可用 |
| RTX 6000 Ada 48GB | ✅ | ~$0.84/hr | 好選擇 |
| L40S 48GB | ✅ | ~$0.99/hr | ⭐ 甜蜜點 |
| RTX PRO 6000 96GB | ✅ | ~$1.99/hr | 頂規 |
| **A100 / H100** | ❌ | — | ❌ **完全不能跑** |

> 💡 A100/H100 是最常見的雲端 GPU，也是最容易誤選的。租之前一定要確認型號。

---

## 七、其他

| 項目 | 版本 | 備註 |
|---|---|---|
| MuJoCo | 3.11.0 (2026-07) | ✅ 原生 Apple Silicon wheel，`pip install mujoco` 直接可用 |
| mujoco_ros2_control | — | 已是官方 ros-controls 套件 |
| Webots | R2025a (2025-01) | ⚠️ 18 個月未更新，維護狀態存疑 |
| PyBullet | 3.2.7 (2025-01) | ❌ 無官方 ROS 2 整合，學 ROS 2 不要用 |
| LeRobot | 0.6.0 (2026-07) | 需 Python ≥3.12；**與 ROS 2 是獨立技術棧**，橋接靠社群 |
| SO-101 手臂 | — | leader+follower 一對 BOM 約 US$230 |

---

## 八、複查清單

每次要引入新東西前問自己：

- [ ] 這個教學是哪一年寫的？用哪個 ROS 2 distro？
- [ ] 它用的是 Gazebo Classic 還是 `gz sim`？
- [ ] 這個套件在 Jazzy 的 rosdistro 裡有釋出嗎？（查 https://index.ros.org）
- [ ] 如果要跑在 Jetson 上，Humble 版本存在嗎？
- [ ] 如果牽涉 GPU，那張卡有 RT Core 嗎？

---

## 資料來源

- [ROS 2 Releases（Vulcanexus 鏡像，docs.ros.org 有反爬蟲）](https://docs.vulcanexus.org/en/latest/ros2_documentation/source/Releases.html)
- [endoflife.date/ros-2](https://endoflife.date/ros-2)
- [REP 2000 — Target Platforms](https://reps.openrobotics.org/rep-2000/)
- [Gazebo ↔ ROS 配對表](https://gazebosim.org/docs/latest/ros_installation/)
- [Gazebo Releases](https://gazebosim.org/docs/latest/releases/)
- [ROS Index（查套件是否釋出）](https://index.ros.org/)
- [Isaac Sim 系統需求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Sim ROS 2 相容性](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros.html)
- [Isaac ROS Releases](https://nvidia-isaac-ros.github.io/releases/index.html)
- [Isaac ROS：Orin 支援討論](https://forums.developer.nvidia.com/t/does-isaac-ros-4-0-0-officially-support-jetson-orin/349564)
- [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)
- [Isaac Lab Release Notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html)
