# 課程資源索引（2026-07-31 逐一驗證）

> 所有連結都實際驗證過。**付費/過時/ROS 1 的資源已明確標記**，不要浪費時間。

---

## 一、核心四本（照這個順序走，涵蓋 80% 的技能樹）

### 1. ROS 2 官方教學（Jazzy）— 一切的基準

- 🔗 https://docs.ros.org/en/jazzy/Tutorials.html
- 免費（CC-BY-4.0）｜圖文＋CLI 動手做｜約 20–30 h｜入門→進階
- **注意**：`docs.ros.org` 有 Anubis 反爬蟲，自動抓取會失敗，但**瀏覽器開完全正常**
- 對應節點：A0-3 ~ A0-5、A1-1 ~ A1-15、A2-1 ~ A2-4

### 2. Articulated Robotics（Josh Newans）— 最好的直觀解釋

- 🔗 https://articulatedrobotics.xyz/tutorials/
- 🔗 https://www.youtube.com/@ArticulatedRobotics
- 完全免費｜影片＋完整圖文對照｜約 25–40 h
- 五大系列：Getting Ready to build a ROS robot（9 部）、Build a Mobile Robot with ROS（13+ 部）、
  Docker for Robotics、Coordinate Transforms、Geometry Tips
- ⚠️ 舊的 `mobile-robot/project-overview` 頁面還寫 Foxy / Ubuntu 20.04 — **新的分類頁與內頁已更新為 Jazzy**，以新頁為準
- **這個系列最大的價值**：它是唯一把「概念 → 模擬 → 真實硬體」完整串起來的免費系列
- 對應節點：A0-1、A1-14（tf2 講得極好）、A2-1、A2-5、A3-4、A4-1

### 3. Murilo's ROS2 Tutorial（曼徹斯特大學）— Python 優先

- 🔗 https://ros2-tutorial.readthedocs.io/en/humble/
- 完全免費（可下 PDF）｜ROS 2 Humble（指令對 Jazzy 幾乎一致）｜10–15 h
- **對 Python 工程師特別合適**：先把 rclpy、package、build 系統講透，最後才進 C++
- package/build 系統的解釋比官方清楚
- 對應節點：A0-3、A1-1 ~ A1-6

### 4. ROS 2 Design Docs — 給有系統背景的人

- 🔗 https://design.ros2.org/
- 免費｜設計文件｜5–10 h（選讀）
- 五大類：Overview、Middleware（DDS/QoS）、Interfaces（IDL）、Security、Uncategorized
  （real-time、clock/time、launch、actions、managed nodes、intra-process）
- **這是你的甜蜜點** — 它解釋「為什麼這樣設計」，不是「怎麼用」
- 建議時機：走完第一章之後回頭讀，體會完全不同
- 對應節點：A1-2、A1-4、A1-6 ~ A1-11、A1-15、A1-K

---

## 二、大學課程（公開教材）

### MOGI-ROS（布達佩斯科技經濟大學）⭐ 唯一完整對齊 Jazzy 的學期課程

- 🔗 https://github.com/MOGI-ROS/Week-1-2-Introduction-to-ROS2
- 🔗 https://github.com/MOGI-ROS （其他週次）
- 完全免費（Apache 2.0）｜**ROS 2 Jazzy + Ubuntu 24.04**｜14 週，每週 3–5 h
- 內容：ROS2 基礎 → 模擬 → 建圖 → 定位 → 導航 → 機械手臂控制 → 期末專案
- GitHub 圖文 ＋ YouTube 影片

### Modern Robotics（Northwestern, Kevin Lynch）⭐ 理論基礎的標準教材

- 🔗 http://hades.mech.northwestern.edu/index.php/Modern_Robotics （教材總站）
- 🔗 https://modernrobotics.northwestern.edu/ （影片講座）
- **教科書 PDF 免費**（4 種版本）＋ 影片講座 ＋ **Python / MATLAB / Mathematica 函式庫** ＋ 習題解答
- 不涉及 ROS — 教的是剛體運動、正/逆運動學、雅可比、動力學、軌跡規劃、抓取
- 對應節點：**A7-1（運動學理論）**，這是技能樹裡唯一的純理論節點
- ⚠️ **Coursera 版本要付費**（Coursera FAQ 明文「you cannot take this course for free」）。
  免費 PDF + 影片 + Python 函式庫的內容實質相同，**直接跳過 Coursera**

### Peter Corke's Robot Academy（QUT）

- 🔗 https://robotacademy.net.au/
- 免費｜200+ 部短片（每部 <10 min）＋ masterclasses｜30–40 h
- 機器人學與機器人視覺，不涉及 ROS
- 每課標示難度，約 20% 只需一般知識

---

## 三、動手做平台

| 資源 | URL | 說明 |
|---|---|---|
| **Docker 官方 ROS 2 指南** | https://docs.docker.com/guides/ros2/ | 免費，4 節，2–3 h。A0-1 的主要參考 |
| **Robotic Sea Bass Docker 指南** | https://roboticseabass.com/2023/07/09/updated-guide-docker-and-ros2/ | 多層 Dockerfile、dev container 設計，2–3 h |
| **The Construct — ROS2 Learning Week** | https://www.theconstruct.ai/ros2-learning-week/ | **免費** 5 天 × 60 min，含瀏覽器內模擬環境 |
| **The Construct 免費層** | https://www.theconstruct.ai/pricing/ | ⚠️ **限制嚴格**：只有 Linux/Python3/C++ for Robotics 三門完整開放，其餘課程**只給第一單元**。付費 €39.97/月 |
| **Husarion ROS 2 教學** | https://husarion.com/tutorials/ros2-tutorials/1-ros2-introduction/ | 免費，Humble，8–12 h。launch 的三種寫法講得很清楚 |
| **Gazebo ROS 2 整合** | https://gazebosim.org/docs/harmonic/ros2_integration/ | ⭐ `ros_gz_bridge` 的權威參考，A3-2 必讀 |

---

## 四、專門主題

| 主題 | 官方文件 | 備註 |
|---|---|---|
| **ros2_control** | https://control.ros.org/jazzy/index.html | A4 全章 |
| **MoveIt 2** | https://moveit.picknik.ai/main/index.html | ⚠️ **沒有 jazzy 版路徑**（`/jazzy/` 會 404）。`main` 追蹤 Rolling，但 API 與 Jazzy 的 2.12.4 幾乎一致 |
| **Nav2** | https://docs.nav2.org/ | ⚠️ 網址沒有 distro 區段，永遠是最新版。30+ 個 tutorial，第六章支線 |
| **Isaac Sim** | https://docs.isaacsim.omniverse.nvidia.com/latest/index.html | 第八章 |
| **Isaac Lab** | https://isaac-sim.github.io/IsaacLab/main/index.html | A8-5 |
| **Isaac ROS** | https://nvidia-isaac-ros.github.io/getting_started/index.html | ⚠️ Orin 要用 3.2 分支 |
| **LeRobot** | https://huggingface.co/docs/lerobot/index | 終章 |
| **MuJoCo Menagerie** | https://github.com/google-deepmind/mujoco_menagerie | A3-5 支線，80+ 模型 |
| **Foxglove** | https://docs.foxglove.dev/docs | A1-13，Mac 使用者的救星 |

---

## 五、線上免費書

### Programming Multiple Robots with ROS 2（OSRF 開放書）

- 🔗 https://osrf.github.io/ros2multirobotbook/
- 🔗 https://osrf.github.io/ros2multirobotbook/print.html （單頁全文版，適合離線）
- 完全免費｜⚠️ 內文以 **Foxy**（已 EOL）為基準，概念有效但指令需自行對應
- 特色：涵蓋 **RMF 多機器人車隊協調**，其他教材沒有的內容

---

## 六、❌ 不要浪費時間的（都被驗證過）

| 資源 | 問題 |
|---|---|
| **ETH Zurich "Programming for Robotics"** | 2026 年版確實改教 ROS 2，但**教材鎖在 Moodle，不再公開**。網路上流傳的公開 PDF 與 GitHub 鏡像**全部是 ROS 1 舊版** |
| **Modern Robotics on Coursera** | Coursera FAQ 明文不可免費修讀。用免費 PDF + 影片即可 |
| **《A Concise Introduction to Robot Programming with ROS2》** | Routledge **付費書**。只有[程式碼](https://github.com/fmrico/book_ros2)免費，且 C++ 佔 77.5% |
| **Robotics Back-End 的影片課程** | Udemy / Manning / Packt **全部付費**。只有 [roboticsbackend.com 的文字教學](https://roboticsbackend.com/category/ros2/)免費 |
| **The Construct OPEN Class 播放清單（217 部）** | ROS 1 與 ROS 2 混雜且未分類，需逐部確認版本，CP 值低 |
| **任何使用 `gazebo_ros_pkgs` / `gzserver` 的教學** | Gazebo Classic 已於 2025-01 EOL，在 Jazzy 上裝不起來 |
| **PyBullet 相關的 ROS 2 教學** | 無官方 ROS 2 整合，唯一的橋接是 ROS 1 時代專案 |

---

## 七、建議的閱讀順序

```
第 1-2 週   Docker 指南 → ROS 2 官方 Beginner CLI + Client Libraries
第 3-6 週   Murilo's Tutorial（Python 深入）+ 官方 Intermediate
            ↳ 中途穿插 ROS 2 Design Docs 的 QoS / Executor / Actions
第 7-8 週   Articulated Robotics 的 Coordinate Transforms（tf2）
第 9-11 週  Articulated Robotics 的 URDF + Gazebo 系列
            ↳ 搭配 Gazebo 官方 ROS 2 Integration
第 12-14 週 ros2_control 官方 + Articulated Robotics 的 ros2_control 系列
第 15-19 週 Modern Robotics Ch.3-6, 9（理論）
第 20-24 週 MoveIt 2 官方教學
第 25 週~   Isaac Sim / Isaac Lab 官方文件
（Jetson 線可以在任何時間點平行進行）
```

---

## 八、驗證方法說明

- `docs.ros.org` 與 `robotacademy.net.au` 有反爬蟲/robots.txt 限制，
  自動抓取會失敗但**瀏覽器完全正常**。已用官方 GitHub 原始碼與鏡像交叉驗證
- 版本資訊來自 [ROS Index](https://index.ros.org/) 與 [ros/rosdistro](https://github.com/ros/rosdistro)（權威來源）
