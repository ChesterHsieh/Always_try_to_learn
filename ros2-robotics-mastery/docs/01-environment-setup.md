# 環境建置：Mac (Apple Silicon) 上的 ROS 2 Jazzy

> 對應技能樹 A0-1。目標：**30 秒內能回到可寫 code 的狀態**。

---

## 為什麼用 Docker，不直接裝在 macOS 上

ROS 2 的平台分級（REP 2000）：

| 平台 | 等級 | 意義 |
|---|---|---|
| Ubuntu 24.04 / 26.04 **arm64** | **Tier 1** | 官方二進位套件、完整 CI |
| Windows 11 | Tier 1 | 同上 |
| **macOS** | **Tier 3** | ❌ 無二進位套件，只能自己編譯，best-effort |

Apple Silicon 是 arm64，而 Ubuntu arm64 是 **Tier 1**。
所以在 M 系列 Mac 上跑 `linux/arm64` 容器是**原生速度、官方完整支援**——
比在 macOS 上硬編譯好太多。

---

## 一、前置

```bash
# 安裝 Docker Desktop 後驗證 arm64 原生執行
docker run --rm --platform linux/arm64 ubuntu:24.04 uname -m
# 應該印出 aarch64（不是 x86_64，也不該有 QEMU 模擬警告）
```

Docker Desktop 設定建議：
- Memory：8 GB 以上（Gazebo 很吃）
- CPUs：盡量給
- 啟用 VirtioFS（檔案掛載效能好很多）

---

## 二、目錄結構

```
ros2-robotics-mastery/
├── docker/
│   ├── Dockerfile
│   ├── compose.yaml
│   └── entrypoint.sh
└── ros2_ws/
    └── src/          ← 你的 package 放這，掛載進容器
```

`ros2_ws` 掛載進容器，所以你用 Mac 上的編輯器改 code，容器裡立刻看得到。
`build/`、`install/`、`log/` 建議放在容器內的 volume（避免 macOS 檔案系統拖慢 colcon）。

---

## 三、啟動

```bash
cd docker
docker compose build
docker compose up -d
docker compose exec ros2 bash
```

進去之後：

```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 run demo_nodes_cpp talker
```

另開一個終端：

```bash
docker compose exec ros2 bash
source /ros2_ws/install/setup.bash
ros2 run demo_nodes_cpp listener
```

看到訊息流動 → A0-1 的 DoD 達成。

---

## 四、GUI 的三種方案

RViz2 與 Gazebo 需要圖形介面。Mac 上有三條路：

### 方案 A：Foxglove（⭐ 推薦，最省事）

Foxglove Studio 是原生 macOS app / 瀏覽器應用，透過 WebSocket 連進容器，
**完全不需要 X11**。可視化 3D、影像、圖表、tf tree，功能比 RViz 還全面。

```bash
# 容器內
sudo apt install -y ros-jazzy-foxglove-bridge
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Mac 上開 Foxglove → Open connection → `ws://localhost:8765`

> 缺點：MoveIt 的 MotionPlanning panel、Nav2 的 goal 設定等 RViz 專屬 plugin 沒有對應。
> 到第七章（MoveIt）時會需要真正的 RViz。

### 方案 B：X11 轉發（XQuartz）

```bash
brew install --cask xquartz
# XQuartz Preferences → Security → 勾選 "Allow connections from network clients"
# 重登入後
xhost + 127.0.0.1
```

compose.yaml 中設定 `DISPLAY=host.docker.internal:0`。

> 缺點：Gazebo 的 3D 渲染在 X11 轉發下會很慢，且偶爾崩潰。

### 方案 C：容器內 VNC（⭐ 跑 Gazebo/RViz 建議用這個）

在容器內跑 Xvfb + x11vnc + noVNC，從瀏覽器連 `http://localhost:6080`。
渲染在容器內完成，只傳畫面，比 X11 轉發穩定且快很多。

`docker/Dockerfile` 已包含這個選項（build arg `WITH_VNC=1`）。

---

## 五、常見問題

**Q: `colcon build` 很慢**
A: 把 `build/` `install/` `log/` 放進 Docker named volume，不要放在掛載的 macOS 目錄上。
compose.yaml 已經這樣設定。

**Q: 改了 Python 檔還要重 build？**
A: 用 `--symlink-install` 就不用。但改 `setup.py`、`package.xml`、加新檔案時還是要。

**Q: `source` 之後還是找不到我的 package**
A: 檢查 `setup.py` 的 `data_files` 有沒有把 launch/config 檔裝進去。
這是 ament_python 最常見的坑。

**Q: Gazebo 開不起來，說找不到 GPU**
A: 容器內沒有 GPU 加速，Gazebo 會走軟體渲染（llvmpipe）。
簡單場景可以接受；複雜場景建議降低 `--render-engine ogre` 品質或改用雲端。

**Q: 兩個終端一個看得到 topic 一個看不到**
A: 檢查兩邊的 `ROS_DOMAIN_ID` 是否一致，以及有沒有都 `source install/setup.bash`。

---

## 六、驗收（A0-1 的 DoD）

```bash
docker compose down
docker compose up -d
docker compose exec ros2 bash -c "source /ros2_ws/install/setup.bash && ros2 pkg list | wc -l"
```

30 秒內完成，且 `ros2_ws/src` 的內容還在 → 通過。
