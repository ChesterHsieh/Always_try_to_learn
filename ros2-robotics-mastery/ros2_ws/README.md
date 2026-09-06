# ros2_ws —— P7 Pickleball Tracker 工作區

Phase 0 骨幹已可編譯、可測試、可端到端執行。

## 快速開始

```bash
# 1. 啟動容器（Rancher Desktop 需先開啟）
cd ros2-robotics-mastery/docker
docker compose up -d

# 2. 編譯
#    注意：docker compose exec 不經過 entrypoint，必須自己 source
docker compose exec ros2 bash -c '
  source /opt/ros/jazzy/setup.bash &&
  cd /ros2_ws && colcon build --symlink-install'

# 3. 測試
docker compose exec ros2 bash -c '
  source /opt/ros/jazzy/setup.bash &&
  cd /ros2_ws && colcon test && colcon test-result --all'

# 4. 跑起來（互動式 shell 會自動 source，不必手動）
docker compose exec ros2 bash
ros2 launch pickleball_bringup phase0_full.launch.py \
    video_path:=/path/to/clip.mp4 loop:=true
```

Mac 上的視覺化建議用 Foxglove（`docs/01-environment-setup.md` 方案 A），
不必開 RViz；launch 的 `rviz:=true` 預設關閉。

## Package

| Package | 內容 | 狀態 |
|---|---|---|
| `pickleball_msgs` | 9 個 msg、2 個 srv、1 個 action | 介面已完整定義 |
| `pickleball_perception` | `camera_node`、`ball_detector`、`ball_tracker` + `ball_filter` 函式庫 | Phase 0 可跑 |
| `pickleball_analysis` | `score_machine`（side-out 計分狀態機） | 純邏輯骨架，Phase 2 長節點 |
| `pickleball_viz` | `viz_node`（RViz MarkerArray） | Phase 0 可跑 |
| `pickleball_bringup` | launch、RViz config | Phase 0 可跑 |

## 資料流

```
camera_node ──Image──► ball_detector ──BallDetection──► ball_tracker ──BallTrack──► viz_node
  影片/串流              MOG2 背景相減                     Kalman 濾波                MarkerArray
  影片時間戳              ROI 遮掉記分板                    遺失補插                    軌跡＋球位
```

實際 topic 名稱（節點都用 `~/` 私有命名空間，由 launch 做 remap）：

```
/camera_node/image_raw  →  /ball_detector/detections  →  /ball_tracker/tracks  →  /viz_node/markers
```

## 為什麼這樣切 callback group

四個節點目前**全部使用預設的 MutuallyExclusive callback group**，這是刻意的：

- `camera_node`：只有一個 timer。`cv::VideoCapture` 不是 thread-safe，
  互斥 group 保證 read → publish 不會重入。
- `ball_detector`：MOG2 的背景模型是**有狀態**的。若改成 Reentrant，
  兩張影格並行更新同一個模型會讓模型損壞。
- `ball_tracker`：訂閱 callback 與 coast timer 都會碰同一個 `BallFilter`。
  用 callback group 表達互斥，比到處加 mutex 更容易推理——
  這是 A1-4 Executor 與 Callback Group 的實例。
- `viz_node`：`trail_` 是共享狀態，同理。

Phase 3 加入 `Challenge` service 時才會需要獨立的 Reentrant group，
避免 service 呼叫擋住影格輸出。

## 測試

```
61 tests, 0 errors, 0 failures
```

- `test_ball_filter`（8 個）：Kalman 濾波的行為測試。每個測試的一句話說明寫在檔案開頭。
- `test_score_machine`（9 個）：side-out 計分規則。開局 0-0-2、只有發球方得分、
  兩位發球員、side out 視角切換、11 分領先 2 分獲勝。

核心演算法（`ball_filter`、`score_machine`）刻意**不依賴 rclcpp**，
可以直接用 gtest 驗證，不需要跑起 ROS 節點。

## 跨版本相容（Humble / Jazzy）

- msg 只用兩邊都有的型別，欄位避開 C/C++ 保留字（不能叫 `auto`）。
- `cv_bridge` 標頭用 `__has_include` 同時支援 `.h`（Humble）與 `.hpp`（Jazzy）。
- C++ 標準固定 17，不用 C++20。

## 尚未完成（Phase 1 起）

`court_calibrator`、`shot_engine`、`player_tracker`、`rally_engine`、`stats_node`
都還沒有節點。Phase 0 的 `BallTrack.position` 目前是**像素乘上固定係數**、`z=0`、
`frame_id` 仍是 `camera`；Phase 1 接上 homography 後才會變成真正的 `court` 座標。
