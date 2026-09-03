# P7 · Pickleball Tracker —— 單鏡頭球體追蹤與比賽分析

> 對應技能樹節點 **P7-1 / P7-2 / P7-3**（約 50 h）。
> 起源：2026-09-02 與 Gemini 的討論（[分享連結](https://share.gemini.google/KddigVvt0owY)）。
> 目標：做一個 **SwingVision 等級功能面** 的 ROS 2 感知＋分析管線，
> 從 YouTube 影片離線跑起，最後在 Jetson Orin Nano 上即時跑。

---

## 0. 一句話定義

```
影片 / Pixel 5 串流 ──► 球偵測 ──► 追蹤 ──► 球場幾何 ──► 落點/In-Out/球速
                                                   │
                                                   ▼
                            回合切段 ──► 擊球分類 ──► AI 計分 ──► 統計 / 熱圖 / RViz
```

**這是感知 + 分析管線，不含控制迴路。** 不做模擬手臂。
但所有輸出都走 `pickleball_msgs`，未來若要接 P2 的手臂，只需訂閱 `BallTrack`。

---

## 1. 已決定的事（不要重開）

| 決策 | 選擇 | 理由 |
|---|---|---|
| 語言 | **全 C++**（rclcpp、OpenCV、TensorRT） | 面試展示點；AI 寫、我審測試與整合驗證。見第 8 節的補償條款 |
| 開發平台 | Mac Docker（**Jazzy**）離線迭代 → Jetson（**Humble**）部署 | Jetson 上迭代太慢；程式碼維持雙版本可編譯 |
| 輸入來源順序 | YouTube 轉播 → Pixel 5 USB tethering 串流 → （選配）Arducam OV9281 | 先不花錢；影片先錄成 rosbag（MCAP）當所有測試的基準 |
| 偵測器 | 先傳統 CV，後 DNN（TrackNet 類或 YOLO 微調）；**共用同一個輸出 message** | 換偵測器本身成為可量化實驗 |
| 回饋機制 | RViz 軌跡視覺化 + 分析數據 | 不做 Gazebo 手臂 |
| 2D → 3D | 球場四角 homography 算地面落點；高度用彈道擬合 | 單鏡頭可在一個週末驗證；stereo 留作支線 |
| 計分 | **只做全自動 AI 計分**（不留手動 service） | 從落點、結果與回合結束推點數；錯了就是資料 |
| 範圍 | 見第 2 節 | SwingVision 的分享／匯出／教練市集／Watch／雲端全部排除 |
| 球員追蹤 | 只做位置與擊球位置，**不做姿態骨架** | Jetson 8 GB 共享記憶體養不起第二個大模型 |

---

## 2. 功能範圍（對照 SwingVision）

SwingVision 功能全貌研究於 2026-09-03（官方站、App Store、ITF 核可報告、評測）。
⚠ = 官方未證實、當 optional 欄位保留。

| SwingVision 類別 | 納入 | 排除 |
|---|---|---|
| 擷取／設定 | 單鏡頭、匯入影片、球場自動偵測與框位驗證 | Swing Stick、Apple Watch、Remote Control、雲端備份 |
| 球追蹤 | 3D 擊球點、3D 落點、觸網、軌跡、球速、In/Out、落在哪側、深度、方向、結果 | spin RPM ⚠、net clearance ⚠（欄位保留） |
| 球員追蹤 | 球員偵測與 id、擊球位置、站位時間分布、跑動距離 | 姿態骨架、心率 |
| 單拍統計 | 擊球種類（serve/return/serve+1/return+1/dink/drop/3rd shot/volley/drive/lob）、Deuce/Ad、進球率、球速、深度分布、winner/UE/FE、回合長度 | 原始資料匯出 XLSX（改為 rosbag 就是原始資料） |
| 比賽統計 | 自動計分、point-by-point 時間軸、發球／接發統計、三種熱圖、對手統計 | 趨勢圖、週目標 |
| 影片輸出 | 按回合切段、overlay | 精華剪輯、分享、自動跟球裁切、4K |
| 即時 | 挑戰線審（最近 N 拍）、語音 "Out!"、即時計分板 | Goal Mode、Watch 震動 |
| 教練 | — | 全部排除 |
| Pickleball 專屬 | dink/drop/3rd shot、發球深度 | kitchen 違例（SwingVision 官方也沒有） |

---

## 3. 系統架構

### 節點（全 C++）

```
┌──────────────┐  Image   ┌───────────────┐ BallDetection ┌──────────────┐ BallTrack
│ camera_node  ├─────────►│ ball_detector ├──────────────►│ ball_tracker ├────────┐
│ file/pixel5  │          │ cv | trt      │               │ Kalman + tf  │        │
└──────────────┘          └───────────────┘               └──────────────┘        │
        │ Image                                                                    ▼
        ▼                                                                  ┌──────────────┐
┌──────────────────┐ tf: court→camera                                      │ shot_engine  │
│ court_calibrator ├──────────────────────────────────────────────────────►│ bounce/shot  │
│ homography       │                                                       └──────┬───────┘
└──────────────────┘                                                              │ ShotEvent
        │ Image                                                                    ▼
        ▼                                                                  ┌──────────────┐
┌──────────────────┐ PlayerState                                           │ rally_engine │
│ player_tracker   ├──────────────────────────────────────────────────────►│ rally/score  │
└──────────────────┘                                                       └──────┬───────┘
                                                                                  │ Rally, MatchState
                                                                                  ▼
                                                              ┌──────────────┐  ┌──────────────┐
                                                              │ stats_node   │  │ viz_node     │
                                                              │ Stats/heatmap│  │ RViz Marker  │
                                                              └──────────────┘  │ overlay/audio│
                                                                                └──────────────┘
```

| 節點 | 職責 | 階段 |
|---|---|---|
| `camera_node` | 影片檔 / Pixel 5 串流（GStreamer）→ `sensor_msgs/Image` + `CameraInfo`；`use_sim_time` 下用影片時間戳 | 0 |
| `ball_detector` | 兩個 plugin：`cv`（背景相減 + 顏色/形狀）與 `trt`（TensorRT）；輸出 `BallDetection` | 0 / 3 |
| `ball_tracker` | Kalman（等加速度模型）、遺失補插、彈道擬合估 z；輸出 `BallTrack` | 0 / 1 |
| `court_calibrator` | 球場四角 → homography；發布 `court` → `camera` 的 static tf 與 `CourtModel` | 1 |
| `shot_engine` | 從 `BallTrack` 抓 bounce（z 最低點 + 速度反向）、hit（加速度突變）、net contact；In/Out、深度、方向；輸出 `BounceEvent`、`ShotEvent` | 1 / 2 |
| `player_tracker` | 人物偵測（DNN）+ 追蹤；擊球時最近的球員 = hitter；輸出 `PlayerState` | 2 |
| `rally_engine` | 回合切段（球靜止 > N 秒 或 出界 → 回合結束）、擊球分類、**AI 計分**（side-out scoring 規則引擎）；輸出 `Rally`、`MatchState` | 2 |
| `stats_node` | 累積 `Stats`、熱圖 bins | 2 |
| `viz_node` | RViz `MarkerArray`（軌跡、落點、球場線）、影像 overlay、`Challenge` service、語音 "Out!" | 1 / 3 |

### 座標系（tf2）

```
court ──static──► camera ──► (image plane)
  │
  └── court 原點：近端底線與中線交點；x 沿底線、y 朝對面、z 向上（右手系）
```

所有 3D 輸出都在 `court` frame。這是 A1-14 tf2 的實戰題。

---

## 4. `pickleball_msgs` 介面

一次定義完，各階段只填入更多欄位。**所有 ⚠ 欄位都有 `_valid` bool 旗標。**

```
msg/
  BallDetection.msg
    std_msgs/Header header          # frame_id = camera
    float32 u, v                    # 像素座標
    float32 radius_px
    float32 confidence
    uint8   source                  # SOURCE_CV=0, SOURCE_DNN=1

  BallTrack.msg
    std_msgs/Header header          # frame_id = court
    uint32  track_id
    geometry_msgs/Point position    # court 座標 (x, y, z)；Phase 0 只有 u/v 投影，z=0
    geometry_msgs/Vector3 velocity
    float32 speed_mps
    bool    is_predicted            # 遺失補插
    uint8   spin_type               # SPIN_UNKNOWN/TOPSPIN/SLICE/FLAT ⚠
    bool    spin_valid

  BounceEvent.msg
    std_msgs/Header header
    uint32  track_id, rally_id
    geometry_msgs/Point position    # court
    uint8   in_out                  # IN / OUT / UNCERTAIN
    float32 in_out_confidence
    uint8   side                    # NEAR / FAR
    uint8   depth_zone              # KITCHEN / MID / DEEP / BEYOND_BASELINE
    bool    is_net_contact

  ShotEvent.msg
    std_msgs/Header header
    uint32  shot_id, rally_id
    uint8   player_id
    uint8   stroke_type             # SERVE / RETURN / SERVE_PLUS_ONE / RETURN_PLUS_ONE /
                                    # THIRD_SHOT_DROP / DINK / DRIVE / VOLLEY / LOB / OVERHEAD / UNKNOWN
    uint8   serve_side              # DEUCE / AD / NA
    uint8   result                  # IN / OUT_LONG / OUT_WIDE / NET / UNKNOWN
    geometry_msgs/Point hit_point   # 3D 擊球點
    BounceEvent bounce
    float32 speed_mps
    uint8   direction               # CROSS_COURT / DOWN_THE_LINE / MIDDLE
    uint8   depth_zone
    float32 net_clearance_m         # ⚠
    bool    net_clearance_valid

  PlayerState.msg
    std_msgs/Header header
    uint8   player_id
    geometry_msgs/Point position    # court
    uint8   court_zone              # BEHIND_BASELINE / BASELINE_TO_KITCHEN / KITCHEN_LINE
    float32 distance_run_m          # 累積 ⚠
    bool    distance_valid

  Rally.msg
    uint32  rally_id
    builtin_interfaces/Time start, end
    ShotEvent[] shots
    uint8   winner_player_id
    uint8   how_ended               # WINNER / UNFORCED_ERROR / FORCED_ERROR / UNKNOWN
    uint8   ending_shot_result

  MatchState.msg
    std_msgs/Header header
    uint8   scoring_mode            # SIDE_OUT / RALLY
    uint8[] score                   # [server_team, receiver_team, server_number]
    uint8   serving_team, serving_player, serving_side
    Rally[] point_history           # 或只帶 rally_id[]，看訊息大小
    float32 score_confidence        # AI 計分的信心

  Stats.msg
    uint8   player_id
    uint32  shots_total, shots_in
    float32 pct_in, avg_speed_mps, max_speed_mps
    uint32[] stroke_histogram       # 依 stroke_type 索引
    uint32[] depth_histogram        # 依 depth_zone 索引
    uint32[] placement_heatmap      # 球場切成 6x11 格的落點計數
    uint32[] position_heatmap       # 擊球位置
    uint32   rallies_won, rallies_lost, winners, unforced_errors
    float32  avg_rally_length, longest_rally

  CourtModel.msg
    geometry_msgs/Point[4] corners_court
    float32[9] homography           # image → court
    float32 net_height_m

srv/
  Challenge.srv                     # in: uint8 last_n → out: BounceEvent[] calls
  Calibrate.srv                     # in: 四角像素座標（或 auto=true） → out: CourtModel, success

action/
  ProcessSession.action             # goal: video_path → feedback: percent, current_rally → result: Stats[]
```

**跨版本注意**：msg 定義不得用 Jazzy 才有的型別，Humble 上要能編。這是 A9-2 的 task。

---

## 5. 分階段實現

| 階段 | 產出 | 前置節點 | 樹上節點 |
|---|---|---|---|
| **Phase 0 · 離線骨幹** | YouTube 影片 → MCAP bag；`camera_node`、`ball_detector(cv)`、`ball_tracker`；RViz 畫 2D 軌跡 | A1-6 launch、A1-12 rosbag、A5-2 cv_bridge | P7-1 |
| **Phase 1 · 球場幾何** | `court_calibrator`、tf；`shot_engine` 出落點、In/Out、深度、方向、球速；RViz 3D 軌跡 | A1-14 tf2 | P7-1 |
| **Phase 2 · 回合與比賽層** | `player_tracker`、`rally_engine`（切段、擊球分類、AI 計分）、`stats_node`、熱圖 | A1-4 action、A1-5 params | P7-2 |
| **Phase 3 · Jetson 即時** | `ball_detector(trt)`、Pixel 5 串流、語音 Out、`Challenge` service、latency 量測、Mac↔Jetson 分散式 | A5-5 DNN、A9-2、A9-4 | P7-3 |

**Phase 2 的 AI 計分規則引擎**：
- 回合結束條件：球出界（`BounceEvent.in_out == OUT`）、雙彈跳、觸網未過、球靜止 > 2 s
- 贏家：最後一個合法擊球者
- 計分：side-out scoring（0-0-2 起）；發球權轉移、發球者換邊、第二發球者邏輯全部由狀態機處理
- `score_confidence` = 該回合所有關鍵 `in_out_confidence` 的乘積；低於門檻的回合在 RViz 標 ⚠

---

## 6. 量化驗收

| 指標 | 目標 | 量測方式 |
|---|---|---|
| 偵測召回率（離線，Mac） | ≥ 90% | 手標 200 幀當 ground truth，bag 重播比對 |
| 落點誤差 | ≤ 15 cm | 對照手標落點 |
| In/Out 準確率 | ≥ 95%（排除 UNCERTAIN） | 手標 |
| 擊球分類準確率 | ≥ 80%（serve/return/dink/drop/drive/volley） | 手標 100 拍 |
| AI 計分 | 一局比賽終局分數正確；point-by-point 錯誤 ≤ 2 分 | 對照影片實際比分 |
| Jetson 即時 | 720p ≥ 30 fps | `ros2 topic hz` |
| End-to-end latency | 相機 stamp → RViz Marker ≤ 100 ms | header stamp 差 |
| cv vs trt 偵測器 | 同一 bag 下召回率、fps、記憶體三欄對照表 | 同一份 bag |

---

## 7. 素材與工具

- **影片**：YouTube pickleball 轉播，挑 **固定機位、看得到四角、鏡頭不切換** 的片段（多為業餘賽事直播）。轉播切鏡的片段每段都要重校正，先避開。
  - **長度**：兩份 bag。開發用 **約 1 分鐘**（5–8 個乾淨回合，含出界、觸網、dink 各至少一次），每天反覆重播；驗收用 **一整局約 8–15 分鐘**（發球到 11 分結束，比分可對照影片），只在階段結束跑。
  - **大小**：1080p60 未壓縮 Image 每分鐘約 22 GB，錄 bag 一律用 `compressed` topic，5 分鐘約 5–10 GB。超過 10 分鐘會拖慢迭代。
  - **Ground truth 抽樣**：200 幀要跨 10 個以上回合抽，不要連續取。
  - **已取得的素材（2026-09-03）**：`~/Downloads/pickle-ball-sample-court-only-clean.mp4`，md5 `66aba633048ce85d569aec2f3240d725`。PPA Invisalign North Carolina Open 雙打轉播，固定廣角機位，1280×720 @ 30 fps，264.9 s / 7938 幀，切鏡已全部剪掉。四角全部入鏡。比分 0-0 → 8-10，**一局未打完**：當開發用 bag 綽綽有餘（截 1 分鐘），驗收用要另找一段打完 11 分的。
    - **右上角轉播比分板 = 免費 ground truth**：AI 計分驗收改用 OCR 比分板逐幀比對，不用人工記分。
    - **偵測 ROI 必須遮掉比分板（約 x 890–1280, y 0–125）與四周贊助看板**，否則背景相減會把它們當移動物體。
    - 30 fps 非 60 fps，快速抽擊會拖影；Phase 0 傳統 CV 召回率可能達不到 90%，這正是 cv vs trt 對照的意義。
    - 檔案不進 git（`.gitignore` 掉 `results/bags/`），只記錄來源與 md5。
- **Ground truth**：CVAT 或 Label Studio 手標；標注檔存 `results/gt/`。
- **Pixel 5 串流**：USB tethering + GStreamer `rtsp`／`udpsrc`；手機端用 IP Webcam 或 Larix。轉成 `sensor_msgs/Image` 在 Jetson 端做。
- **相機（選配）**：Arducam OV9281 全域快門 MIPI 模組，預算 NT$4,000 內。只有 Pixel 5 的 rolling shutter 拖影證實是瓶頸時才買。
- **參考實作**：TrackNet / TrackNetV2（羽球、網球球體熱圖模型）、WASB（球體追蹤 benchmark）、`opencv_apps`。

---

## 8. 全 C++ 的補償條款

我沒寫過 C++，程式由 AI 產生。為了讓「我審測試」不是空話：

1. **每個節點必附 gtest 單元測試**，且測的是行為（給定輸入序列 → 期望輸出），不是覆蓋率。
2. **整合測試以 bag 重播為基準**：`launch_testing` 重播固定 bag，斷言輸出 message 數量與關鍵欄位。
3. **CI 過 `ament_lint`、`clang-tidy`、`ament_cmake_gtest`**，Jazzy 與 Humble 兩個 job 都要綠。
4. **每個 PR 我要能用一句話說出每個測試在驗什麼**。說不出來的測試不合併。
5. 每個 C++ 節點的 README 要有「資料流」與「為什麼這樣切 callback group」兩段，讓我能對照 A1-7 的知識審。

---

## 9. 專案結構

```
projects/P7-pickleball-tracker/
├── README.md              # 本文件：規格、決策、驗收
├── NOTES.md               # 「我以為 X，結果是 Y」
├── results/
│   ├── gt/                # 手標 ground truth
│   ├── bags/              # MCAP（.gitignore，只留 md5 與來源 URL）
│   └── metrics/           # 驗收數據、cv vs trt 對照表
└── docs/
    ├── architecture.md    # 節點圖、tf 樹、latency budget
    └── swingvision-inventory.md  # 功能對照清單

ros2_ws/src/
├── pickleball_msgs/       # 介面（Humble/Jazzy 皆可編）
├── pickleball_perception/ # camera_node, ball_detector, ball_tracker, court_calibrator, player_tracker
├── pickleball_analysis/   # shot_engine, rally_engine, stats_node
├── pickleball_viz/        # viz_node
└── pickleball_bringup/    # launch、params、rviz config
```

---

## 10. 面試展示物

1. **Demo 影片**：即時 overlay（球軌跡、落點、In/Out、比分）+ RViz 3D 視角並排。
2. **技術文件**（格式參照 `gpu-memory-reading-club`）：架構、message 設計理由、tf 設計、AI 計分狀態機、latency budget、cv vs trt 對照。
3. **數據表**：第 6 節全部指標的實測值。

---

## 11. 已知風險

| 風險 | 對策 |
|---|---|
| YouTube 轉播找不到固定機位片段 | 退而求其次用 Pixel 5 自拍一段（原本的 Q4 (a) 選項） |
| 球太小、傳統 CV 召回率上不去 | Phase 0 只求管線通；召回率指標在 Phase 3 換 DNN 後才驗 |
| AI 計分在雙打 side-out 規則下錯誤累積 | `score_confidence` 顯性化；先單打驗證再開雙打 |
| Humble 與 Jazzy 的 rclcpp API 差異 | 只用兩邊都有的 API；CI 雙 job |
| Jetson 8 GB 同時跑球偵測 + 人物偵測 OOM | 人物偵測降頻（5 fps）或只在 Phase 3 的 Mac 端跑 |
| Pixel 5 rolling shutter 拖影 | 先量測；真的不行才買 OV9281 |
