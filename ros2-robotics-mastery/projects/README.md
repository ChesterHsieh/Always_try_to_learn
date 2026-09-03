# 實作專案

技能樹裡有幾個節點不只是「學會」，而是要**做出一個可展示的東西**。
這些專案的規格與筆記放在這裡，每個專案一個子資料夾。

---

## P1 · ROS 2 通訊實驗台（節點 A1-K）

> 目標：把第一章學的所有東西整合成一個可觀測、可重播、可壓測的系統。
> 這是「master ROS 溝通系統」的畢業考。

### 規格

**系統組成**（至少 5 個節點）

```
sensor_sim ──topic──► processor ──topic──► aggregator ──► monitor
     │                                          ▲
     └──────────── action server ◄──────────────┘
```

- 每個節點都是 **LifecycleNode**，且有 **composable** 版本
- 一個 launch 啟動全部，支援 `use_sim_time:=true/false` 切換
- 全系統可用一個 YAML 設定參數

**刻意埋入的 5 種故障**

| # | 故障 | 症狀 | 診斷線索 |
|---|---|---|---|
| 1 | QoS 不相容 | 訂閱者完全收不到，且不報錯 | `ros2 topic info -v` 兩端 QoS 不同 |
| 2 | Callback 阻塞 | 訊息延遲累積、頻率掉 | executor 是 SingleThreaded + callback 內有 sleep |
| 3 | tf 斷鏈 | lookup 拋 extrapolation / connectivity 錯誤 | `view_frames` 看到兩棵樹 |
| 4 | 時間不同步 | tf extrapolation into the future | `use_sim_time` 沒設對 |
| 5 | 訊息堆積 | 記憶體上升、延遲增加 | 發布頻率 > 處理能力，QoS depth 太大 |

**診斷腳本**

寫一支 `diagnose.py`，能自動偵測並定位上述 5 種故障。
這直接對應 `ai-monitor-system` 的 probe 驅動驗證框架 —— 只是換到機器人領域。

```python
# 概念示意
PROBES = [
    QoSCompatibilityProbe(),      # 掃所有 topic，比對 pub/sub 的 QoS
    CallbackLatencyProbe(),        # 監測 callback 執行時間分佈
    TfTreeConnectivityProbe(),     # 檢查 tf tree 是否單一連通
    ClockConsistencyProbe(),       # 比對各節點的 /clock 與 use_sim_time
    QueueDepthProbe(),             # 監測訊息到達 vs 處理速率
]
```

**回歸測試**

錄一段 bag，寫測試用 bag 重播驗證處理結果一致。

### 驗收

把這個實驗台丟給另一個工程師，他能靠你的診斷腳本自己找出 5 個 bug。

---

## P2 · 我的第一台機器人（節點 A2 ~ A4）

> 目標：一台完整的、可控制的模擬機器人。

- URDF/xacro 參數化描述（改一個參數能生 3/5/7 DOF）
- 正確的慣性與碰撞（在 Gazebo 中站得住、不抖）
- Gazebo Harmonic 中可生成、可控制
- `ros2_control` 掛上，`joint_trajectory_controller` 能驅動
- 感測器（相機 + 深度 + IMU）資料在 RViz 中對齊

### 驗收

一行 `ros2 launch`，機器人出現在模擬中、可被程式驅動、所有感測器正常。

---

## P3 · 視覺驅動抓取（節點 A5 + A7）

> 目標：模擬階段的畢業考。

```
深度相機 ──► 點雲分割 ──► 物件位姿 ──► MoveIt Planning Scene
                                            │
                                            ▼
                                    MTC Pick & Place
                                            │
                                            ▼
                                 joint_trajectory_controller
```

- 抓取目標**由感知決定**，不是寫死座標
- Planning Scene 中有桌面與障礙物
- 處理抓取失敗（規劃失敗、抓空）

### 驗收

相機看到桌上任意位置的方塊 → 手臂自動抓起 → 放到指定位置。全程無人工介入。

---

## P4 · Sim-to-Real Pipeline（節點 A8）

> 目標：訓練出來的策略能跨模擬器存活。

- Isaac Sim 場景 + Replicator domain randomization
- Isaac Lab 環境定義 + PPO 訓練
- **對照組**：手寫規則 baseline（呼應 `heuristic-learning` 專案的做法）
- policy 包成 ROS 2 節點或 ros2_control controller
- **跨模擬器驗證**：同一個 policy 在 Gazebo 中也能完成任務

### 驗收

policy 在 Isaac Sim 與 Gazebo 中都能完成任務，且有 baseline 對照數據。

---

## P5 · 分散式機器人系統（節點 A9）

> 目標：真實部署架構。

- Jetson（Humble）跑感知 + 即時控制
- Mac（Jazzy）跑規劃 + 視覺化
- 跨版本、跨機器的穩定通訊
- 實測延遲與抖動數據
- 網路斷線的降級行為

### 驗收

系統架構圖 + 實測延遲數據 + 斷線降級策略文件。

---

## P6 · 🏆 Capstone（節點 A10-4）

> 完整閉環。

```
Isaac Sim 訓練 → domain randomization → Gazebo 跨模擬器驗證
      → 部署到 ROS 2 → Jetson 感知 + 推論 → 真實手臂執行
```

### 驗收

20 次試驗成功率 >60%，且對每一次失敗能給出根因分析。

### 產出

- 完整技術文件（格式參照 `gpu-memory-reading-club`）
- 可選：讀書會投影片 —— 這個題目很適合分享

---

## P7 · Pickleball Tracker（節點 P7-1 ~ P7-3）

> 目標：單鏡頭 pickleball 球體追蹤 + 比賽分析，功能面對標 SwingVision，全 C++。
> 從 YouTube 影片離線跑起，最後在 Jetson Orin Nano 上即時跑。**面試展示專案。**
> 完整規格：[P7-pickleball-tracker/README.md](P7-pickleball-tracker/README.md)

```
影片 / Pixel 5 ──► ball_detector ──► ball_tracker ──► shot_engine ──► rally_engine ──► stats / RViz
                      cv | trt          Kalman+tf      落點/In-Out      AI 計分
```

| 階段 | 產出 | 樹上節點 |
|---|---|---|
| Phase 0+1 | bag 重播 → 3D 軌跡、落點、In/Out、球速 | P7-1 |
| Phase 2 | 回合切段、擊球分類、全自動 side-out 計分、熱圖 | P7-2 |
| Phase 3 | TensorRT 偵測器、Pixel 5 串流、Jetson 即時、latency 數據 | P7-3 |

### 驗收

召回率 ≥ 90%、落點誤差 ≤ 15 cm、一局終局比分正確、Jetson 720p ≥ 30 fps、end-to-end ≤ 100 ms。
加上 demo 影片與技術文件。

---

## 專案筆記格式建議

每個專案資料夾放：

```
P1-comms-lab/
├── README.md         # 規格、決策、結果
├── NOTES.md          # 過程中的踩雷與領悟（這個最有價值）
├── src/              # 或直接指向 ros2_ws/src 中的 package
└── results/          # 數據、圖表、截圖
```

**NOTES.md 特別重要** —— 記下「我以為 X，結果是 Y」的每一刻。
那些才是真正學到的東西，也是之後寫文章/做分享的素材。
