# Platform Lab — Always-on Robot Runtime（A9-1 Capstone 規格書）

> 對應技能樹節點：A2-4 健康監控、A3-4 混沌演練、A4-6 雲管線、A5-3 分散式、**A9-1 Capstone**。
> 目標：在 Jetson Orin Nano 上運營一套 **72 小時不斷線** 的機器人執行時，Mac 當地面站。
> 這就是 Anvil JD 第一條 Focus Area（robot runtime platform）的縮影。

## 系統拓撲

```
┌────────────── Jetson Orin Nano（機器人）──────────────┐
│ systemd                                              │
│  └─ robot-bringup.service（watchdog + restart 策略）  │
│      └─ ros2 launch bringup.launch.py                │
│          ├─ sensor 節點（lifecycle；相機/模擬感測器）    │
│          ├─ can_bridge 節點（vcan 或實體 can0）         │
│          ├─ health_monitor（diagnostic + 頻率 watchdog）│
│          └─ snapshot recorder（rosbag2 環形緩衝）       │
│ node_exporter + 自製 robot_exporter → Prometheus 格式  │
│ uploader.service（事件觸發 snapshot 上雲、斷線續傳）     │
└──────────────────────┬───────────────────────────────┘
                       │ DDS（同網段）+ WebSocket + HTTPS
┌──────────────────────┴───────────────┐
│ Mac（地面站）                          │
│  Foxglove（即時觀測）                   │
│  Prometheus + Grafana（docker compose）│
│  物件儲存（S3 / MinIO）+ metadata 索引   │
└──────────────────────────────────────┘
```

## 分階段里程碑

| 階段 | 內容 | 對應節點 | 驗收 |
|---|---|---|---|
| M1 | bringup：systemd → launch → lifecycle 節點鏈，開機自動全 active | A2-1..A2-3 | 冷開機 20 次，順序正確、零 flaky |
| M2 | 健康層：diagnostic aggregator、topic 頻率 watchdog、三層 watchdog 圖 | A2-4 | 殺任一節點，30 秒內告警＋自動復原 |
| M3 | 觀測層：journald 日誌規範、snapshot 錄製、node_exporter + Grafana | A4-1..A4-4 | 首屏 10 秒回答「現在健康嗎」 |
| M4 | CAN 迴路：can-lab 的模擬馬達接進 ros2 topic（見 ../can-lab/） | A6-1, A6-2 | 指令框→狀態框閉環進 Foxglove |
| M5 | 雲管線：告警觸發 snapshot 上雲、冪等續傳、metadata 索引 | A4-6 | 拔網線再插回，資料全到、無重複 |
| M6 | 混沌演練：10+ 種故障注入，全流程記錄 | A3-4 | 每種故障有「偵測時間×行為×復原」紀錄 |
| M7 | 72 小時運行：期間注入 3 次故障 | A9-1 | 三件套：告警截圖＋日誌時間線＋復原證據 |
| M8 | 敘事：architecture.md（含選型理由）、postmortem 兩版本、demo 腳本 | A9-2 | 非技術讀者能複述根因 |

## 目錄約定

```
platform-lab/
├── README.md            ← 本檔（規格）
├── architecture.md      ← M8 產出：一張圖＋每個元件的選型理由（ADR 式）
├── bringup/             ← launch、lifecycle 節點、systemd unit 檔
├── health/              ← diagnostic、頻率 watchdog、exporter
├── uploader/            ← 事件觸發上傳、冪等協定、metadata 寫入
├── chaos/               ← 故障注入腳本與 drills.md（故障×偵測×行為×修復表）
└── postmortem/          ← 工程版 + 主管版（300 字無術語）
```

## 故障注入清單（M6 起點，至少涵蓋五類）

- **行程**：kill -9 感測節點／hang 住 callback（sleep 注入）／crash loop
- **裝置**：拔 USB 相機／vcan link down／udev 權限破壞
- **資源**：memory 洩漏模擬（MemoryMax 觸發）／塞滿磁碟／CPU 搶佔
- **網路**：tc netem 延遲 200ms＋5% 丟包／斷網 10 分鐘／AP isolation 模擬
- **時間**：chrony 停掉讓時鐘漂移／時區跳變

每一項記錄：注入方式、預期偵測時間、實際偵測時間、系統行為、復原方式、行動項。

## 面試素材對照（做完就有的數字）

- 「72 小時不中斷、期間 3 次故障全自動復原」→ JD: keep robots running 24/7
- 「任一節點死亡 30 秒內告警、90 秒內復原」→ JD: reliability & debugging
- 「snapshot 環形緩衝，任何事故可交出前 30 秒資料」→ JD: observability
- 「同一 image Mac/Jetson 通用，CI 自動產 arm64」→ JD: containers & CI/CD
- 「主管版 postmortem 由非工程背景讀者驗收」→ JD: internal communication
