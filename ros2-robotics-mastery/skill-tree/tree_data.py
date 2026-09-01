# -*- coding: utf-8 -*-
"""ROS 2 Robot Platform Engineer — 技能樹單一資料來源 (v2)

v2 (2026-09-01)：對齊 Robot Platform Software Engineer JD 重構主線——
執行時平台、可靠性除錯、可觀測性、CAN/硬體介面、C++ 生產力、Jetson 部署。
原 manipulation / Isaac 路線濃縮為支線。題庫（Q）分檔在 quiz_*.py。

修改後執行:  python3 build.py
"""

from quiz_bank import Q  # noqa: F401

META = {
    "title": "ROS 2 Robot Platform Engineer",
    "subtitle": "通訊核心 → 執行時平台 → 可靠性 → 可觀測性 → Jetson 真機",
    "owner": "Chester Hsieh",
    "slug": "ros2-platform",
    "storage_key": "ros2-platform-skilltree-v2",
    "version": "2.0.0",
    "generated": "2026-09-01",
    "distro": "ROS 2 Jazzy Jalisco (LTS → 2029-05)；Jetson 上以容器跑",
    "weekly_hours": 15,
    "xp_per_level": 120,
    "quiz_gate": True,
    "dig_intro": "我正在準備 Robot Platform Engineer 的能力（ROS 2 / Linux 可靠性 / 硬體介面），",
    "dig_level": "資料工程與雲端 infra 底子強，但 ROS 2 與 C++/硬體側還在補",
    "titles": [[0, "見習生"], [4, "節點工匠"], [8, "系統整合者"], [12, "平台工程師"], [16, "值班救火隊長"], [20, "Platform 大師"]],
    "acts": [
        {"id": "A0", "name": "序章 · 立足點",         "en": "Foothold",       "color": "#89a9d6"},
        {"id": "A1", "name": "第一章 · 通訊核心",      "en": "Comms Core",     "color": "#4fc3f7"},
        {"id": "A2", "name": "第二章 · 執行時平台",    "en": "Runtime",        "color": "#ffa726"},
        {"id": "A3", "name": "第三章 · 可靠性除錯",    "en": "Reliability",    "color": "#ef5350"},
        {"id": "A4", "name": "第四章 · 可觀測性與交付", "en": "Observability",  "color": "#9ccc65"},
        {"id": "A5", "name": "第五章 · Jetson 部署",   "en": "Edge Deploy",    "color": "#26c6da"},
        {"id": "A6", "name": "第六章 · 硬體介面",      "en": "HW Interfaces",  "color": "#ffca28"},
        {"id": "A7", "name": "第七章 · C++ 生產力",    "en": "Production C++", "color": "#ab47bc"},
        {"id": "A8", "name": "支線 · 模擬與手臂鳥瞰",  "en": "Sim Detour",     "color": "#66bb6a"},
        {"id": "A9", "name": "終章 · Platform Capstone", "en": "Capstone",     "color": "#ff7043"},
    ],
}

N = [

# ─────────────────────────── A0 · 序章 ───────────────────────────
{
 "id": "A0-1", "act": "A0", "type": "start", "track": "main",
 "title": "工作站落地", "en": "Dev Environment",
 "x": 1200, "y": 110, "hours": 3, "deps": [],
 "desc": "在 Mac (Apple Silicon) 用 Docker 跑 linux/arm64 的 ROS 2 Jazzy。Ubuntu arm64 是 ROS 2 Tier-1 平台，M 系列 Mac 跑容器是原生速度；直接在 macOS 裝 ROS 2 是 Tier-3，不要走。你的 Docker 底子已經有了，這節點只補 ROS 特有的坑：GUI、網路模式、workspace 掛載。",
 "why": "環境一旦穩定，後面 40+ 個節點都在同一個容器裡跑，不會有『我的環境壞了』這種學習斷點。",
 "tasks": [
   "用 repo 的 docker/compose.yaml 起容器，確認 `uname -m` 回傳 aarch64",
   "容器內跑 `ros2 run demo_nodes_cpp talker` 與 `listener`，看到訊息流動",
   "確認 ros2_ws 掛載進容器：本機改檔、容器內立刻看得到",
   "搞懂為什麼 DDS 在 Docker bridge 網路下跨容器發現會失靈（multicast），以及 network_mode: host 在 Mac 上的限制",
 ],
 "dod": "關掉容器再開一次，30 秒內回到可寫 code 的狀態，且能說出本機↔容器↔跨容器三種情境下 DDS 發現各會發生什麼事。",
 "res": [
   {"t": "Docker 官方 ROS 2 指南", "u": "https://docs.docker.com/guides/ros2/", "k": "hands-on", "h": 1},
   {"t": "ROS 2 官方：VSCode + Docker 開發環境指南", "u": "https://docs.ros.org/en/jazzy/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html", "k": "read", "h": 2},
 ],
},
{
 "id": "A0-2", "act": "A0", "type": "normal", "track": "main",
 "title": "colcon 與 workspace", "en": "Workspace & Build",
 "x": 1040, "y": 235, "hours": 4, "deps": ["A0-1"],
 "desc": "ROS 2 的 build 系統：ament_python / ament_cmake、package.xml 依賴宣告、colcon 的 symlink-install、rosdep 自動裝依賴、underlay/overlay 的 source 順序。",
 "why": "package 同時是 build 單位、依賴單位、與執行期的資源查找單位——這跟 pip/npm 的心智模型都不同，混著理解會在每個環節被咬一口。",
 "tasks": [
   "`ros2 pkg create --build-type ament_python my_first_pkg` 並讀懂產生的每一個檔案",
   "用 `colcon build --symlink-install` 並驗證改 Python 檔不用重 build",
   "印出 `source install/setup.bash` 前後的 AMENT_PREFIX_PATH 差異，解釋 underlay vs overlay",
   "用 rosdep 安裝一個外部依賴，說出它從 package.xml 到 apt 套件名的解析路徑",
 ],
 "dod": "能從空資料夾建出一個可 `ros2 run` 的 package，並解釋為什麼 build 完不重新 source 會跑到舊版本。",
 "res": [
   {"t": "ROS 2 官方：Creating a workspace", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html", "k": "hands-on", "h": 2},
   {"t": "Murilo's ROS2 Tutorial（Python 優先）", "u": "https://ros2-tutorial.readthedocs.io/en/humble/", "k": "read", "h": 2},
 ],
},
{
 "id": "A0-3", "act": "A0", "type": "normal", "track": "main",
 "title": "CLI 內視鏡", "en": "ros2 CLI",
 "x": 1360, "y": 235, "hours": 3, "deps": ["A0-1"],
 "desc": "ros2 node / topic / service / action / param / interface / doctor / bag。這是你以後除錯的全部武器，等同 kubectl 之於 K8s。",
 "why": "ROS 2 是分散式系統，CLI 是你唯一能『看見』系統狀態的方式。JD 說 root-cause across layers——第一層就是這裡。",
 "tasks": [
   "`ros2 topic echo / hz / bw / info -v`（-v 顯示 QoS，之後很重要）",
   "`ros2 node info` 看一個節點的完整拓撲（訂什麼、發什麼、提供什麼 service）",
   "`ros2 doctor --report` 跑一次，讀懂每一段在檢查什麼",
   "`ros2 interface show` 查 msg 定義，不開瀏覽器完成一次型別查找",
 ],
 "dod": "給你一個陌生的執行中系統，10 分鐘內畫出節點-topic 拓撲圖，且標出每個 topic 的發布頻率。",
 "res": [
   {"t": "ROS 2 官方：Beginner CLI Tools 全系列", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools.html", "k": "hands-on", "h": 3},
 ],
},
{
 "id": "A0-4", "act": "A0", "type": "notable", "track": "main",
 "title": "turtlesim 首殺", "en": "First Blood",
 "x": 1200, "y": 340, "hours": 2, "deps": ["A0-2", "A0-3"],
 "desc": "用 turtlesim 把 node/topic/service/param 全部走一遍，並用 CLI 即時觀察。第一次把抽象概念接到會動的東西上。",
 "why": "接下來每一章的心智模型都建立在『我親手看過它動』之上。跳過這 2 小時，後面每個概念都要多花一倍時間想像。",
 "tasks": [
   "跑 turtlesim + teleop，畫個圈",
   "`ros2 service call` 產生第二隻烏龜",
   "`ros2 param set` 改背景色，並用 `ros2 param dump` 匯出",
   "錄一段 rosbag 再重播，觀察烏龜重演軌跡",
 ],
 "dod": "不看教學，只靠 CLI 補全指令，5 分鐘內完成生龜、控龜、錄放三件事。",
 "res": [
   {"t": "ROS 2 官方：turtlesim 教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Introducing-Turtlesim/Introducing-Turtlesim.html", "k": "hands-on", "h": 2},
 ],
},

# ─────────────────────────── A1 · 通訊核心 ───────────────────────────
{
 "id": "A1-1", "act": "A1", "type": "notable", "track": "main",
 "title": "Topics 與自訂介面", "en": "Topics & Interfaces",
 "x": 1200, "y": 470, "hours": 6, "deps": ["A0-4"],
 "desc": "rclpy 寫 publisher/subscriber，定義自訂 msg，理解 IDL → 生成程式碼的流程，以及 topic 名稱空間與 remap。",
 "why": "topic 是 ROS 2 的血管。之後的 QoS、rosbag、觀測管線全部長在它上面。",
 "tasks": [
   "寫一對 pub/sub 節點（Python），發自訂 msg",
   "自訂 msg 放在獨立的 interface package（業界慣例），跨 package 引用",
   "用 remap 把同一個節點跑兩份而不互撞",
   "說出 msg 欄位改名後，哪些下游會壞、怎麼發現",
 ],
 "dod": "從零建出含自訂 msg 的雙節點系統，並能解釋 interface package 為什麼要獨立。",
 "res": [
   {"t": "ROS 2 官方：Writing a simple publisher and subscriber (Python)", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html", "k": "hands-on", "h": 2},
   {"t": "ROS 2 官方：Creating custom msg and srv files", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A1-2", "act": "A1", "type": "normal", "track": "main",
 "title": "Service 與 Action", "en": "Service & Action",
 "x": 1010, "y": 590, "hours": 5, "deps": ["A1-1"],
 "desc": "request/response（service）與長任務（action：goal / feedback / result / cancel）。什麼時候用 topic、service、action——這是 API 設計題，不是語法題。",
 "why": "JD 要你給 sensors/actuators 提供 clean, stable APIs——選錯通訊原語的 API 天生就不 clean。",
 "tasks": [
   "寫一個 service server/client",
   "寫一個 action server，回報 feedback、支援 cancel",
   "整理一張決策表：什麼情境用 topic / service / action，各給一個真實例子",
   "示範 service 卡死 client 的情況，說明為什麼呼叫端要用 async + timeout",
 ],
 "dod": "能對任一機器人功能（開燈、移動到座標、韌體更新）說出該用哪種原語並講出理由。",
 "res": [
   {"t": "ROS 2 官方：Writing a simple service and client (Python)", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html", "k": "hands-on", "h": 2},
   {"t": "ROS 2 官方：Writing an action server and client (Python)", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A1-3", "act": "A1", "type": "normal", "track": "main",
 "title": "Parameters 與 Launch", "en": "Params & Launch",
 "x": 1390, "y": 590, "hours": 5, "deps": ["A1-1"],
 "desc": "宣告式參數、YAML 參數檔、動態改參數與 callback；launch 檔（Python DSL）組合多節點、傳參數、命名空間、條件啟動。",
 "why": "真實機器人是一個 launch 檔拉起 30 個節點的系統。launch 就是機器人版的 docker-compose——你會秒懂，但細節不同。",
 "tasks": [
   "節點宣告參數＋型別＋描述，用 YAML 檔載入",
   "註冊 on-set 參數 callback，拒絕非法值",
   "寫 launch 檔啟動 3 個節點，其中一個帶 namespace 與 remap",
   "用 launch argument 讓同一個 launch 檔切 sim / real 兩種模式",
 ],
 "dod": "一條指令拉起可配置的多節點系統，且參數改壞時系統會拒絕而不是默默吃下去。",
 "res": [
   {"t": "ROS 2 官方：Using parameters in a class (Python)", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html", "k": "hands-on", "h": 1},
   {"t": "ROS 2 官方：Launch 系列教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Launch/Launch-Main.html", "k": "hands-on", "h": 3},
 ],
},
{
 "id": "A1-4", "act": "A1", "type": "notable", "track": "main",
 "title": "Executor 與 Callback Group", "en": "Concurrency Model",
 "x": 1200, "y": 710, "hours": 5, "deps": ["A1-2"],
 "desc": "single-threaded vs multi-threaded executor、MutuallyExclusive vs Reentrant callback group、deadlock 的經典成因（callback 裡同步等另一個 callback 的結果）。",
 "why": "『節點活著但不動了』的頭號兇手。JD 的 24/7 reliability 有一半是在跟這個模型搏鬥。",
 "tasks": [
   "重現經典 deadlock：在 subscriber callback 裡同步呼叫 service",
   "用 Reentrant group + MultiThreadedExecutor 解掉它，說出代價（重入、鎖）",
   "寫一個 timer + subscriber 的節點，量測 callback 被延遲的情況",
   "整理：哪些情況該開多執行緒 executor、哪些反而會引入 race",
 ],
 "dod": "能畫出 executor 的排程流程圖，並對一段卡死的程式碼指出卡在哪、兩種解法各有什麼代價。",
 "res": [
   {"t": "ROS 2 官方：Executors 概念", "u": "https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Executors.html", "k": "read", "h": 2},
   {"t": "ROS 2 官方：Using callback groups", "u": "https://docs.ros.org/en/jazzy/How-To-Guides/Using-callback-groups.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A1-5", "act": "A1", "type": "keystone", "track": "main",
 "title": "QoS 策略", "en": "Quality of Service",
 "x": 1200, "y": 840, "hours": 6, "deps": ["A1-4"],
 "desc": "reliability / durability / history / depth / deadline / liveliness / lease。QoS 相容性矩陣：不相容的 pub/sub 會『安靜地』不建立連線。sensor data profile vs 預設 profile。",
 "why": "90% 的『收不到訊息』都是它。這也是你 DDIA/串流系統背景的最佳切入點——backpressure 與 delivery guarantee 的機器人版。",
 "tasks": [
   "做出 QoS 不相容實驗：BEST_EFFORT pub × RELIABLE sub，觀察無訊息、無報錯",
   "用 `ros2 topic info -v` 與 rqt 診斷出不相容原因",
   "註冊 incompatible QoS event callback，把這種故障變成看得見的告警",
   "用 TRANSIENT_LOCAL 做 latched topic（如 map），解釋 durability 的語意",
   "整理相機影像、控制指令、診斷訊息各該用什麼 profile 與理由",
 ],
 "dod": "給任一『訂了但收不到』的場景，能在 5 分鐘內用 CLI 判定是否 QoS 問題並指出是哪個欄位不相容。",
 "res": [
   {"t": "ROS 2 官方：About QoS settings", "u": "https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html", "k": "read", "h": 2},
   {"t": "ROS 2 官方：QoS 不相容 demo", "u": "https://docs.ros.org/en/jazzy/Tutorials/Demos/Quality-of-Service.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A1-6", "act": "A1", "type": "normal", "track": "main",
 "title": "DDS 與節點發現", "en": "Discovery & RMW",
 "x": 1000, "y": 950, "hours": 4, "deps": ["A1-5"],
 "desc": "RMW 抽象層、DDS 的去中心化發現（multicast announce）、ROS_DOMAIN_ID、discovery server 模式、為什麼大型系統會有 discovery storm。",
 "why": "跨機通訊（Mac↔Jetson）與『同事的節點出現在我的機器上』這類靈異事件，根源都在發現機制。",
 "tasks": [
   "兩個 domain id 隔離實驗：同機器、不同 ROS_DOMAIN_ID，互相看不見",
   "切換 rmw（如 CycloneDDS ↔ FastDDS），觀察行為差異與設定方式",
   "解釋 multicast 在 Docker bridge / 公司 Wi-Fi 下失靈的原因與替代方案（discovery server / unicast peers）",
   "抓一次 discovery 流量（tcpdump port 7400），看 participant announcement",
 ],
 "dod": "能畫出兩台機器上的節點從開機到互相收到訊息的完整時序（announce → match → data），並指出每一步可能斷在哪。",
 "res": [
   {"t": "ROS 2 官方：Different middleware vendors", "u": "https://docs.ros.org/en/jazzy/Concepts/Advanced/About-Different-Middleware-Vendors.html", "k": "read", "h": 1},
   {"t": "ROS 2 官方：Discovery 概念", "u": "https://docs.ros.org/en/jazzy/Concepts/Basic/About-Discovery.html", "k": "read", "h": 1},
 ],
},
{
 "id": "A1-7", "act": "A1", "type": "normal", "track": "main",
 "title": "tf2 與時間", "en": "Transforms & Time",
 "x": 1400, "y": 950, "hours": 5, "deps": ["A1-3"],
 "desc": "座標框架樹、static vs dynamic transform、lookup 的時間語意（過去的 transform 要用 buffer 查）、sim time 與 wall time。平台工程師不必精通幾何，但要看得懂 tf 錯誤。",
 "why": "『TF_OLD_DATA』『frame does not exist』是值班時最常見的紅字之一——你要能判斷是誰的鍋，才能把 ticket 派對人。",
 "tasks": [
   "手發 static transform，用 `ros2 run tf2_tools view_frames` 畫出 tf 樹",
   "寫一個 broadcaster + listener，查 100ms 前的 transform",
   "重現 TF_OLD_DATA 與 extrapolation 錯誤，說出各自成因",
   "解釋 use_sim_time 為什麼會讓 tf 查詢整組壞掉（時鐘不一致）",
 ],
 "dod": "看到任一 tf 錯誤訊息，能分類成「樹斷了／時間不同步／查詢時機錯」三類之一並說出下一步。",
 "res": [
   {"t": "ROS 2 官方：tf2 教學系列", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html", "k": "hands-on", "h": 4},
 ],
},

# ─────────────────────────── A2 · 執行時平台 ───────────────────────────
{
 "id": "A2-1", "act": "A2", "type": "notable", "track": "main",
 "title": "systemd 深潛", "en": "systemd Deep Dive",
 "x": 1000, "y": 1090, "hours": 5, "deps": ["A0-1"],
 "desc": "unit 檔剖析、Restart= 策略、WatchdogSec 與 sd_notify、依賴（After/Requires/Wants）、resource control（cgroups：MemoryMax、CPUQuota）、journald。",
 "why": "Always-on robot 的第一層就是 init 系統。你會 K8s 的 liveness probe——systemd watchdog 就是單機版，而機器人上沒有 K8s。",
 "tasks": [
   "寫一個 unit 跑 ROS 2 節點：Restart=on-failure、RestartSec、StartLimitBurst 全配上",
   "用 WatchdogSec + sd_notify 做心跳，故意讓程式 hang 住觀察被重啟",
   "用 MemoryMax 限制記憶體，觀察超限時發生什麼（誰殺的、什麼訊號）",
   "`journalctl -u <unit> --since -1h -o json` 撈出結構化 log",
   "解釋 After= 與 Requires= 的差別，以及只寫 Requires 不寫 After 的坑",
 ],
 "dod": "做出一個殺不死的 demo 服務：kill -9、OOM、hang 三種死法都能在 10 秒內自動復活並留下可查的日誌。",
 "res": [
   {"t": "systemd.service 官方 man page", "u": "https://man7.org/linux/man-pages/man5/systemd.service.5.html", "k": "read", "h": 2},
   {"t": "systemd.exec / resource-control man page", "u": "https://man7.org/linux/man-pages/man5/systemd.resource-control.5.html", "k": "read", "h": 1},
 ],
},
{
 "id": "A2-2", "act": "A2", "type": "normal", "track": "main",
 "title": "Lifecycle Node", "en": "Managed Nodes",
 "x": 1400, "y": 1090, "hours": 4, "deps": ["A1-4"],
 "desc": "unconfigured → inactive → active 的標準狀態機、transition service、為什麼硬體驅動節點都該是 lifecycle node（資源取得與資料流分離）。",
 "why": "沒有 lifecycle 的系統只有『開著』和『掛了』兩種狀態；有了它，才有『正在啟動』『降級中』『安全停止』——24/7 系統的語彙。",
 "tasks": [
   "寫一個 lifecycle node，在 on_configure 開資源、on_activate 才開始發布",
   "用 `ros2 lifecycle set` 手動驅動狀態機，觀察每個 transition 的 callback",
   "示範 on_configure 失敗時的行為（回到 unconfigured），對比一般節點直接 crash",
   "訂閱 transition event topic，把狀態變化印出來",
 ],
 "dod": "能說出把一個普通相機驅動節點改成 lifecycle node 的具體改法，以及系統 bringup 順序如何因此變得可控。",
 "res": [
   {"t": "ROS 2 demos：lifecycle 範例與設計文件", "u": "https://github.com/ros2/demos/tree/jazzy/lifecycle", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A2-3", "act": "A2", "type": "normal", "track": "main",
 "title": "系統編排與 bringup", "en": "Orchestration",
 "x": 1200, "y": 1210, "hours": 5, "deps": ["A1-3", "A2-2"],
 "desc": "用 launch 的 event handler（OnProcessExit / OnStateTransition）做開機順序控制、respawn、composable container 把多節點塞進同一 process 省 IPC。整機 bringup = systemd 拉 launch、launch 拉節點。",
 "why": "『機器人開機後有時相機沒起來』——這種 flaky bringup 是平台工程師的日常。把順序寫成程式，而不是靠 sleep 碰運氣。",
 "tasks": [
   "寫 bringup launch：驅動節點 activate 成功後才啟動上層節點（OnStateTransition）",
   "給關鍵節點配 respawn=True + respawn_delay，殺掉觀察重生",
   "把兩個節點改成 composable node 放進同一 container，量測 CPU 差異",
   "把整套 bringup 掛進 systemd unit，開機自動拉起",
 ],
 "dod": "冷開機 20 次，系統每次都以正確順序進入全部 active，一次都不 flaky。",
 "res": [
   {"t": "ROS 2 官方：Using event handlers in launch", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Launch/Using-Event-Handlers.html", "k": "hands-on", "h": 2},
   {"t": "ROS 2 官方：Composition 概念與教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Composition.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A2-4", "act": "A2", "type": "keystone", "track": "main",
 "title": "健康監控與 watchdog", "en": "Health & Watchdog",
 "x": 1200, "y": 1340, "hours": 6, "deps": ["A2-1", "A2-3"],
 "desc": "diagnostic_msgs 與 diagnostic_updater/aggregator、topic 頻率監控（跌頻＝早期病徵）、多層 watchdog（節點內心跳 → systemd → 硬體 WDT）、故障時的降級策略。",
 "why": "這是本章的分水嶺：過了這裡，你設計的每個系統都會自帶『它現在健康嗎』的答案，而不是等客戶打電話來告訴你。",
 "tasks": [
   "給節點加 diagnostic_updater，回報自身頻率與內部狀態",
   "用 diagnostic_aggregator 匯總成一頁系統健康總覽",
   "寫一個 topic 頻率 watchdog：某 topic 跌出範圍就發告警並嘗試重啟該節點",
   "畫出你的三層 watchdog 架構圖：各層偵測什麼、誰重啟誰、多久算逾時",
   "設計一個降級案例：相機掛了，系統退到安全模式而不是整機停擺",
 ],
 "dod": "拔掉任一感測器（或殺掉任一節點），30 秒內告警出現、系統進入明確的降級狀態、且事後能從日誌還原時間線。",
 "res": [
   {"t": "ros/diagnostics（diagnostic_updater / aggregator）", "u": "https://github.com/ros/diagnostics", "k": "hands-on", "h": 3},
 ],
},
{
 "id": "A2-5", "act": "A2", "type": "normal", "track": "side",
 "title": "配置與版本管理", "en": "Config & Release",
 "x": 940, "y": 1340, "hours": 3, "deps": ["A2-3"],
 "desc": "機器人隊的配置分層（機型共通 / 單機校正 / 部署環境）、參數檔的版本化、image tag 與 rollback 策略。",
 "why": "『那台機器人行為怪怪的』有三成是配置漂移。你的資料工程直覺（schema、版本、lineage）在這裡直接變現。",
 "tasks": [
   "設計參數檔分層：base.yaml + per-robot overlay，launch 時合併",
   "把單機校正值（如相機外參）與程式碼分離，說明更新流程",
   "寫出 rollback 劇本：新版 image 部署後行為異常，5 分鐘內退回上一版",
 ],
 "dod": "任一台機器人的完整配置可以用「image tag + config 版本 + 校正檔版本」三個座標唯一描述。",
 "res": [
   {"t": "ROS 2 官方：Using ros2 param（YAML 匯出入）", "u": "https://docs.ros.org/en/jazzy/How-To-Guides/Using-ros2-param.html", "k": "read", "h": 1},
 ],
},

# ─────────────────────────── A3 · 可靠性除錯 ───────────────────────────
{
 "id": "A3-1", "act": "A3", "type": "notable", "track": "main",
 "title": "行程級除錯工具", "en": "strace / lsof / gdb",
 "x": 1000, "y": 1480, "hours": 5, "deps": ["A2-1"],
 "desc": "strace（卡在哪個 syscall）、lsof（誰佔著裝置/埠）、/proc 檔案系統、gdb attach 到活的行程、py-spy dump Python 行程的呼叫堆疊。",
 "why": "『logs are incomplete』是 JD 原文。log 沒寫的，用工具直接問作業系統——這是把你和只會看 log 的人分開的那條線。",
 "tasks": [
   "用 strace -p 診斷一個卡住的節點，判讀它 block 在哪個 syscall",
   "用 lsof 找出誰佔著 /dev/ttyUSB0 導致驅動起不來",
   "gdb attach 到跑著的 C++ 行程，印出所有執行緒的 backtrace",
   "py-spy dump 一個 hang 住的 Python 節點，找出卡住的那一行",
   "讀 /proc/<pid>/status 與 /proc/<pid>/fd，說出各欄位在查案時的用途",
 ],
 "dod": "面對一個「活著但不動」的行程，不重啟它，5 分鐘內產出「卡在哪、等什麼、誰害的」三行結論。",
 "res": [
   {"t": "Julia Evans：strace zine（免費線上版）", "u": "https://jvns.ca/strace-zine-unfolded.pdf", "k": "read", "h": 1},
   {"t": "py-spy README", "u": "https://github.com/benfred/py-spy", "k": "hands-on", "h": 1},
 ],
},
{
 "id": "A3-2", "act": "A3", "type": "normal", "track": "main",
 "title": "效能剖析", "en": "perf & Flame Graphs",
 "x": 940, "y": 1610, "hours": 4, "deps": ["A3-1"],
 "desc": "perf record/report、flame graph、CPU 使用率的正確讀法（iowait vs user vs sys）、Python 側用 py-spy record。找出「CPU 吃滿但沒人知道在忙什麼」。",
 "why": "Jetson 只有 6 個小核心。桌機上無感的浪費，在邊緣裝置上就是掉幀與過熱降頻。",
 "tasks": [
   "perf record 一個忙碌的 C++ 節點，產出 flame graph 指出前三名熱點",
   "py-spy record 一個 Python 節點 60 秒，找出意料之外的熱點",
   "重現並診斷一次 iowait 型的「CPU 看起來很忙」（其實在等磁碟）",
   "量測 serialization 成本：同一 msg 走 intra-process vs 跨 process 的 CPU 差",
 ],
 "dod": "拿到任一「這台機器人好慢」的抱怨，能在 30 分鐘內產出 flame graph 並指名道姓熱點函式。",
 "res": [
   {"t": "Brendan Gregg：Flame Graphs", "u": "https://www.brendangregg.com/flamegraphs.html", "k": "read", "h": 2},
   {"t": "perf Examples（Gregg）", "u": "https://www.brendangregg.com/perf.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A3-3", "act": "A3", "type": "normal", "track": "main",
 "title": "記憶體與 OOM", "en": "Memory & OOM",
 "x": 1260, "y": 1610, "hours": 4, "deps": ["A3-1"],
 "desc": "RSS vs VSZ、記憶體洩漏的觀察法（長期趨勢）、OOM killer 的行為（SIGKILL、oom_score）、cgroup 記憶體限制、dmesg 讀 OOM 紀錄。Jetson 8GB 是稀缺資源。",
 "why": "OOM 是最殘忍的死法——不給訊號、不留遺言（SIGKILL 不可捕捉）。不懂它的人會花三天找一個「隨機重啟」的鬼。",
 "tasks": [
   "寫個慢速洩漏的節點，用 RSS 趨勢圖抓到它",
   "觸發一次 OOM kill，從 dmesg / journalctl -k 還原「誰被殺、為什麼是它」",
   "解釋為什麼 OOM kill 是 SIGKILL 而不是 SIGTERM，以及這對「優雅清理」的含義",
   "用 systemd MemoryMax 把爆記憶體的服務關進籠子，保護整機",
 ],
 "dod": "看到「行程無故消失、log 斷在一半」，能在 3 分鐘內確認或排除 OOM，並指出證據位置。",
 "res": [
   {"t": "kernel 文件：OOM killer / cgroup v2 memory", "u": "https://docs.kernel.org/admin-guide/cgroup-v2.html", "k": "read", "h": 2},
 ],
},
{
 "id": "A3-4", "act": "A3", "type": "keystone", "track": "main",
 "title": "混沌演練", "en": "Chaos Drills",
 "x": 1200, "y": 1750, "hours": 6, "deps": ["A3-2", "A3-3", "A2-4"],
 "desc": "對自己的系統做故障注入：殺行程、拔（模擬）裝置、塞滿磁碟、netem 加延遲丟包、時鐘跳變。每種故障都要走完「偵測 → 告警 → 降級 → 復原 → 復盤」全流程。",
 "why": "這是整章的畢業考，也是面試的黃金素材：『我對自己的系統注入過 12 種故障，其中 4 種讓我重寫了監控』比任何證照都有說服力。",
 "tasks": [
   "列出 10+ 種故障假設（行程/裝置/資源/網路/時間五類至少各兩種）",
   "逐一注入，記錄：多久被偵測到、告警長怎樣、系統行為、復原方式",
   "至少修復 3 個演練暴露的監控盲點或單點故障",
   "用 tc netem 對 DDS 流量加 200ms 延遲與 5% 丟包，觀察各 QoS profile 的行為差異",
   "寫成 chaos-drill 報告（表格：故障 × 偵測時間 × 行為 × 修復），放進 repo",
 ],
 "dod": "10 種故障全部有紀錄、有結論；其中至少 3 種的偵測時間因你的修復而顯著縮短。",
 "res": [
   {"t": "tc-netem man page", "u": "https://man7.org/linux/man-pages/man8/tc-netem.8.html", "k": "hands-on", "h": 1},
   {"t": "Google SRE Book：Ch.4 SLO / Ch.15 Postmortem 文化", "u": "https://sre.google/sre-book/table-of-contents/", "k": "read", "h": 3},
 ],
},
{
 "id": "A3-5", "act": "A3", "type": "normal", "track": "side",
 "title": "即時性基礎", "en": "Real-time Basics",
 "x": 700, "y": 1750, "hours": 4, "deps": ["A1-4"],
 "desc": "soft vs hard real-time、PREEMPT_RT 是什麼、排程策略（SCHED_FIFO）、cyclictest 量測延遲抖動、為什麼 malloc 和 page fault 是即時迴路的敵人。概念與量測為主，不深入 RT 開發。",
 "why": "JD 的 nice-to-have。你不必寫 RT 程式，但要能回答『這個控制迴路需要 RT kernel 嗎』——答錯方向會浪費整個月。",
 "tasks": [
   "用 cyclictest 量測目前環境的延遲分佈，解讀 max latency",
   "解釋 jitter 對 1kHz 控制迴路 vs 30Hz 視覺管線的不同影響",
   "整理：哪些手段能不換 kernel 就改善延遲（isolcpus、優先權、鎖記憶體）",
 ],
 "dod": "能對「我們需要 RT kernel 嗎」給出一個含量測數據與成本的建議，而不是感覺。",
 "res": [
   {"t": "ROS 2 官方：Real-time programming 設計文件", "u": "https://design.ros2.org/articles/realtime_background.html", "k": "read", "h": 2},
 ],
},

# ─────────────────────────── A4 · 可觀測性與交付 ───────────────────────────
{
 "id": "A4-1", "act": "A4", "type": "normal", "track": "main",
 "title": "日誌管線", "en": "Logging Pipeline",
 "x": 950, "y": 1890, "hours": 4, "deps": ["A2-1"],
 "desc": "ROS 2 logging（rcutils → spdlog）、/rosout topic、log 等級與節點級動態調整、把 stdout 導進 journald 再外送、結構化欄位設計。",
 "why": "查案的起點永遠是 log。把 30 個節點的 log 收攏成一條有時間戳、可過濾的流，是平台的基本供水系統。",
 "tasks": [
   "動態調某個節點的 log level（不重啟），確認生效",
   "比較 /rosout 與 stdout 兩條路徑：誰會漏什麼（早期訊息、非 ROS 行程）",
   "設計 log 欄位規範：時間、節點、severity、事件碼，寫進 platform-lab 文件",
   "做一次「從症狀到 log 證據」的演練：給症狀，10 分鐘內撈出關鍵三行",
 ],
 "dod": "任一節點的 log 都能用單一指令按時間窗與等級撈出，且早於 ROS 初始化的錯誤也不會消失。",
 "res": [
   {"t": "ROS 2 官方：Logging 概念與設定", "u": "https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Logging.html", "k": "read", "h": 1},
 ],
},
{
 "id": "A4-2", "act": "A4", "type": "notable", "track": "main",
 "title": "rosbag2 與 MCAP", "en": "Record & Replay",
 "x": 1200, "y": 1890, "hours": 4, "deps": ["A1-5"],
 "desc": "MCAP 格式（Jazzy 預設）、錄製時的 QoS override、snapshot mode（環形緩衝，事故前 N 秒）、bag 重播除錯法、bag 就是機器人的資料工程原料。",
 "why": "客戶說『它昨天下午怪怪的』——你唯一的時光機就是 bag。snapshot mode 是事故取證的行車記錄器。",
 "tasks": [
   "錄一段多 topic bag（MCAP），用 `ros2 bag info` 檢視",
   "設定 snapshot mode：平時只進環形緩衝，觸發時才落盤最後 30 秒",
   "重播 bag 餵給下游節點除錯，處理 use_sim_time 與時鐘來源",
   "用 Python API 讀 MCAP 抽出某 topic 做離線分析（你最熟的部分）",
 ],
 "dod": "系統常駐 snapshot 錄製，任何人喊「剛剛發生什麼事」都能在 2 分鐘內交出事發前 30 秒的資料。",
 "res": [
   {"t": "ROS 2 官方：rosbag2 教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html", "k": "hands-on", "h": 2},
   {"t": "MCAP 官方文件", "u": "https://mcap.dev/", "k": "read", "h": 1},
 ],
},
{
 "id": "A4-3", "act": "A4", "type": "normal", "track": "main",
 "title": "Metrics 與 Prometheus", "en": "Robot Metrics",
 "x": 1450, "y": 1890, "hours": 3, "deps": ["A2-4"],
 "desc": "你已會 Prometheus/Grafana——這節點只做機器人化：node_exporter 上 Jetson（溫度、降頻、記憶體）、把 topic 頻率與 diagnostic 轉成 metrics、告警規則設計。",
 "why": "把你最強的技能插進機器人領域，一小時內就有「這我熟」的立足感——求職路上要留幾個這種節點。",
 "tasks": [
   "Jetson 跑 node_exporter，Grafana 畫出溫度/CPU/記憶體",
   "寫一個小 exporter 把關鍵 topic 頻率與 diagnostic 狀態轉成 Prometheus metrics",
   "設 3 條告警規則：topic 跌頻、溫度過熱、記憶體趨勢異常",
 ],
 "dod": "一個 Grafana dashboard 能在 10 秒內回答「機器人現在健康嗎、過去一小時發生過什麼」。",
 "res": [
   {"t": "Prometheus node_exporter", "u": "https://github.com/prometheus/node_exporter", "k": "hands-on", "h": 1},
 ],
},
{
 "id": "A4-4", "act": "A4", "type": "normal", "track": "main",
 "title": "Foxglove 遠端除錯", "en": "Foxglove",
 "x": 950, "y": 2010, "hours": 3, "deps": ["A4-2"],
 "desc": "foxglove_bridge（WebSocket）即時看遠端機器人、layout 設計、回放 MCAP、比 X11 轉發 RViz 靠譜十倍的遠端視覺化。",
 "why": "客戶現場的機器人不會給你接螢幕。一條 WebSocket 就能看到全部 topic，是 deployment support 的標配工具。",
 "tasks": [
   "Jetson（或容器）跑 foxglove_bridge，從 Mac 瀏覽器連上看即時 topic",
   "做一個值班 layout：健康總覽、關鍵 topic 圖表、log 面板",
   "把 A4-2 的 snapshot MCAP 拖進 Foxglove 回放一次事故",
 ],
 "dod": "從一台只有 SSH 的機器人，3 分鐘內建立完整視覺化——不裝任何 GUI 套件。",
 "res": [
   {"t": "Foxglove 官方文件", "u": "https://docs.foxglove.dev/docs", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A4-5", "act": "A4", "type": "keystone", "track": "main",
 "title": "ROS 2 CI", "en": "CI for Robots",
 "x": 1450, "y": 2010, "hours": 5, "deps": ["A0-2"],
 "desc": "你已會 GitHub Actions——這裡補 ROS 特有的：colcon test / launch_testing（整合測試起真節點）、industrial_ci、multi-arch image build（給 Jetson 的 arm64）。",
 "why": "本章分水嶺：過了這裡，你 repo 裡每個 package 的改動都有測試與 arm64 image 自動產出——面試官一眼就看得出這是生產習慣。",
 "tasks": [
   "給一個 package 寫單元測試 + launch_testing 整合測試，本地 colcon test 通過",
   "GitHub Actions 跑 industrial_ci（或等效 colcon 流程）",
   "CI 產 arm64 image（buildx），push 到 registry，Jetson 直接 pull 得動",
   "加一條 lint gate（ament_lint / ruff），紅了就擋 merge",
 ],
 "dod": "對 main 的每個 PR 自動跑測試並產出 Jetson 可用的 image；全綠才能合。",
 "res": [
   {"t": "ros-industrial/industrial_ci", "u": "https://github.com/ros-industrial/industrial_ci", "k": "hands-on", "h": 2},
   {"t": "ROS 2 官方：Testing 主頁", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Testing/Testing-Main.html", "k": "read", "h": 1},
 ],
},
{
 "id": "A4-6", "act": "A4", "type": "notable", "track": "main",
 "title": "機器人→雲資料管線", "en": "Robot-to-Cloud",
 "x": 1200, "y": 2130, "hours": 5, "deps": ["A4-2", "A4-3"],
 "desc": "把 bag/metrics/log 從機器人送上雲：頻寬預算（現場常是 4G）、斷線續傳、分級上傳（metrics 即時、bag 事件觸發、全量夜間）、資料落地後的索引。",
 "why": "JD 明寫 data pipelines between robots and cloud。這題你比多數機器人工程師強——用資料工程的肌肉打薄弱環節。",
 "tasks": [
   "設計上傳分級策略並寫成文件：什麼資料、何時傳、頻寬上限、斷線行為",
   "實作事件觸發上傳：告警發生 → 對應 snapshot bag 自動上雲（S3 或任一物件儲存）",
   "做斷線續傳測試：上傳中拔網路，恢復後不重傳已完成部分",
   "雲端側建最小索引：能按「機器人 × 時間窗 × 告警類型」查到對應 bag",
 ],
 "dod": "拔網路線再插回去，該上傳的資料最終全部到位、無重複、且雲端查得到。",
 "res": [
   {"t": "MCAP Python API（讀寫與切片）", "u": "https://mcap.dev/docs/python", "k": "hands-on", "h": 2},
 ],
},

# ─────────────────────────── A5 · Jetson 部署 ───────────────────────────
{
 "id": "A5-1", "act": "A5", "type": "notable", "track": "main",
 "title": "JetPack 刷機", "en": "Flash & Firmware",
 "x": 1000, "y": 2270, "hours": 4, "deps": ["A2-1"],
 "desc": "Jetson Orin Nano 刷 JetPack 6.x：UEFI 韌體版本地雷（舊韌體不能直上 JP6）、SDK Manager vs SD 卡映像、NVMe 開機、電源模式（7W/15W/25W）與 jetson_clocks。",
 "why": "刷機是所有 Jetson 專案的第一道濾網——韌體版本錯誤的救磚流程比正常刷機難十倍。照 checklist 走，別探索。",
 "tasks": [
   "確認目前韌體版本，按官方順序升級到可上 JetPack 6.x 的版本",
   "刷 JetPack 6.x 並完成 oem-config，跑通 `sudo tegrastats`",
   "（強烈建議）裝 NVMe SSD 並把 rootfs 移過去，比 SD 卡快且耐寫",
   "用 nvpmodel 切換電源模式，記錄各模式下的 CPU/GPU 時脈差異",
 ],
 "dod": "Jetson 乾淨開機進 JetPack 6.x，tegrastats 正常，且你能說出手上這台的韌體版本與升級路徑。",
 "res": [
   {"t": "NVIDIA：Jetson Orin Nano Developer Kit Getting Started", "u": "https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit", "k": "hands-on", "h": 2},
   {"t": "repo 內 docs/04-jetson-orin-nano.md（韌體地雷整理）", "u": "https://github.com/ChesterHsieh/Always_try_to_learn/blob/main/ros2-robotics-mastery/docs/04-jetson-orin-nano.md", "k": "read", "h": 1},
 ],
},
{
 "id": "A5-2", "act": "A5", "type": "normal", "track": "main",
 "title": "Jetson 上的 ROS 2 容器", "en": "ROS 2 on Edge",
 "x": 1000, "y": 2390, "hours": 4, "deps": ["A5-1"],
 "desc": "JetPack 6 是 Ubuntu 22.04（原生對應 Humble）；用容器跑 Jazzy 避開發行版綁定。jetson-containers 專案、GPU 進容器（--runtime nvidia）、裝置掛載。",
 "why": "『系統 Ubuntu 版本 ≠ 你要的 ROS 版本』是邊緣部署的常態。容器化讓你的 Mac 與 Jetson 跑同一套 image——CI 產物直接落地。",
 "tasks": [
   "Jetson 用 A4-5 CI 產的 arm64 image 跑起 ROS 2 節點",
   "確認容器內看得到 GPU（--runtime nvidia + torch.cuda 或 deviceQuery）",
   "把 /dev 裝置與 host network 正確掛進容器，DDS 跨機發現正常",
   "整理 image 更新流程：pull → 換 tag → systemd 重啟服務",
 ],
 "dod": "同一個 image 在 Mac（模擬）與 Jetson（真機）都跑得起來，且 Jetson 上以 systemd 管理、開機自動拉起。",
 "res": [
   {"t": "dusty-nv/jetson-containers", "u": "https://github.com/dusty-nv/jetson-containers", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A5-3", "act": "A5", "type": "keystone", "track": "main",
 "title": "Mac ↔ Jetson 分散式系統", "en": "Distributed Robot",
 "x": 1200, "y": 2510, "hours": 5, "deps": ["A5-2", "A1-6"],
 "desc": "跨實體機器的 DDS：同網段 multicast 發現、Wi-Fi 的坑（AP isolation）、discovery server 替代方案、時鐘同步（chrony/PTP）、頻寬管理（影像 topic 壓縮）。",
 "why": "本章分水嶺：從這裡開始你有一台「真的機器人」（Jetson）與一台「地面站」（Mac）——之後所有章節都在這個拓撲上做。",
 "tasks": [
   "Mac 與 Jetson 同 domain id，跨機 echo 到彼此的 topic",
   "刻意破壞再修復：AP isolation / Docker bridge 導致發現失敗，改用 discovery server 或 unicast peers 解掉",
   "chrony 同步兩機時鐘，量測 offset，說明時鐘漂移對 tf 與 bag 的影響",
   "影像 topic 改用 compressed transport，量測頻寬差",
 ],
 "dod": "Mac 上的 Foxglove/CLI 能穩定觀測 Jetson 全部 topic，斷網重連後 60 秒內自動恢復。",
 "res": [
   {"t": "ROS 2 官方：Discovery Server 教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Advanced/Discovery-Server/Discovery-Server.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A5-4", "act": "A5", "type": "normal", "track": "side",
 "title": "邊緣推論", "en": "Edge Inference",
 "x": 700, "y": 2510, "hours": 5, "deps": ["A5-2"],
 "desc": "在 Jetson 用 TensorRT 跑一個偵測模型並包成 ROS 2 節點：模型轉換（ONNX→engine）、zero-copy 的意義、量測 fps 與功耗。",
 "why": "JD 的 nice-to-have（Physical AI models）。做一次就能在面試聊 GPU 加速管線的實際數字，投資報酬率高的支線。",
 "tasks": [
   "把一個 ONNX 偵測模型轉成 TensorRT engine，量測 fps",
   "包成 ROS 2 節點：訂影像、發偵測框",
   "比較 15W vs 25W 模式下的 fps 與 tegrastats 溫度",
 ],
 "dod": "Jetson 即時偵測管線跑通，能報出「模型 × 精度 × fps × 功耗」一組數字。",
 "res": [
   {"t": "NVIDIA TensorRT 文件", "u": "https://docs.nvidia.com/deeplearning/tensorrt/", "k": "read", "h": 2},
 ],
},

# ─────────────────────────── A6 · 硬體介面 ───────────────────────────
{
 "id": "A6-1", "act": "A6", "type": "keystone", "track": "main",
 "title": "SocketCAN 基礎", "en": "SocketCAN",
 "x": 1200, "y": 2650, "hours": 5, "deps": ["A2-1"],
 "desc": "CAN bus 原理（廣播、ID 仲裁、越小越優先）、Linux SocketCAN（can0 是網路介面！）、vcan 虛擬介面、candump/cansend/canbusload、DBC 檔解碼、錯誤框與 bus-off。",
 "why": "JD 點名 CAN。它是馬達控制器與電池的母語；SocketCAN 把它變成你熟悉的 socket 程式設計——先在 vcan 上練，零硬體成本。",
 "tasks": [
   "建 vcan0，candump 一邊、cansend 另一邊，看到自己的框",
   "寫 Python（python-can）收發，模擬一個「馬達控制器」回覆狀態框",
   "用 cangen 灌流量、canbusload 觀察匯流排負載，解釋負載過高會發生什麼",
   "解釋 ID 仲裁如何不破壞資料就決定誰先傳，以及 error frame / bus-off 的意義",
   "拿一個公開 DBC 檔（如 Tesla/公開資料集）解碼一段 log",
 ],
 "dod": "在 vcan 上跑通「指令框 → 模擬控制器 → 狀態框」閉環，並能白板講清楚仲裁與 bus-off。",
 "res": [
   {"t": "Linux kernel 文件：SocketCAN", "u": "https://docs.kernel.org/networking/can.html", "k": "read", "h": 2},
   {"t": "python-can 文件", "u": "https://python-can.readthedocs.io/", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A6-2", "act": "A6", "type": "notable", "track": "main",
 "title": "Jetson CAN 實戰", "en": "CAN on Jetson",
 "x": 1000, "y": 2770, "hours": 5, "deps": ["A6-1", "A5-1"],
 "desc": "Orin Nano 有內建 CAN controller（40-pin header 出 CAN_TX/RX），但需要外接 transceiver（如 SN65HVD230，百元內）。pinmux 設定、ip link 設 bitrate、迴環測試、（可選）接第二個節點做真雙機。",
 "why": "把「我懂 CAN」升級成「我在真硬體上調通過 CAN」——面試時這兩句話的重量差三倍。",
 "tasks": [
   "查 Orin Nano 40-pin header 的 CAN 腳位，接上 transceiver（無第二裝置就先 loopback 模式）",
   "設定 pinmux 啟用 CAN controller，`ip link set can0 up type can bitrate 500000`",
   "candump 收到真實電氣訊號的框（loopback 或雙節點）",
   "把收到的 CAN 狀態框轉發成 ROS 2 topic（一個小橋接節點）",
   "整理一頁「Jetson CAN 啟用 checklist」進 repo（pinmux、模組、常見錯誤）",
 ],
 "dod": "Jetson 的 can0 以 500kbps 收發實際電氣框，且資料流進 ROS 2 topic 可被 Foxglove 看到。",
 "res": [
   {"t": "NVIDIA Jetson 官方文件：Controller Area Network (CAN)", "u": "https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/HR/ControllerAreaNetworkCan.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A6-3", "act": "A6", "type": "normal", "track": "main",
 "title": "感測器接入", "en": "Sensor Bring-up",
 "x": 1400, "y": 2770, "hours": 4, "deps": ["A5-2"],
 "desc": "USB 相機（V4L2 → ros2 camera 節點）、serial 裝置（IMU/GPS 類）、udev rule 固定裝置名與權限、熱插拔偵測。",
 "why": "『接上去沒反應』的九成是權限、裝置名漂移、或 udev。這一節把它們變成 checklist 而不是玄學。",
 "tasks": [
   "USB 相機接 Jetson，v4l2-ctl 列出格式，跑 usb_cam 節點發布影像",
   "寫 udev rule：固定 symlink（/dev/my_imu）+ 權限，重插驗證",
   "模擬裝置消失（拔掉），確認驅動節點的行為被 A2-4 的健康監控抓到",
   "整理「新感測器接入 SOP」：接線 → 權限 → 驅動 → topic → 健康監控",
 ],
 "dod": "任一新 USB 裝置照你的 SOP 在 30 分鐘內走完接入到監控的全流程。",
 "res": [
   {"t": "ros-drivers/usb_cam", "u": "https://github.com/ros-drivers/usb_cam", "k": "hands-on", "h": 1},
 ],
},
{
 "id": "A6-4", "act": "A6", "type": "notable", "track": "main",
 "title": "ros2_control 與硬體抽象", "en": "Hardware Abstraction",
 "x": 1200, "y": 2890, "hours": 6, "deps": ["A1-7", "A6-1"],
 "desc": "ros2_control 架構：controller_manager、hardware_interface（read/write 迴圈）、controller 與硬體外掛的分離。寫一個假硬體（或 vcan 後端）的 SystemInterface。",
 "why": "這是機器人業界『clean API between sensors/actuators and application layers』的標準答案——JD 那句話幾乎就是在描述 ros2_control。",
 "tasks": [
   "讀懂 ros2_control 架構圖：誰呼叫誰、read/update/write 的迴圈在哪",
   "寫一個最小 SystemInterface（假硬體：狀態=上次指令+雜訊），跑 JTC 或 velocity controller",
   "把後端換成 vcan：write 送 CAN 指令框、read 收狀態框——接上 A6-1 的模擬控制器",
   "說明 controller 熱切換（switch_controller）的流程與用途",
 ],
 "dod": "同一個 controller 不改一行，後端從假硬體換成 vcan 模擬控制器仍正常運作——用實作證明抽象層存在。",
 "res": [
   {"t": "ros2_control 官方文件", "u": "https://control.ros.org/jazzy/index.html", "k": "read", "h": 3},
   {"t": "ros2_control_demos（範例集）", "u": "https://github.com/ros-controls/ros2_control_demos", "k": "hands-on", "h": 3},
 ],
},
{
 "id": "A6-5", "act": "A6", "type": "normal", "track": "main",
 "title": "乾淨硬體 API 設計", "en": "Clean HW APIs",
 "x": 1000, "y": 3010, "hours": 4, "deps": ["A6-2", "A6-4"],
 "desc": "把前面做的橋接與驅動整併成一個有版本、有文件、有錯誤語意的 package：介面用 ROS 2 idiom（lifecycle + diagnostic + 標準 msg）、內部細節（CAN ID、暫存器）完全隱藏。",
 "why": "JD 的 Hardware interfaces 條目要的不是會接硬體，而是接完之後別人不用懂硬體。這是資深與資淺的分界線。",
 "tasks": [
   "為你的 CAN 橋接節點定義公開介面：topics/services、單位、座標、錯誤碼，寫成 README",
   "把它改成 lifecycle node、掛上 diagnostic，錯誤語意分「可重試/需人工/致命」三級",
   "請一個不懂 CAN 的人（或 Claude 扮演）只讀 README 使用你的節點，記錄卡點並修文件",
 ],
 "dod": "使用者不需要知道任何 CAN ID 就能完整使用該硬體，且每種故障都有明確的對外表現。",
 "res": [
   {"t": "ROS 2 官方：Developer guide（介面設計慣例）", "u": "https://docs.ros.org/en/jazzy/The-ROS2-Project/Contributing/Developer-Guide.html", "k": "read", "h": 1},
 ],
},

# ─────────────────────────── A7 · C++ 生產力 ───────────────────────────
{
 "id": "A7-1", "act": "A7", "type": "notable", "track": "main",
 "title": "現代 C++ 核心", "en": "Modern C++ Core",
 "x": 1750, "y": 1090, "hours": 6, "deps": ["A0-2"],
 "desc": "為讀寫生產級機器人程式碼所需的最小集合：RAII、unique_ptr/shared_ptr、move 語意、const 正確性、lambda、std::thread/mutex 基礎。目標是讀懂 rclcpp 生態並寫出不漏資源的程式。",
 "why": "JD 第一條 requirement 就是 C++。你不必成為模板法師，但 code review 時看不懂 move 跟裸指標的差別會直接出局。",
 "tasks": [
   "寫一個 RAII 包裝（開檔/開 socket），示範例外發生時資源仍被釋放",
   "解釋 unique_ptr vs shared_ptr 的所有權語意，各給一個機器人場景",
   "寫一個雙執行緒 + mutex 的計數器，再用 data race 版本對照（thread sanitizer 抓）",
   "讀一段 rclcpp 官方 demo 程式碼，標出每個智慧指標與 callback 綁定在做什麼",
 ],
 "dod": "給一段含裸 new/delete 與共享狀態的 C++，能改寫成 RAII + 智慧指標版本並通過 sanitizer。",
 "res": [
   {"t": "learncpp.com（免費、系統性）", "u": "https://www.learncpp.com/", "k": "read", "h": 10},
   {"t": "C++ Core Guidelines（R 章：資源管理）", "u": "https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines", "k": "read", "h": 2},
 ],
},
{
 "id": "A7-2", "act": "A7", "type": "normal", "track": "main",
 "title": "rclcpp 節點實作", "en": "rclcpp Nodes",
 "x": 1750, "y": 1350, "hours": 5, "deps": ["A7-1", "A1-4"],
 "desc": "把 A1 學過的 pub/sub/service/timer 用 C++ 重寫一遍：rclcpp API、std::bind vs lambda、參數與 QoS 的 C++ 寫法、ament_cmake 編譯。",
 "why": "驅動與控制層的程式碼幾乎都是 C++。用已懂的概念學新語法，是第二次學習曲線最平的路。",
 "tasks": [
   "用 C++ 重寫 A1-1 的 pub/sub 對，跑通",
   "寫一個含 timer + service 的 C++ 節點，用 lambda 綁 callback",
   "在 C++ 節點宣告參數並從 YAML 載入",
   "刻意寫出一個 use-after-free（callback 捕捉懸空參考），用 AddressSanitizer 抓到",
 ],
 "dod": "不抄範本，30 分鐘內從空白寫出一個可編譯、可跑的 C++ pub/sub 節點。",
 "res": [
   {"t": "ROS 2 官方：Writing a simple publisher and subscriber (C++)", "u": "https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A7-3", "act": "A7", "type": "normal", "track": "main",
 "title": "CMake 與 ament_cmake", "en": "Build System",
 "x": 1750, "y": 1610, "hours": 4, "deps": ["A7-1"],
 "desc": "target 導向 CMake（target_link_libraries / target_include_directories）、find_package、ament_cmake 的巨集、依賴傳遞、debug vs release build。",
 "why": "看不懂 CMake 的人連別人的專案都編不起來，更別說修。目標是能讀懂並修改，不是從零寫框架。",
 "tasks": [
   "讀懂一個中型 ROS 2 C++ package 的 CMakeLists.txt，畫出 target 依賴圖",
   "加一個新的執行檔 target 並連結既有函式庫",
   "解釋 ament_target_dependencies 與 target_link_libraries 的差別",
   "用 -DCMAKE_BUILD_TYPE=RelWithDebInfo 編譯，說明它與 Debug/Release 的取捨",
 ],
 "dod": "給任一編譯錯誤（undefined reference / 找不到標頭），能定位是哪一層依賴宣告缺了並修好。",
 "res": [
   {"t": "ament_cmake 使用者文件", "u": "https://docs.ros.org/en/jazzy/How-To-Guides/Ament-CMake-Documentation.html", "k": "read", "h": 2},
 ],
},
{
 "id": "A7-4", "act": "A7", "type": "keystone", "track": "main",
 "title": "C++ 除錯與 sanitizers", "en": "Debugging C++",
 "x": 1750, "y": 1890, "hours": 5, "deps": ["A7-2", "A3-1"],
 "desc": "gdb 實戰（break/backtrace/watch）、core dump 設定與事後解剖、AddressSanitizer / ThreadSanitizer、symbols 與 -g 的部署策略。",
 "why": "本章分水嶺：C++ 的 crash 不會給你 traceback。會讀 core dump 的人修一小時，不會的人猜三天——JD 的 root-cause 精神在 C++ 世界就長這樣。",
 "tasks": [
   "開 core dump（ulimit / coredumpctl），讓一個節點 segfault，事後用 gdb 還原 backtrace",
   "用 ASan 抓出 A7-2 埋的 use-after-free，讀懂報告每一段",
   "用 TSan 抓出一個 data race，解釋為什麼它平常「看起來沒事」",
   "整理「值班撿到 core dump」SOP：從檔案到指認兇手函式的每一步",
 ],
 "dod": "拿到一個陌生程式的 core dump 與 binary，30 分鐘內產出 crash 的函式、行號與初步成因假設。",
 "res": [
   {"t": "coredumpctl man page（systemd 的 core dump 管理）", "u": "https://man7.org/linux/man-pages/man1/coredumpctl.1.html", "k": "hands-on", "h": 1},
   {"t": "Google sanitizers wiki", "u": "https://github.com/google/sanitizers/wiki", "k": "read", "h": 2},
 ],
},

# ─────────────────────────── A8 · 支線：模擬鳥瞰 ───────────────────────────
{
 "id": "A8-1", "act": "A8", "type": "normal", "track": "side",
 "title": "URDF 與 Gazebo 速覽", "en": "URDF & Gazebo",
 "x": 640, "y": 1090, "hours": 6, "deps": ["A1-7"],
 "desc": "壓縮版：URDF/xacro 描述機器人、RViz 視覺化、Gazebo Harmonic 起一個世界跑差速小車。目標是看得懂、跑得動，不深潛。",
 "why": "平台工程師不建模，但要能起模擬環境重現 bug——「在 sim 重現」是最便宜的除錯手段。",
 "tasks": [
   "跑通官方 diff drive 模擬範例，用 teleop 開車",
   "讀懂該範例的 URDF：link/joint/plugin 各在宣告什麼",
   "用 ros_gz_bridge 把 sim topic 接進 ROS 2，Foxglove 看到",
 ],
 "dod": "一條指令起 sim 小車，且能解釋 sim topic 如何流進 ROS 2 世界。",
 "res": [
   {"t": "ROS 2 官方：URDF 教學", "u": "https://docs.ros.org/en/jazzy/Tutorials/Intermediate/URDF/URDF-Main.html", "k": "hands-on", "h": 3},
   {"t": "gz_ros2 整合文件", "u": "https://gazebosim.org/docs/harmonic/ros2_integration/", "k": "hands-on", "h": 2},
 ],
},
{
 "id": "A8-2", "act": "A8", "type": "normal", "track": "side",
 "title": "MoveIt 2 鳥瞰", "en": "MoveIt Overview",
 "x": 640, "y": 1450, "hours": 5, "deps": ["A8-1"],
 "desc": "壓縮版：用官方 panda demo 跑 MoveIt 2，理解 planning scene / move_group 的角色。知道手臂軟體棧長什麼樣即可。",
 "why": "Anvil 的客戶大概率在跑手臂。你不寫規劃器，但 debug 時要認得 move_group 掛掉長什麼樣。",
 "tasks": [
   "跑通 MoveIt 2 官方 demo（RViz 拖目標、規劃、執行）",
   "畫出 move_group 的輸入輸出：誰給它目標、它對誰發軌跡",
   "刻意弄壞一次（改壞 SRDF 或 controller 配置），觀察錯誤長相",
 ],
 "dod": "能在白板畫出手臂軟體棧（感知→規劃→控制→硬體）並標出 MoveIt 管哪段。",
 "res": [
   {"t": "MoveIt 2 官方 tutorials", "u": "https://moveit.picknik.ai/main/index.html", "k": "hands-on", "h": 3},
 ],
},
{
 "id": "A8-3", "act": "A8", "type": "normal", "track": "side",
 "title": "Isaac 生態鳥瞰", "en": "Isaac Landscape",
 "x": 640, "y": 1810, "hours": 4, "deps": ["A8-2"],
 "desc": "地圖級理解：Isaac Sim（高保真模擬）/ Isaac Lab（RL 訓練）/ Isaac ROS（Jetson 加速套件）三者的分工、與 Physical AI 敘事的關係。讀與看為主，不部署。",
 "why": "JD 要你向 leadership 解釋 robot learning 動態。能把 Isaac 三兄弟講到非技術者聽懂，就是那個 internal communication 技能的展演。",
 "tasks": [
   "整理一頁三者分工圖（各自輸入輸出、跑在哪、誰在用）",
   "寫 300 字向非技術主管解釋「sim-to-real 為什麼是 Physical AI 的關鍵」",
   "查證 Isaac ROS 對 Orin Nano 的支援現況（版本分歧），寫兩句結論",
 ],
 "dod": "5 分鐘不看稿向非技術者講完 Isaac 生態與它對機器人公司的意義，對方能複述重點。",
 "res": [
   {"t": "NVIDIA Isaac 平台總覽", "u": "https://developer.nvidia.com/isaac", "k": "read", "h": 2},
 ],
},

# ─────────────────────────── A9 · 終章 ───────────────────────────
{
 "id": "A9-1", "act": "A9", "type": "keystone", "track": "main",
 "title": "★ Platform Lab Capstone", "en": "Always-on Robot Runtime",
 "x": 1200, "y": 3170, "hours": 20, "deps": ["A2-4", "A3-4", "A4-6", "A5-3", "A6-2"],
 "desc": "把全部串起來：Jetson 上一套 always-on 機器人執行時——systemd 管 bringup、lifecycle 節點、CAN 模擬馬達迴路、健康監控三層 watchdog、metrics/log/bag 上雲、Mac 地面站 Foxglove。然後對它跑完整 chaos 演練並連續運行 72 小時。",
 "why": "這就是 JD 第一條 Focus Area 的完整縮影。面試時你不是「學過這些」，而是「運營過一台 72 小時不斷線的機器人」。",
 "tasks": [
   "依 projects/platform-lab 規格搭起全套（bringup、健康、CAN 迴路、觀測管線）",
   "跑 A3-4 的 chaos 清單全套，全部達到偵測與復原目標",
   "連續運行 72 小時：期間注入 3 次故障，全部自動復原，事後用雲端資料還原時間線",
   "寫 architecture.md：一張圖 + 每個元件的選型理由",
   "錄 5 分鐘 demo 影片（或現場演示腳本）：拔感測器 → 告警 → 降級 → 復原",
 ],
 "dod": "72 小時不間斷運行達成，且任一故障都有「告警截圖 + 日誌時間線 + 復原證據」三件套。",
 "res": [
   {"t": "repo 內 projects/platform-lab/（本節點的實作規格書）", "u": "https://github.com/ChesterHsieh/Always_try_to_learn/tree/main/ros2-robotics-mastery/projects/platform-lab", "k": "hands-on", "h": 20},
 ],
},
{
 "id": "A9-2", "act": "A9", "type": "notable", "track": "main",
 "title": "事故復盤與敘事", "en": "Postmortem & Narrative",
 "x": 1000, "y": 3310, "hours": 4, "deps": ["A9-1"],
 "desc": "JD 說這職位一半是溝通。把 capstone 的一次故障寫成兩個版本：工程版 postmortem（時間線、根因、行動項）與主管版摘要（300 字、無術語、講影響與對策）。",
 "why": "『making them smarter is part of the job』——能把 root cause 講到非技術者點頭的工程師，在客戶現場價值翻倍。",
 "tasks": [
   "選 capstone 中最有戲的一次故障，寫工程版 postmortem（blameless、含行動項）",
   "同一事故寫 300 字主管版：影響、原因類比、我們做了什麼、客戶需要做什麼",
   "把主管版唸給一個非工程背景的人聽，對方能正確轉述根因才算過",
 ],
 "dod": "兩版文件完成，且非技術讀者能正確轉述「發生什麼、為什麼、之後怎麼防」。",
 "res": [
   {"t": "Google SRE Book：Postmortem Culture 章節", "u": "https://sre.google/sre-book/postmortem-culture/", "k": "read", "h": 2},
 ],
},
{
 "id": "A9-3", "act": "A9", "type": "notable", "track": "main",
 "title": "面試演武", "en": "Interview Gauntlet",
 "x": 1400, "y": 3310, "hours": 6, "deps": ["A9-2", "A7-4"],
 "desc": "對著 JD 逐條演練：每個 Focus Area 準備一個「我做過的具體例子＋數字」；系統設計題（設計 24/7 機器人監控）；debug 情境題（ログ不全的跨層故障）用 capstone 素材回答。",
 "why": "三個月的努力要在 45 分鐘內被看見。素材你都有了，這節點只是把它們排成子彈上膛。",
 "tasks": [
   "JD 每條 requirement 寫一張卡：STAR 例子 + 一個量化數字",
   "用 Claude 模擬面試 3 輪：系統設計、debug 情境、行為題，錄音復盤",
   "準備 3 個反問面試官的問題（顯示你懂 deployment 現場）",
   "把 capstone repo README 打磨到「面試官點開 3 分鐘就懂你做了什麼」",
 ],
 "dod": "模擬面試中，任一 JD 條目被問到都能在 90 秒內給出具體案例與數字。",
 "res": [
   {"t": "repo 內 skill-tree 與 projects/ 的全部產出", "u": "https://github.com/ChesterHsieh/Always_try_to_learn/tree/main/ros2-robotics-mastery", "k": "read", "h": 1},
 ],
},
]
