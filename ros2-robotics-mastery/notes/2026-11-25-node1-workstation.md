# 節點 1「工作站落地」完成筆記（2026-11-25）

## 環境
- Mac（Apple Silicon）＋ Rancher Desktop（關掉 Kubernetes 省資源）
- 用 `docker/compose.yaml` 起容器，ROS 2 Jazzy
- 容器內 `uname -m` 顯示 `aarch64` → 原生 ARM，沒有走模擬，速度正常

## 驗證過的事
- **Pub/Sub 通了**：容器內開兩個 shell，分別跑 `ros2 run demo_nodes_cpp talker` 和 `listener`，`/chatter` 上訊息正常收發。
- `ros2 topic info /chatter -v` 可以看到誰在發、誰在收，以及各自的 QoS 設定。
- **Bind mount**（`ros2_ws` 掛進容器）：
  - Mac 上新增的檔案，容器裡立刻看得到。
  - 是**雙向**的：在容器裡刪檔，Mac 上也跟著不見。
  - 想要唯讀就在掛載後面加 `:ro`。

## 學到的觀念
- **DDS（Data Distribution Service）**：沒有中央 broker 的 pub/sub。節點先用 multicast 互相「發現」，之後直接點對點（peer-to-peer）傳資料。
- **QoS**：主要看三個
  - reliability：`RELIABLE`（保證送達）vs `BEST_EFFORT`（丟了就算）
  - durability：晚加入的訂閱者能不能拿到之前的訊息
  - history / depth：保留幾筆
  - ⚠️ 發布端和訂閱端 QoS 不相容時，**不會報錯，就是默默連不上**。
- **Unicast vs multicast 是「探索（discovery）」設定，不是 QoS**。Jazzy 可以用 `ROS_STATIC_PEERS` 直接列出對方 IP。

## 為什麼 Docker 網路會讓 DDS 找不到彼此
- Docker bridge 網路是隔離的虛擬子網＋NAT，multicast 過不去 → discovery 失敗。
- 在 Mac 上 Docker 其實跑在一台 Linux VM 裡，所以 `network_mode: host` 拿到的是 **VM 的網路**，不是 Mac 的。
  - 同一台 VM 裡的容器彼此看得到；
  - 但 Mac 本機或區網上的設備（例如之後的 Jetson）靠 multicast 看不到它。
- 跨機器的解法：
  1. 改用 unicast 指定對象（`ROS_STATIC_PEERS` 或 CycloneDDS XML）
  2. FastDDS discovery server（集中式探索）
  3. `rmw_zenoh`（走 TCP＋router，天生適合跨網段）

## 用白話跟老闆解釋
機器人裡的各個程式是靠「在區網裡大聲喊一聲、互相認識」來接上線的，但 Mac 上的 Docker 被包在一台虛擬機裡，喊的聲音傳不出去。所以 Mac 上開發沒問題，可是要跟真的機器人（Jetson）連線時，必須改成「直接給對方地址」或「設一個總機」的方式，這是部署前一定要先規劃好的網路設定。
