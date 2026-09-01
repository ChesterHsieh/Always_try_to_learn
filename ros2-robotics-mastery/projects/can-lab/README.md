# CAN Lab — SocketCAN 從 vcan 到 Jetson 實體匯流排（A6-1 / A6-2 規格書）

> 對應技能樹節點：**A6-1 SocketCAN 基礎（keystone）**、A6-2 Jetson CAN 實戰、A6-5 乾淨硬體 API。
> 原則：**軟體問題在 vcan 上修完，上真硬體只剩電氣問題。**

## Phase 1 · vcan（零硬體成本，任何 Linux 容器可跑）

```bash
# 建虛擬介面（容器需 --cap-add NET_ADMIN）
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# 一邊看、一邊發
candump vcan0
cansend vcan0 123#DEADBEEF
```

任務（對照 A6-1 tasks）：

1. `python-can` 寫「模擬馬達控制器」：收 `0x100` 指令框 → 回 `0x180` 狀態框（位置/速度/錯誤碼）
2. `cangen vcan0 -g 1` 灌流量、`canbusload vcan0@500000` 看負載
3. 拿公開 DBC 檔用 `cantools` 解碼一段 log
4. 寫 `can_bridge` ROS 2 節點：狀態框 → 語意化 topic（**不要**原樣轉發 raw frame——見 A6-2 題庫）

## Phase 2 · Jetson Orin Nano 實體 CAN

### 購物清單（百元等級）

| 項目 | 說明 |
|---|---|
| CAN transceiver ×1–2 | SN65HVD230 模組（3.3V，Orin 相容）。買兩顆可做雙節點真通訊 |
| 120Ω 電阻 ×2 | 終端電阻（多數 SN65HVD230 模組已內建，確認再買） |
| 杜邦線 | header 接線用 |

⚠️ Orin Nano **內建 CAN controller**，40-pin header 出的是 **邏輯電平 TX/RX**，必須經 transceiver 轉 CAN_H/CAN_L 差動訊號才能上匯流排。

### 啟用步驟（checklist）

1. `sudo /opt/nvidia/jetson-io/jetson-io.py` 設定 pinmux 啟用 CAN（腳位見 NVIDIA CAN 文件，JetPack 版本不同腳位表可能不同——以官方文件為準）
2. 載入模組：`sudo modprobe can && sudo modprobe mttcan`
3. 起介面：`sudo ip link set can0 up type can bitrate 500000`
4. 無第二節點時先 loopback 自測：`... type can bitrate 500000 loopback on`
5. `candump can0` 收到框 → 電氣層通了
6. 把 Phase 1 的 `can_bridge` 從 `vcan0` 改參數指到 `can0`（程式碼零修改才算過 A6-4 的抽象驗收）

### 除錯速查

| 症狀 | 第一嫌犯 |
|---|---|
| candump 一片死寂 | 兩端 bitrate 不一致／缺 120Ω 終端電阻／沒共地 |
| error frame 暴增後介面沉默 | bus-off（錯誤計數超標自我隔離）→ 查電氣，`ip link set can0 type can restart-ms 100` |
| ip link 報 No such device | pinmux 沒設或 mttcan 模組沒載 |
| 收得到但資料亂 | DBC 對不上／byte order（Motorola vs Intel）搞反 |

## 產出

- `sim_motor.py` — vcan 模擬馬達控制器
- `can_bridge/` — ROS 2 橋接 package（lifecycle + diagnostic + 三級錯誤語意，A6-5 驗收）
- `jetson-can-checklist.md` — 你實際走過一遍後修訂的啟用 checklist（含踩雷紀錄）
