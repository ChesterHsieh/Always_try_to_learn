# 經 Kalman 濾波後的球體軌跡狀態，court 座標系。
# Phase 0 只有 u/v 反投影，z 固定為 0；Phase 1 起才有真正的高度。

uint8 SPIN_UNKNOWN=0
uint8 SPIN_TOPSPIN=1
uint8 SPIN_SLICE=2
uint8 SPIN_FLAT=3

std_msgs/Header header          # frame_id = court

uint32 track_id
geometry_msgs/Point position    # court 座標 (x, y, z)，公尺
geometry_msgs/Vector3 velocity  # m/s
float32 speed_mps
bool is_predicted               # true = 本筆為遺失補插，非實際偵測

uint8 spin_type                 # SwingVision 官方未證實，保留欄位
bool spin_valid
