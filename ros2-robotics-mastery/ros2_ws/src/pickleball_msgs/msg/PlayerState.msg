# 單一球員在某時刻的位置與場區。
# 只做位置，不做姿態骨架（README 第 1 節決策：Jetson 8 GB 養不起第二個大模型）。

uint8 ZONE_BEHIND_BASELINE=0
uint8 ZONE_BASELINE_TO_KITCHEN=1
uint8 ZONE_KITCHEN_LINE=2

std_msgs/Header header          # frame_id = court

uint8 player_id
geometry_msgs/Point position    # court 座標
uint8 court_zone                # ZONE_*

float32 distance_run_m          # 累積跑動距離
bool distance_valid
