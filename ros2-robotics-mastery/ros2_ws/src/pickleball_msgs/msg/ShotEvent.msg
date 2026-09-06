# 一次擊球，是分析層的核心單位。
# 由 shot_engine 從 BallTrack 的加速度突變偵測擊球時刻，
# 再由 rally_engine 補上 stroke_type 與 player_id。

uint8 STROKE_UNKNOWN=0
uint8 STROKE_SERVE=1
uint8 STROKE_RETURN=2
uint8 STROKE_SERVE_PLUS_ONE=3
uint8 STROKE_RETURN_PLUS_ONE=4
uint8 STROKE_THIRD_SHOT_DROP=5
uint8 STROKE_DINK=6
uint8 STROKE_DRIVE=7
uint8 STROKE_VOLLEY=8
uint8 STROKE_LOB=9
uint8 STROKE_OVERHEAD=10

uint8 SERVE_SIDE_NA=0
uint8 SERVE_SIDE_DEUCE=1
uint8 SERVE_SIDE_AD=2

uint8 RESULT_UNKNOWN=0
uint8 RESULT_IN=1
uint8 RESULT_OUT_LONG=2
uint8 RESULT_OUT_WIDE=3
uint8 RESULT_NET=4

uint8 DIR_MIDDLE=0
uint8 DIR_CROSS_COURT=1
uint8 DIR_DOWN_THE_LINE=2

std_msgs/Header header          # frame_id = court

uint32 shot_id
uint32 rally_id
uint8 player_id

uint8 stroke_type               # STROKE_*
uint8 serve_side                # SERVE_SIDE_*
uint8 result                    # RESULT_*

geometry_msgs/Point hit_point   # 3D 擊球點，court 座標
BounceEvent bounce              # 這一拍造成的落點（若有）
float32 speed_mps
uint8 direction                 # DIR_*
uint8 depth_zone                # 對應 BounceEvent.ZONE_*

float32 net_clearance_m         # SwingVision 官方未證實，保留欄位
bool net_clearance_valid
