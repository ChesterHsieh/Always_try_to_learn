# 球的一次落地（或觸網）事件。
# shot_engine 從 BallTrack 的 z 最低點 + 垂直速度反向偵測。

uint8 IN=0
uint8 OUT=1
uint8 UNCERTAIN=2

uint8 SIDE_NEAR=0
uint8 SIDE_FAR=1

uint8 ZONE_KITCHEN=0
uint8 ZONE_MID=1
uint8 ZONE_DEEP=2
uint8 ZONE_BEYOND_BASELINE=3

std_msgs/Header header          # frame_id = court

uint32 track_id
uint32 rally_id
geometry_msgs/Point position    # 落點，court 座標

uint8 in_out                    # IN / OUT / UNCERTAIN
float32 in_out_confidence       # 進 MatchState.score_confidence 的乘積
uint8 side                      # SIDE_NEAR / SIDE_FAR
uint8 depth_zone                # ZONE_*
bool is_net_contact
