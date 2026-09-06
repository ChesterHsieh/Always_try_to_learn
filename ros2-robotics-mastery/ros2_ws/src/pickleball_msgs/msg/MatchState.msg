# 比賽當前狀態，由 rally_engine 的 AI 計分狀態機發布。
# README 第 1 節決策：只做全自動 AI 計分，不留手動 service。

uint8 SCORING_SIDE_OUT=0
uint8 SCORING_RALLY=1

std_msgs/Header header

uint8 scoring_mode              # SCORING_*
uint8[] score                   # [server_team, receiver_team, server_number]，side-out 從 0-0-2 起

uint8 serving_team
uint8 serving_player
uint8 serving_side              # 對應 ShotEvent.SERVE_SIDE_*

# 只帶 id 不內嵌 Rally，避免訊息隨比賽進行平方成長。
# 完整回合內容請訂閱 /rally topic 或從 rosbag 取。
uint32[] point_history

float32 score_confidence        # 該局所有關鍵 in_out_confidence 的乘積
