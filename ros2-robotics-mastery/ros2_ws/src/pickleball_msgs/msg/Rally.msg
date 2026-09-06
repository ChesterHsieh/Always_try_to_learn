# 一個回合（一分）的完整記錄。
#
# 設計註記：這裡內嵌完整的 ShotEvent[]，一個長回合約數十拍，
# 單一 Rally 訊息仍在數 KB 等級，可接受。
# 但 MatchState 內若再內嵌 Rally[] 會平方成長，故 MatchState 只帶 rally_id。

uint8 ENDED_UNKNOWN=0
uint8 ENDED_WINNER=1
uint8 ENDED_UNFORCED_ERROR=2
uint8 ENDED_FORCED_ERROR=3

uint32 rally_id
builtin_interfaces/Time start
builtin_interfaces/Time end

ShotEvent[] shots

uint8 winner_player_id
uint8 how_ended                 # ENDED_*
uint8 ending_shot_result        # 對應 ShotEvent.RESULT_*
