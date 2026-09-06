# 單一球員的累積統計，由 stats_node 發布。
# 直方圖與熱圖都用扁平陣列，索引語意見各欄位註解。

uint8 HEATMAP_COLS=6            # 球場橫向切 6 格
uint8 HEATMAP_ROWS=11           # 球場縱向切 11 格

uint8 player_id

uint32 shots_total
uint32 shots_in
float32 pct_in
float32 avg_speed_mps
float32 max_speed_mps

uint32[] stroke_histogram       # 依 ShotEvent.STROKE_* 索引
uint32[] depth_histogram        # 依 BounceEvent.ZONE_* 索引
uint32[] placement_heatmap      # 落點計數，長度 = ROWS*COLS，row-major
uint32[] position_heatmap       # 擊球位置計數，同上

uint32 rallies_won
uint32 rallies_lost
uint32 winners
uint32 unforced_errors
float32 avg_rally_length
float32 longest_rally
