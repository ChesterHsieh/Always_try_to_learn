# 單一影格中的球偵測結果，像素座標系。
# 由 ball_detector 發布；cv 與 trt 兩種 plugin 共用此型別，
# 讓「換偵測器」成為可量化實驗（README 第 1 節決策）。

uint8 SOURCE_CV=0
uint8 SOURCE_DNN=1

std_msgs/Header header      # frame_id = camera

float32 u                   # 像素座標 x
float32 v                   # 像素座標 y
float32 radius_px           # 偵測到的球半徑（像素）
float32 confidence          # 0.0 ~ 1.0
uint8 source                # SOURCE_CV / SOURCE_DNN
