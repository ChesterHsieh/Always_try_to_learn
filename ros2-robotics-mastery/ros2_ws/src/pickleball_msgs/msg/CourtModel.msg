# 球場幾何模型，由 court_calibrator 發布（latched / transient_local）。

geometry_msgs/Point[4] corners_court   # 球場四角在 court 座標的位置
float32[9] homography                  # image -> court 的 3x3，row-major
float32 net_height_m
