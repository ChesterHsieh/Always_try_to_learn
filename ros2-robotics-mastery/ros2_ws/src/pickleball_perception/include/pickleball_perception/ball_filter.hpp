// 球體追蹤的核心數學：等加速度模型的 Kalman 濾波。
//
// 刻意不依賴 rclcpp 與 OpenCV，讓它能在單元測試中直接被驗證
// （對應 README 第 8 節「補償條款」第 1 點：測行為，不是測覆蓋率）。
//
// 狀態向量 x = [px, py, vx, vy, ax, ay]^T，單位為像素與秒。
// Phase 0 在影像平面上濾波；Phase 1 加入 homography 後改在 court 平面。
#ifndef PICKLEBALL_PERCEPTION__BALL_FILTER_HPP_
#define PICKLEBALL_PERCEPTION__BALL_FILTER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace pickleball_perception
{

/// 一次觀測：影像平面上的球心。
struct Observation
{
  double u{0.0};
  double v{0.0};
};

/// 濾波後的球體狀態。
struct BallState
{
  double px{0.0};
  double py{0.0};
  double vx{0.0};
  double vy{0.0};
  double ax{0.0};
  double ay{0.0};

  /// 本筆是否為純預測（該影格沒有觀測）。
  bool is_predicted{false};

  /// 速度大小，像素/秒。
  [[nodiscard]] double speed() const;
};

/// 濾波器參數。全部可由 ROS param 覆寫，預設值適用 720p @ 30fps。
struct BallFilterParams
{
  /// 過程雜訊：加速度的不確定性 (px/s^2)。調大 = 更信任觀測。
  double process_noise{800.0};

  /// 觀測雜訊：偵測器的像素誤差標準差。調大 = 更信任模型。
  double measurement_noise{4.0};

  /// 初始狀態共變異數，代表「第一筆觀測前我幾乎一無所知」。
  double initial_covariance{500.0};

  /// 連續遺失超過這個影格數就放棄該軌跡。
  std::size_t max_coast_frames{8};

  /// 觀測與預測的距離超過這個像素值視為離群值而丟棄，
  /// 避免背景相減把球員或記分板誤判成球時把軌跡拉走。
  double outlier_gate_px{160.0};
};

/// 等加速度 Kalman 濾波器，單一軌跡。
///
/// 用法：
///   BallFilter f;
///   f.update(obs, dt);            // 有偵測到球
///   f.predictOnly(dt);            // 該影格沒偵測到
///   if (f.hasTrack()) { ... f.state() ... }
class BallFilter
{
public:
  BallFilter() = default;
  explicit BallFilter(const BallFilterParams & params);

  /// 餵入一筆觀測。dt 為距上一次更新的秒數（第一筆可傳任意值）。
  /// 回傳 false 表示該觀測被離群值閘門擋掉，濾波器改走純預測。
  bool update(const Observation & obs, double dt);

  /// 該影格沒有觀測，只做預測。連續呼叫超過 max_coast_frames 會使軌跡失效。
  void predictOnly(double dt);

  /// 目前是否有有效軌跡。
  [[nodiscard]] bool hasTrack() const {return has_track_;}

  /// 目前狀態；沒有軌跡時內容無意義，呼叫前請先檢查 hasTrack()。
  [[nodiscard]] const BallState & state() const {return state_;}

  /// 連續未收到觀測的影格數。
  [[nodiscard]] std::size_t coastFrames() const {return coast_frames_;}

  /// 軌跡 id，每次重新起始會遞增。
  [[nodiscard]] std::uint32_t trackId() const {return track_id_;}

  /// 清空軌跡，下一筆觀測會開新的 track_id。
  void reset();

private:
  static constexpr std::size_t kDim = 6;

  void predictStep(double dt);
  void startTrack(const Observation & obs);

  BallFilterParams params_{};
  BallState state_{};
  bool has_track_{false};
  std::size_t coast_frames_{0};
  std::uint32_t track_id_{0};

  /// 狀態共變異數 P，row-major 6x6。
  std::array<double, kDim * kDim> P_{};
};

}  // namespace pickleball_perception

#endif  // PICKLEBALL_PERCEPTION__BALL_FILTER_HPP_
