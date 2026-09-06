// ball_tracker — BallDetection → BallTrack
//
// 資料流：
//   ~/detections (BallDetection, camera frame)
//     → BallFilter（等加速度 Kalman，見 ball_filter.hpp）
//     → coast timer：偵測器沉默時仍以 fps 節奏輸出預測值
//     ── publish ──► ~/tracks (BallTrack, court frame)
//
// 為什麼這樣切 callback group：
//   訂閱 callback 與 coast timer 都會碰同一個 BallFilter 實例。
//   兩者刻意放在同一個預設 MutuallyExclusive callback group，
//   由 executor 保證互斥，因此 filter_ 不需要自己的 mutex。
//   這是 A1-4 Executor 與 Callback Group 的實例：
//   「用 callback group 表達互斥」比「到處加鎖」更容易推理。
//
// Phase 0 的座標系妥協：
//   還沒有 homography，position 直接放像素值、z=0，frame_id 仍標 camera。
//   Phase 1 court_calibrator 上線後改成真正的 court 座標，
//   屆時 frame_id 才會變成 court，下游不需改程式。

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <pickleball_msgs/msg/ball_detection.hpp>
#include <pickleball_msgs/msg/ball_track.hpp>

#include "pickleball_perception/ball_filter.hpp"

namespace pickleball_perception
{

class BallTrackerNode : public rclcpp::Node
{
public:
  explicit BallTrackerNode(const rclcpp::NodeOptions & options)
  : Node("ball_tracker", options)
  {
    BallFilterParams p;
    p.process_noise = declare_parameter<double>("process_noise", p.process_noise);
    p.measurement_noise = declare_parameter<double>("measurement_noise", p.measurement_noise);
    p.initial_covariance = declare_parameter<double>("initial_covariance", p.initial_covariance);
    p.outlier_gate_px = declare_parameter<double>("outlier_gate_px", p.outlier_gate_px);
    p.max_coast_frames = static_cast<std::size_t>(
      declare_parameter<int>("max_coast_frames", static_cast<int>(p.max_coast_frames)));

    fps_ = declare_parameter<double>("expected_fps", 30.0);
    // Phase 0 尚無 homography，這個係數只是讓 RViz 上的軌跡大小合理。
    // Phase 1 會被真正的 court 座標取代。
    px_to_m_ = declare_parameter<double>("pixels_to_meters", 0.01);
    frame_id_ = declare_parameter<std::string>("output_frame_id", "camera");

    filter_ = BallFilter(p);

    track_pub_ = create_publisher<pickleball_msgs::msg::BallTrack>(
      "~/tracks", rclcpp::SensorDataQoS());

    det_sub_ = create_subscription<pickleball_msgs::msg::BallDetection>(
      "detections", rclcpp::SensorDataQoS(),
      std::bind(&BallTrackerNode::onDetection, this, std::placeholders::_1));

    // 偵測器沉默時，仍以影格節奏推進濾波器並輸出預測值，
    // 讓下游拿到連續軌跡（README：遺失補插）。
    coast_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / fps_),
      std::bind(&BallTrackerNode::onCoastTick, this));

    RCLCPP_INFO(get_logger(), "ball_tracker 就緒（fps=%.1f）", fps_);
  }

private:
  void onDetection(const pickleball_msgs::msg::BallDetection::ConstSharedPtr & msg)
  {
    const rclcpp::Time stamp(msg->header.stamp);
    const double dt = computeDt(stamp);

    const bool accepted = filter_.update({msg->u, msg->v}, dt);
    if (!accepted) {
      RCLCPP_DEBUG(get_logger(), "離群觀測被閘門擋下 (u=%.1f v=%.1f)", msg->u, msg->v);
    }

    last_stamp_ = stamp;
    have_last_stamp_ = true;
    frames_since_detection_ = 0;

    if (filter_.hasTrack()) {
      publishTrack(msg->header.stamp);
    }
  }

  void onCoastTick()
  {
    if (!filter_.hasTrack()) {
      return;
    }
    // 剛收到偵測的那一格不要重複推進，避免 dt 被算兩次。
    if (frames_since_detection_ == 0) {
      ++frames_since_detection_;
      return;
    }

    filter_.predictOnly(1.0 / fps_);
    if (!filter_.hasTrack()) {
      RCLCPP_INFO(get_logger(), "軌跡遺失超過上限，已放棄");
      have_last_stamp_ = false;
      return;
    }
    publishTrack(now());
  }

  double computeDt(const rclcpp::Time & stamp)
  {
    if (!have_last_stamp_) {
      return 1.0 / fps_;
    }
    const double dt = (stamp - last_stamp_).seconds();
    // 影片重播 seek 或時間戳亂跳時，退回名目影格間隔。
    if (dt <= 0.0 || dt > 1.0) {
      return 1.0 / fps_;
    }
    return dt;
  }

  void publishTrack(const rclcpp::Time & stamp)
  {
    const auto & s = filter_.state();

    pickleball_msgs::msg::BallTrack t;
    t.header.stamp = stamp;
    t.header.frame_id = frame_id_;
    t.track_id = filter_.trackId();
    t.position.x = s.px * px_to_m_;
    t.position.y = s.py * px_to_m_;
    t.position.z = 0.0;                    // Phase 1 才有真正的高度
    t.velocity.x = s.vx * px_to_m_;
    t.velocity.y = s.vy * px_to_m_;
    t.velocity.z = 0.0;
    t.speed_mps = static_cast<float>(s.speed() * px_to_m_);
    t.is_predicted = s.is_predicted;
    t.spin_type = pickleball_msgs::msg::BallTrack::SPIN_UNKNOWN;
    t.spin_valid = false;

    track_pub_->publish(t);
  }

  BallFilter filter_;
  double fps_{30.0};
  double px_to_m_{0.01};
  std::string frame_id_{"camera"};

  rclcpp::Time last_stamp_;
  bool have_last_stamp_{false};
  std::size_t frames_since_detection_{0};

  rclcpp::Subscription<pickleball_msgs::msg::BallDetection>::SharedPtr det_sub_;
  rclcpp::Publisher<pickleball_msgs::msg::BallTrack>::SharedPtr track_pub_;
  rclcpp::TimerBase::SharedPtr coast_timer_;
};

}  // namespace pickleball_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<pickleball_perception::BallTrackerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
