// camera_node — 影片檔 / 串流 → sensor_msgs/Image + CameraInfo
//
// 資料流：
//   cv::VideoCapture(video_path | gstreamer pipeline)
//     → cv::Mat (BGR8)
//     → cv_bridge → sensor_msgs/Image  ── publish ──►  /camera/image_raw
//                   sensor_msgs/CameraInfo ─ publish ─►  /camera/camera_info
//
// 為什麼這樣切 callback group：
//   本節點只有一個 wall timer，沒有訂閱者，因此不需要額外的 callback group。
//   預設的 MutuallyExclusive group 已足夠，且能保證 read → publish 不會重入
//   （VideoCapture 不是 thread-safe）。等 Phase 3 加入 Challenge service 時，
//   才需要把 service 放進獨立的 Reentrant group 避免擋住影格輸出。
//
// 時間戳策略（README：use_sim_time 下用影片時間戳）：
//   publish_video_time=true 時，header.stamp = 影片起始時間 + 影片內 PTS，
//   讓 rosbag 重播與離線分析的時間軸和影片一致。

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

// cv_bridge 的標頭在 Jazzy 改名為 .hpp，Humble 只有 .h。
// 用 __has_include 讓同一份程式碼在兩個發行版都能編（README 第 1 節：雙版本可編譯）。
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace pickleball_perception
{

class CameraNode : public rclcpp::Node
{
public:
  explicit CameraNode(const rclcpp::NodeOptions & options)
  : Node("camera_node", options)
  {
    video_path_ = declare_parameter<std::string>("video_path", "");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera");
    loop_ = declare_parameter<bool>("loop", false);
    publish_video_time_ = declare_parameter<bool>("publish_video_time", true);
    const double rate_override = declare_parameter<double>("publish_rate", 0.0);

    if (video_path_.empty()) {
      RCLCPP_FATAL(get_logger(), "必須設定 video_path 參數");
      throw std::runtime_error("video_path is empty");
    }

    if (!cap_.open(video_path_)) {
      RCLCPP_FATAL(get_logger(), "無法開啟影片來源: %s", video_path_.c_str());
      throw std::runtime_error("cannot open video source");
    }

    // 影片的原生 fps；串流來源可能回報 0，此時退回 30。
    double fps = cap_.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0 || std::isnan(fps)) {
      RCLCPP_WARN(get_logger(), "來源未回報 fps，改用 30.0");
      fps = 30.0;
    }
    fps_ = (rate_override > 0.0) ? rate_override : fps;

    const auto qos = rclcpp::SensorDataQoS();
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("~/image_raw", qos);
    info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("~/camera_info", qos);

    start_time_ = now();
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / fps_),
      std::bind(&CameraNode::tick, this));

    RCLCPP_INFO(
      get_logger(), "camera_node 就緒：source=%s fps=%.2f frame_id=%s",
      video_path_.c_str(), fps_, frame_id_.c_str());
  }

private:
  void tick()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (loop_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        frame_index_ = 0;
        return;
      }
      RCLCPP_INFO(get_logger(), "影片播放結束，共 %lu 幀", frame_index_);
      timer_->cancel();
      return;
    }

    // 影片內的呈現時間戳（毫秒）。以影片時間為準可讓重播結果可重現。
    rclcpp::Time stamp = now();
    if (publish_video_time_) {
      const double pts_ms = cap_.get(cv::CAP_PROP_POS_MSEC);
      const double offset = (pts_ms > 0.0) ? pts_ms / 1000.0 :
        static_cast<double>(frame_index_) / fps_;
      stamp = start_time_ + rclcpp::Duration::from_seconds(offset);
    }

    std_msgs::msg::Header header;
    header.stamp = stamp;
    header.frame_id = frame_id_;

    auto image = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
    image_pub_->publish(*image);
    info_pub_->publish(makeCameraInfo(header, frame.cols, frame.rows));

    ++frame_index_;
  }

  /// Phase 0 還沒做內參標定，先發布單位矩陣占位，
  /// 讓下游拿得到解析度。Phase 1 的 court_calibrator 會取代這裡。
  sensor_msgs::msg::CameraInfo makeCameraInfo(
    const std_msgs::msg::Header & header, int width, int height) const
  {
    sensor_msgs::msg::CameraInfo info;
    info.header = header;
    info.width = static_cast<uint32_t>(width);
    info.height = static_cast<uint32_t>(height);
    info.distortion_model = "plumb_bob";
    info.d.assign(5, 0.0);
    info.k = {1.0, 0.0, static_cast<double>(width) / 2.0,
      0.0, 1.0, static_cast<double>(height) / 2.0,
      0.0, 0.0, 1.0};
    info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    info.p = {1.0, 0.0, static_cast<double>(width) / 2.0, 0.0,
      0.0, 1.0, static_cast<double>(height) / 2.0, 0.0,
      0.0, 0.0, 1.0, 0.0};
    return info;
  }

  cv::VideoCapture cap_;
  std::string video_path_;
  std::string frame_id_;
  bool loop_{false};
  bool publish_video_time_{true};
  double fps_{30.0};
  std::size_t frame_index_{0};
  rclcpp::Time start_time_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pickleball_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(
      std::make_shared<pickleball_perception::CameraNode>(rclcpp::NodeOptions()));
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_node"), "啟動失敗: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
