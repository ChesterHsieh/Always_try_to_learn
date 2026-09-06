// ball_detector — sensor_msgs/Image → pickleball_msgs/BallDetection
//
// 資料流：
//   /camera/image_raw
//     → ROI 遮罩（挖掉記分板與贊助看板，見 README 第 7 節）
//     → MOG2 背景相減 → 形態學開閉運算去雜訊
//     → findContours → 依面積/圓形度/長寬比篩選
//     → 取分數最高的候選 ── publish ──► ~/detections (BallDetection)
//                          └─ publish ──► ~/debug_image (可關)
//
// 為什麼這樣切 callback group：
//   單一影像訂閱、單一 publisher，且 MOG2 的背景模型是有狀態的、非 thread-safe。
//   因此刻意留在預設的 MutuallyExclusive group，保證影格串列處理。
//   若改成 Reentrant，兩張影格並行更新同一個背景模型會讓模型損壞。
//
// Phase 3 會加入 trt plugin：兩者輸出同一個 BallDetection 型別，
// 用 source 欄位區分，讓 cv vs trt 成為可量化實驗（README 第 6 節）。

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// cv_bridge 的標頭在 Jazzy 改名為 .hpp，Humble 只有 .h。
// 用 __has_include 讓同一份程式碼在兩個發行版都能編（README 第 1 節：雙版本可編譯）。
#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <pickleball_msgs/msg/ball_detection.hpp>

namespace pickleball_perception
{

class BallDetectorNode : public rclcpp::Node
{
public:
  explicit BallDetectorNode(const rclcpp::NodeOptions & options)
  : Node("ball_detector", options)
  {
    min_area_ = declare_parameter<double>("min_area_px", 6.0);
    max_area_ = declare_parameter<double>("max_area_px", 600.0);
    min_circularity_ = declare_parameter<double>("min_circularity", 0.55);
    max_aspect_ratio_ = declare_parameter<double>("max_aspect_ratio", 1.8);
    publish_debug_ = declare_parameter<bool>("publish_debug_image", true);

    // ROI 遮罩：[x, y, w, h] 四個一組，這些矩形會被塗黑。
    // 預設值對應素材右上角的轉播比分板（README 第 7 節）。
    mask_rects_ = declare_parameter<std::vector<int64_t>>(
      "mask_rects", std::vector<int64_t>{890, 0, 390, 125});
    if (mask_rects_.size() % 4 != 0) {
      RCLCPP_ERROR(get_logger(), "mask_rects 長度必須是 4 的倍數，已忽略");
      mask_rects_.clear();
    }

    const int history = static_cast<int>(declare_parameter<int>("mog2_history", 300));
    const double var_threshold = declare_parameter<double>("mog2_var_threshold", 24.0);
    bg_ = cv::createBackgroundSubtractorMOG2(history, var_threshold, /*detectShadows=*/false);

    det_pub_ = create_publisher<pickleball_msgs::msg::BallDetection>(
      "~/detections", rclcpp::SensorDataQoS());
    if (publish_debug_) {
      debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "~/debug_image", rclcpp::SensorDataQoS());
    }

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image_raw", rclcpp::SensorDataQoS(),
      std::bind(&BallDetectorNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "ball_detector 就緒（cv plugin）");
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv::Mat frame;
    try {
      frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge 轉換失敗: %s", e.what());
      return;
    }
    if (frame.empty()) {
      return;
    }

    // 記分板/看板一定要先遮掉，否則背景相減會把它們當成移動物體。
    cv::Mat masked = frame.clone();
    for (std::size_t i = 0; i + 3 < mask_rects_.size(); i += 4) {
      const cv::Rect r(
        static_cast<int>(mask_rects_[i]), static_cast<int>(mask_rects_[i + 1]),
        static_cast<int>(mask_rects_[i + 2]), static_cast<int>(mask_rects_[i + 3]));
      const cv::Rect clipped = r & cv::Rect(0, 0, masked.cols, masked.rows);
      if (clipped.area() > 0) {
        masked(clipped).setTo(cv::Scalar(0, 0, 0));
      }
    }

    cv::Mat fg;
    bg_->apply(masked, fg);

    // 開運算去掉單點雜訊，閉運算把球體斷裂的輪廓補回來。
    static const cv::Mat kernel =
      cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(fg, fg, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool found = false;
    double best_score = 0.0;
    cv::Point2f best_center;
    float best_radius = 0.0f;

    for (const auto & c : contours) {
      const double area = cv::contourArea(c);
      if (area < min_area_ || area > max_area_) {
        continue;
      }
      const double perimeter = cv::arcLength(c, true);
      if (perimeter <= 1e-6) {
        continue;
      }
      // 圓形度 = 4πA/P²，完美圓為 1.0。球是圓的，球員的肢體不是。
      const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
      if (circularity < min_circularity_) {
        continue;
      }
      const cv::Rect bbox = cv::boundingRect(c);
      const double aspect = static_cast<double>(std::max(bbox.width, bbox.height)) /
        std::max(1, std::min(bbox.width, bbox.height));
      if (aspect > max_aspect_ratio_) {
        continue;
      }

      cv::Point2f center;
      float radius = 0.0f;
      cv::minEnclosingCircle(c, center, radius);

      if (circularity > best_score) {
        best_score = circularity;
        best_center = center;
        best_radius = radius;
        found = true;
      }
    }

    if (found) {
      pickleball_msgs::msg::BallDetection det;
      det.header = msg->header;
      det.u = best_center.x;
      det.v = best_center.y;
      det.radius_px = best_radius;
      // Phase 0 用圓形度當信心值的代理；Phase 3 換 DNN 後才是真的機率。
      det.confidence = static_cast<float>(std::min(1.0, best_score));
      det.source = pickleball_msgs::msg::BallDetection::SOURCE_CV;
      det_pub_->publish(det);
    }

    if (publish_debug_ && debug_pub_) {
      cv::Mat dbg;
      cv::cvtColor(fg, dbg, cv::COLOR_GRAY2BGR);
      if (found) {
        cv::circle(dbg, best_center, static_cast<int>(best_radius) + 4,
          cv::Scalar(0, 255, 0), 2);
      }
      debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", dbg).toImageMsg());
    }
  }

  double min_area_{6.0};
  double max_area_{600.0};
  double min_circularity_{0.55};
  double max_aspect_ratio_{1.8};
  bool publish_debug_{true};
  std::vector<int64_t> mask_rects_;

  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<pickleball_msgs::msg::BallDetection>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
};

}  // namespace pickleball_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<pickleball_perception::BallDetectorNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
