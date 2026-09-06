// viz_node — BallTrack → visualization_msgs/MarkerArray
//
// 資料流：
//   ~/tracks (BallTrack)
//     → 環形緩衝保留最近 N 點
//     ── publish ──► ~/markers：LINE_STRIP（軌跡）＋ SPHERE（目前球位）
//
// 為什麼這樣切 callback group：
//   只有一個訂閱者且 trail_ 是共享狀態，維持預設 MutuallyExclusive group
//   即可保證不會有兩個 callback 同時改 trail_，不需要額外的鎖。
//
// Phase 1 會再加球場線與落點 Marker；Phase 0 先確認端到端資料流會動。

#include <deque>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pickleball_msgs/msg/ball_track.hpp>

namespace pickleball_viz
{

class VizNode : public rclcpp::Node
{
public:
  explicit VizNode(const rclcpp::NodeOptions & options)
  : Node("viz_node", options)
  {
    trail_length_ = static_cast<std::size_t>(declare_parameter<int>("trail_length", 90));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/markers", 10);
    track_sub_ = create_subscription<pickleball_msgs::msg::BallTrack>(
      "tracks", rclcpp::SensorDataQoS(),
      std::bind(&VizNode::onTrack, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "viz_node 就緒（trail=%zu）", trail_length_);
  }

private:
  void onTrack(const pickleball_msgs::msg::BallTrack::ConstSharedPtr & msg)
  {
    // 換軌跡就清空，避免把兩段不相干的軌跡連成一條線。
    if (msg->track_id != current_track_id_) {
      trail_.clear();
      current_track_id_ = msg->track_id;
    }

    trail_.push_back(msg->position);
    while (trail_.size() > trail_length_) {
      trail_.pop_front();
    }

    visualization_msgs::msg::MarkerArray arr;

    visualization_msgs::msg::Marker line;
    line.header = msg->header;
    line.ns = "ball_trail";
    line.id = 0;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = 0.02;
    line.color.g = 1.0;
    line.color.a = 0.9;
    line.pose.orientation.w = 1.0;
    line.points.assign(trail_.begin(), trail_.end());
    arr.markers.push_back(line);

    visualization_msgs::msg::Marker ball;
    ball.header = msg->header;
    ball.ns = "ball";
    ball.id = 1;
    ball.type = visualization_msgs::msg::Marker::SPHERE;
    ball.action = visualization_msgs::msg::Marker::ADD;
    ball.pose.position = msg->position;
    ball.pose.orientation.w = 1.0;
    ball.scale.x = ball.scale.y = ball.scale.z = 0.08;
    // 預測點畫成紅色，讓遺失補插在畫面上一眼可辨。
    ball.color.r = msg->is_predicted ? 1.0 : 0.1;
    ball.color.g = msg->is_predicted ? 0.2 : 1.0;
    ball.color.a = 1.0;
    arr.markers.push_back(ball);

    marker_pub_->publish(arr);
  }

  std::size_t trail_length_{90};
  std::uint32_t current_track_id_{0};
  std::deque<geometry_msgs::msg::Point> trail_;

  rclcpp::Subscription<pickleball_msgs::msg::BallTrack>::SharedPtr track_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

}  // namespace pickleball_viz

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pickleball_viz::VizNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
