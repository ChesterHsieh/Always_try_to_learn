"""Phase 0 離線骨幹：影片 → 偵測 → 追蹤。

用法：
    ros2 launch pickleball_perception phase0.launch.py \\
        video_path:=/projects/P7-pickleball-tracker/results/bags/clip.mp4
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    video_path = LaunchConfiguration("video_path")
    params = PathJoinSubstitution(
        [FindPackageShare("pickleball_perception"), "config", "phase0.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "video_path",
            description="影片檔路徑，或 GStreamer pipeline 字串",
        ),
        DeclareLaunchArgument(
            "loop", default_value="false", description="播完是否循環"
        ),
        Node(
            package="pickleball_perception",
            executable="camera_node",
            name="camera_node",
            parameters=[params, {
                "video_path": video_path,
                "loop": LaunchConfiguration("loop"),
            }],
            output="screen",
        ),
        Node(
            package="pickleball_perception",
            executable="ball_detector_node",
            name="ball_detector",
            parameters=[params],
            # camera_node 發在 ~/image_raw，即 /camera_node/image_raw
            remappings=[("image_raw", "/camera_node/image_raw")],
            output="screen",
        ),
        Node(
            package="pickleball_perception",
            executable="ball_tracker_node",
            name="ball_tracker",
            parameters=[params],
            remappings=[("detections", "/ball_detector/detections")],
            output="screen",
        ),
    ])
