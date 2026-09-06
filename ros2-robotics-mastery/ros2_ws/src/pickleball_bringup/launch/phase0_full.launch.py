"""Phase 0 完整管線：影片 → 偵測 → 追蹤 → RViz 視覺化。

用法：
    ros2 launch pickleball_bringup phase0_full.launch.py \\
        video_path:=/projects/P7-pickleball-tracker/results/bags/clip.mp4
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    perception_launch = PathJoinSubstitution(
        [FindPackageShare("pickleball_perception"), "launch", "phase0.launch.py"]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("pickleball_bringup"), "rviz", "phase0.rviz"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("video_path", description="影片檔路徑"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument(
            "rviz", default_value="false",
            description="是否開 RViz；Mac 上通常改用 Foxglove，預設關閉",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([perception_launch]),
            launch_arguments={
                "video_path": LaunchConfiguration("video_path"),
                "loop": LaunchConfiguration("loop"),
            }.items(),
        ),
        Node(
            package="pickleball_viz",
            executable="viz_node",
            name="viz_node",
            remappings=[("tracks", "/ball_tracker/tracks")],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
