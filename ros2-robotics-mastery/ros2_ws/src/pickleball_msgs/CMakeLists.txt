cmake_minimum_required(VERSION 3.8)
project(pickleball_msgs)

# Humble (GCC 11) 與 Jazzy (GCC 13) 皆支援 C++17。
# 刻意不要求 C++20：Humble 的預設編譯器對 C++20 支援不完整。
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # ── msg ──────────────────────────────────────────────
  "msg/BallDetection.msg"
  "msg/BallTrack.msg"
  "msg/BounceEvent.msg"
  "msg/CourtModel.msg"
  "msg/MatchState.msg"
  "msg/PlayerState.msg"
  "msg/Rally.msg"
  "msg/ShotEvent.msg"
  "msg/Stats.msg"
  # ── srv ──────────────────────────────────────────────
  "srv/Calibrate.srv"
  "srv/Challenge.srv"
  # ── action ───────────────────────────────────────────
  "action/ProcessSession.action"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 版權標頭檢查對個人學習專案是雜訊，關掉。
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
