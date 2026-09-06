cmake_minimum_required(VERSION 3.8)
project(pickleball_perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)   # Humble/Jazzy 共同基準
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pickleball_msgs REQUIRED)

# ── 核心演算法：刻意不依賴 rclcpp，方便單元測試 ─────────────
add_library(ball_filter SHARED src/ball_filter.cpp)
target_include_directories(ball_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(ball_filter)

# ── 節點 ────────────────────────────────────────────────
add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node rclcpp sensor_msgs std_msgs cv_bridge OpenCV)

add_executable(ball_detector_node src/ball_detector_node.cpp)
ament_target_dependencies(ball_detector_node
  rclcpp sensor_msgs std_msgs cv_bridge OpenCV pickleball_msgs)

add_executable(ball_tracker_node src/ball_tracker_node.cpp)
target_link_libraries(ball_tracker_node ball_filter)
ament_target_dependencies(ball_tracker_node rclcpp std_msgs geometry_msgs pickleball_msgs)

install(TARGETS ball_filter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS camera_node ball_detector_node ball_tracker_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 個人學習專案不需要版權標頭，關掉 copyright 與 cpplint 的同名檢查。
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # launch 檔刻意用雙引號與中文 docstring，與 flake8/pep257 的預設偏好衝突，
  # 這些是風格偏好而非正確性問題，關掉以免淹沒真正的 lint 訊號。
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ball_filter test/test_ball_filter.cpp)
  target_link_libraries(test_ball_filter ball_filter)
endif()

ament_export_include_directories(include)
ament_export_libraries(ball_filter)
ament_package()
