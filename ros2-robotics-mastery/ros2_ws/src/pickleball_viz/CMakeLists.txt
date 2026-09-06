cmake_minimum_required(VERSION 3.8)
project(pickleball_viz)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pickleball_msgs REQUIRED)

add_executable(viz_node src/viz_node.cpp)
ament_target_dependencies(viz_node
  rclcpp visualization_msgs geometry_msgs std_msgs pickleball_msgs)

install(TARGETS viz_node DESTINATION lib/${PROJECT_NAME})

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
endif()

ament_package()
