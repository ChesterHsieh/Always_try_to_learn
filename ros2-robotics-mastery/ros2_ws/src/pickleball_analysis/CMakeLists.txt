cmake_minimum_required(VERSION 3.8)
project(pickleball_analysis)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pickleball_msgs REQUIRED)

# Phase 2 的 AI 計分狀態機。純邏輯、無 ROS 相依，方便用測試逼出規則錯誤。
add_library(score_machine SHARED src/score_machine.cpp)
target_include_directories(score_machine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS score_machine
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

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
  ament_add_gtest(test_score_machine test/test_score_machine.cpp)
  target_link_libraries(test_score_machine score_machine)
endif()

ament_export_include_directories(include)
ament_export_libraries(score_machine)
ament_package()
