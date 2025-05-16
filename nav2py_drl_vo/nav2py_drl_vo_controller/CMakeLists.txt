cmake_minimum_required(VERSION 3.8)
project(nav2py_drl_vo_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2py REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

find_package(ament_index_cpp REQUIRED)

find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(PROJECTFILE pyproject.toml)
nav2py_package(nav2py_drl_vo_controller)

include_directories(
  include
)

set(dependencies
  rclcpp
  pluginlib
  tf2
  geometry_msgs
  nav_msgs
  nav2_costmap_2d
  nav2_core
  nav2_util
  nav2py
  ament_index_cpp
  sensor_msgs   
)

add_library(nav2py_drl_vo_controller SHARED
        src/drl_vo_controller.cpp)

# prevent pluginlib from using boost
target_compile_definitions(nav2py_drl_vo_controller PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

ament_target_dependencies(nav2py_drl_vo_controller
  ${dependencies}
)

install(TARGETS nav2py_drl_vo_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(
  DIRECTORY model
  DESTINATION venv/share/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_export_include_directories(include)
ament_export_libraries(nav2py_drl_vo_controller)
ament_export_dependencies(${dependencies})

pluginlib_export_plugin_description_file(nav2_core nav2py_drl_vo_controller.xml)

ament_package()
