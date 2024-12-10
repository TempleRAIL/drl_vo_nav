cmake_minimum_required(VERSION 3.5)

project(zed_interfaces)

## Generate symbols for IDE indexer
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Fix for QtCretor
#list(APPEND CMAKE_PREFIX_PATH "/opt/ros/$ENV{ROS_DISTRO}/")

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  #actionlib_msgs
  message_generation
  shape_msgs
)

add_message_files( 
    DIRECTORY msg 
    FILES
    BoundingBox2Df.msg
    BoundingBox2Di.msg
    BoundingBox3D.msg
    Keypoint2Df.msg
    Keypoint2Di.msg
    Keypoint3D.msg
    Object.msg
    ObjectsStamped.msg
    Skeleton2D.msg
    Skeleton3D.msg
    RGBDSensors.msg
    PlaneStamped.msg
    PosTrackStatus.msg
  )

#add_action_files(DIRECTORY action FILES act.action)

add_service_files( 
    DIRECTORY srv 
    FILES
    set_pose.srv
    reset_odometry.srv
    reset_tracking.srv
    start_svo_recording.srv
    stop_svo_recording.srv
    start_remote_stream.srv
    stop_remote_stream.srv
    set_led_status.srv
    toggle_led.srv
    start_3d_mapping.srv
    stop_3d_mapping.srv
    save_3d_map.srv
    start_object_detection.srv
    stop_object_detection.srv
    save_area_memory.srv
    set_roi.srv
    reset_roi.srv
  )

generate_messages(
    DEPENDENCIES
    #actionlib_msgs
    std_msgs
    geometry_msgs
    sensor_msgs
    shape_msgs
)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS 
    message_generation 
    std_msgs 
    geometry_msgs 
    sensor_msgs
    #actionlib_msgs
)

###############################################################################

#Add all files in subdirectories of the project in
# a dummy_target so qtcreator have access to all files
FILE(GLOB_RECURSE all_files ${CMAKE_SOURCE_DIR}/*)
add_custom_target(all_files_${PROJECT_NAME} SOURCES ${all_files})


# Install meshes
install(DIRECTORY
  meshes
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
