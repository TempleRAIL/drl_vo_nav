<h1 align="center">
   <img src="./images/Picto+STEREOLABS_Black.jpg" alt="Stereolabs" title="Stereolabs" /><br \>
   ROS Interfaces
</h1>

The `zed-ros-interfaces` repository install the `zed_interfaces` ROS package which defines the custom topics, services and actions used by the [ZED ROS Wrapper](https://github.com/stereolabs/zed-ros-wrapper) to interface with ROS.

If you already installed the [ZED ROS Wrapper](https://github.com/stereolabs/zed-ros-wrapper) or you plan to install it on this machine, this package is not required because it is automatically integrated by `zed-ros-wrapper` as a git submodule to satisfy all the required dependencies.

You must instead install this package on a remote system that must retrieve the topics sent by the ZED Wrapper (e.g. the list of detected objects obtained with the Object Detection module) or call services and actions to control the status of the ZED Wrapper.

**Note:** this package does not require CUDA, hence it can be used to receive the ZED data also on machines not equipped with an Nvidia GPU.

### Prerequisites

- Ubuntu 20.04
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)

### Build the repository

The `zed_interfaces` is a catkin package. It depends on the following ROS packages:

- catkin
- std_msgs
- sensor_msgs
- actionlib_msgs
- geometry_msgs
- message_generation

Open a terminal, clone the repository, update the dependencies and build the packages:

```
  $ cd ~/catkin_ws/src
  $ git clone https://github.com/stereolabs/zed-ros-interfaces.git
  $ cd ../
  $ rosdep install --from-paths src --ignore-src -r -y
  $ catkin_make -DCMAKE_BUILD_TYPE=Release
  $ source ./devel/setup.bash
```

## Custom Topics

 - BoundingBox2Df
 - BoundingBox2Di
 - BoundingBox3D
 - Keypoint2Df
 - Keypoint2Di
 - Keypoint3D
 - Object
 - ObjectsStamped
 - RGBDSensors
 - Skeleton2D
 - Skeleton3D
 - PlaneStamped

You can get more information reading the [Stereolabs online documentation](https://www.stereolabs.com/docs/ros/zed-node/)

## Custom Services

 - reset_odometry
 - reset_tracking
 - set_led_status
 - set_pose
 - save_3d_map
 - save_area_memory
 - start_3d_mapping
 - start_object_detection
 - start_remote_stream
 - start_svo_recording
 - stop_3d_mapping
 - stop_object_detection
 - stop_remote_stream
 - stop_svo_recording
 - toggle_led

You can get more information reading the [Stereolabs online documentation](https://www.stereolabs.com/docs/ros/zed-node/#services)
