# DRL-VO-NAVIGATION
---
Safe and fast navigation: DRL-VO control policy.

## Requirements:
Ubuntu 20.04
ROS-Noetic
Stable-baseline 3
Pytorch 1.7

## Preparation:
install the robot_gazebo, darknet_ros_with_gazebo, openmht, pedsim_ros_with_gazebo, and turtlebot packages

## Usage:
# train:
```
roslaunch drl_vo_nav drl_vo_nav_train.launch
```
# inference (navigation):
```
roslaunch drl_vo_nav drl_vo_nav.launch
```

