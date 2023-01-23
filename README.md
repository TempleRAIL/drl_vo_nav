# DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles

Implementation code for our paper ["DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles"](https://arxiv.org/pdf/2301.06512.pdf). 
This reporsitory includes the code of our DRL-VO control policy and its 3D human-robot interaction Gazebo simulator.
Video demos can be found at [multimedia demonstrations](https://www.youtube.com/watch?v=KneELRT8GzU&list=PLouWbAcP4zIvPgaARrV223lf2eiSR-eSS&index=2&ab_channel=PhilipDames).
Here are two GIFs showing the navigation performance of our DRL-VO control policy in the simulation and real world. 
* Simulation:
![simulation_demo](demo/1.simulation_demo.gif "simulation_demo") 
* Real world:
![hardware_demo](demo/2.hardware_demo.gif "hardware_demo") 

## Requirements:
* Ubuntu 20.04
* ROS-Noetic
* Python 3.7
* Stable-baseline 3
* Pytorch 1.7.1
* Tensorboard

## Installation:
This package requires the robot_gazebo, pedsim_ros_with_gazebo, and turtlebot2 packages.
### Independent installation:
```
cd ~
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src
git clone https://github.com/TempleRAIL/robot_gazebo.git
git clone https://github.com/TempleRAIL/pedsim_ros_with_gazebo.git
git clone https://github.com/TempleRAIL/drl_vo_nav.git
wget https://raw.githubusercontent.com/zzuxzt/turtlebot2_noetic_packages/master/turtlebot2_noetic_install.sh 
sudo sh turtlebot2_noetic_install.sh 
cd ..
catkin_make
```

### Using singularity container: all required packages are installed


## Usage:
*  train:
```
roslaunch drl_vo_nav drl_vo_nav_train.launch
```
*  inference (navigation):
```
roslaunch drl_vo_nav drl_vo_nav.launch
```

## Citation
```
@article{xie2023drl,
  title={DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles},
  author={Xie, Zhanteng and Dames, Philip},
  journal={arXiv preprint arXiv:2301.06512},
  year={2023}
}

```
