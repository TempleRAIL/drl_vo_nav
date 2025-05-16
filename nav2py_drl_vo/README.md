# nav2py_drl_vo

This repository provides the Nav2 implementation of DRL-VO navigation control policy, where the paper is ["DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles"](
https://doi.org/10.1109/TRO.2023.3257549
)([arXiv](https://arxiv.org/pdf/2301.06512.pdf)).
Video demos can be found at [multimedia demonstrations](https://www.youtube.com/watch?v=KneELRT8GzU&list=PLouWbAcP4zIvPgaARrV223lf2eiSR-eSS&index=2&ab_channel=PhilipDames). The original training and implementation code can be found in our [drl_vo_nav](https://github.com/TempleRAIL/drl_vo_nav.git) repository. 


## Requirements:
* Ubuntu 22.04
* ROS2-Humble
* Python 3.10.12
* venv

## 📌 Installation

### **1. Install ROS2 (Humble) and Navigation 2**
- [ROS2 Installation Guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [Nav2 Installation Guide](https://docs.nav2.org/getting_started/index.html)

### **2. Clone this repository into your workspace**
```bash
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws/src
git clone -b humble https://github.com/TempleRAIL/drl_vo_nav.git
```

### **3. Build the workspace**
```bash
cd ~/nav2_ws
source /opt/ros/humble/setup.bash
colcon build
```

### **4. Source the workspace**
```bash
source ~/nav2_ws/install/setup.bash
```

### **5. Run the Nav2 demo launch**
```bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False params_file:=$(pwd)/src/nav2py_drl_vo/tb3_drl_vo_nav2_params.yaml
```

## Citation
```
@article{xie2023drl,
  author={Xie, Zhanteng and Dames, Philip},
  journal={IEEE Transactions on Robotics}, 
  title={DRL-VO: Learning to Navigate Through Crowded Dynamic Scenes Using Velocity Obstacles}, 
  year={2023},
  volume={39},
  number={4},
  pages={2700-2719},
  doi={10.1109/TRO.2023.3257549}
}
```
