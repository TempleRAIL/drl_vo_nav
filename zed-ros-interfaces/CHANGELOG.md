LATEST CHANGES
==============

2023-09-14
----------
- add custom 'PosTrackStaus' message type
- add custom 'set_roi' service
- add custom 'reset-roi' service

Release v1.2
------------
- add custom PlaneStamped topic to publish extracted plane on click

Release v1.1
------------
- add sport class to OD service
- Compatible with ZED ROS Wrapper v3.7.x

Release v1.0
------------
- Created the new repository `zed-ros-interfaces` which contains the ROS package `zed_interfaces` defining the custom topics, services and actions used by the `zed-ros-wrapper`. This new package is useful to receive the topics from a ZED node on system where the `zed-ros-wrapper` package is not installed, e.g systems without CUDA support.
- Added new service `save_memory_map` to save the current area memory to a file to be used in future node running for relocation.
- Added new service `save_3d_map` to save the current 3D fused point cloud as OBJ or PLY file

