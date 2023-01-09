#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish control commands.
#------------------------------------------------------------------------------

import rospy
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
import numpy as np

class VelSwitch:
    # velocities:
    cmd_vel_vx = None
    cmd_vel_wz = None

    # ROS objects
    drl_vel_sub = None # subscriber to get the command veloctiy of human planner
    cmd_vel_pub = None # publisher to send the robot velocity in gazebo frame
    timer = None       # timer to publish cmd vel
    
    rate = None
    
    # Constructor
    def __init__(self):
        # Initialize velocities:
        self.drl_vel = Twist()
        self.cmd_vel_vx = 0.
        self.cmd_vel_wz = 0.
        # Initialize ROS objects
        self.drl_vel_sub = rospy.Subscriber('/drl_cmd_vel', Twist, self.drl_callback)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1, latch=False)
        # timer:
        self.rate = 80  # 20 Hz velocity sampling


    # Get the command velocity from drl planner:
    def drl_callback(self, drl_vel_msg):
        self.drl_vel = drl_vel_msg
        # start the timer if this is the first command velocity received
        #if self.timer is None:
        #    self.start()

        cmd_vel = Twist()
        # get the cmd vel:
        self.cmd_vel_vx = self.drl_vel.linear.x
        self.cmd_vel_wz = self.drl_vel.angular.z
        
        # publish cmd vel:
        cmd_vel.linear.x = self.cmd_vel_vx 
        cmd_vel.angular.z = self.cmd_vel_wz
        self.cmd_vel_pub.publish(cmd_vel)

 
if __name__ == '__main__':
    try:
        rospy.init_node('mix_cmd_vel')
        VelSwitch()
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
