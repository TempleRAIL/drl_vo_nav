#!/usr/bin/env python
import roslib
roslib.load_manifest('yocs_velocity_smoother')
import rospy

import os
import sys
import time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
'''
  Varied translational input for a velocity smoother test.
'''

STATE_RAMP_UP = 0
STATE_RAMP_LEVEL = 1
STATE_RAMP_DOWN = 2
STATE_ZERO = 3
STATE_UP = 4
STATE_DOWN = 5
STATE_UP_AGAIN = 6
STATE_NOTHING = 7

def main():
    rospy.init_node("test_velocity_smoother_input")
    cmd_vel_publisher = rospy.Publisher("~cmd_vel", Twist)
    odom_publisher = rospy.Publisher("~odom", Odometry)
    param = {}
    param['velocity_maximum'] = rospy.get_param("~velocity_maximum", 0.50)
    param['ramp_increment'] = rospy.get_param("~ramp_increment", 0.02)
    rospy.loginfo("Test Input : ramp increment [%f]",param['ramp_increment'])
    param['ramp_decrement'] = rospy.get_param("~ramp_decrement", 0.02)
    rospy.loginfo("Test Input : ramp decrement [%f]",param['ramp_decrement'])
    cmd_vel = Twist()
    cmd_vel.linear.x = 0
    cmd_vel.linear.y = 0
    cmd_vel.linear.z = 0
    cmd_vel.angular.x = 0
    cmd_vel.angular.y = 0
    cmd_vel.angular.z = 0
    odom = Odometry()
    odom.header.frame_id = "base_link"
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    odom.pose.pose.position.z = 0.0
    odom.pose.pose.orientation.x = 0.0
    odom.pose.pose.orientation.y = 0.0
    odom.pose.pose.orientation.z = 0.0
    odom.pose.pose.orientation.w = 1.0

    odom.pose.covariance[0]  = 0.1
    odom.pose.covariance[7]  = 0.1
    odom.pose.covariance[35] = 0.2
    odom.pose.covariance[14] = 10.0
    odom.pose.covariance[21] = 10.0
    odom.pose.covariance[28] = 10.0

    odom.twist.twist.linear.x = 0.0
    odom.twist.twist.linear.y = 0.0
    odom.twist.twist.linear.z = 0.0
    odom.twist.twist.angular.x = 0.0
    odom.twist.twist.angular.y = 0.0
    odom.twist.twist.angular.z = 0.0
    state = STATE_RAMP_UP
    count = 0
    count_max = 100
    publish = True
    #period = 0.01
    timer = rospy.Rate(100) # 10hz
    rospy.loginfo("Test Input : STATE_RAMP_UP")
    while not rospy.is_shutdown():
        if state == STATE_RAMP_UP:
            cmd_vel.linear.x = cmd_vel.linear.x + param['ramp_increment']
            if cmd_vel.linear.x >= param['velocity_maximum']:
                state = STATE_RAMP_LEVEL
                count = 0
                rospy.loginfo("Test Input : STATE_RAMP_UP -> STATE_RAMP_LEVEL")
        elif state == STATE_RAMP_LEVEL:
            if count > count_max: # 0.5s
                state = STATE_RAMP_DOWN
                count = 0
                rospy.loginfo("Test Input : STATE_RAMP_LEVEL -> STATE_RAMP_DOWN")
            else:
                count = count + 1
        elif state == STATE_RAMP_DOWN:
            cmd_vel.linear.x = cmd_vel.linear.x - param['ramp_decrement']
            if cmd_vel.linear.x <= 0.0:
                cmd_vel.linear.x = 0.0
                state = STATE_ZERO
                count = 0
                rospy.loginfo("Test Input : STATE_RAMP_DOWN -> STATE_ZERO")
        elif state == STATE_ZERO:
            if count > count_max: # 0.5s
                state = STATE_UP
                cmd_vel.linear.x = param['velocity_maximum']
                count = 0
                rospy.loginfo("Test Input : STATE_ZERO -> STATE_UP")
            else:
                count = count + 1
        elif state == STATE_UP:
            if count > count_max: # 0.5s
                state = STATE_DOWN
                cmd_vel.linear.x = 0.0
                count = 0
                rospy.loginfo("Test Input : STATE_UP -> STATE_DOWN")
            else:
                count = count + 1
        elif state == STATE_DOWN:
            if count > count_max: # 0.5s
                #state = STATE_UP_AGAIN
                #cmd_vel.linear.x = param['velocity_maximum']
                #rospy.loginfo("Test Input : STATE_DOWN -> STATE_UP_AGAIN")
                state = STATE_RAMP_UP
                cmd_vel.linear.x = 0.0
                rospy.loginfo("Test Input : STATE_DOWN -> STATE_RAMP_UP")
                count = 0
            else:
                count = count + 1
        elif state == STATE_UP_AGAIN:
            if count > count_max: # 0.5s
                state = STATE_NOTHING
                count = 0
                publish = False
                rospy.loginfo("Test Input : STATE_UP_AGAIN -> STATE_NOTHING")
            else:
                count = count + 1
        elif state == STATE_NOTHING:
            if count > count_max: # 0.5s
                state = STATE_RAMP_UP
                cmd_vel.linear.x = 0.0
                count = 0
                publish = True
                rospy.loginfo("Test Input : STATE_NOTHING -> STATE_RAMP_UP")
            else:
                count = count + 1
        if publish:
            odom.twist.twist.linear.x = cmd_vel.linear.x
            cmd_vel_publisher.publish(cmd_vel)
        else:
            # How to fake it when it's not publishing a cmd velocity? Up to the velocity controller there
            odom.twist.twist.linear.x = cmd_vel.linear.x
        odom.header.stamp = rospy.Time().now()
        odom_publisher.publish(odom)
        timer.sleep()

if __name__ == "__main__":
  main()
