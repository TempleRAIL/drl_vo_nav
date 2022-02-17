#!/usr/bin/env python

import rospy
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
import tf
from scipy.optimize import linprog
from geometry_msgs.msg import Point

import numpy as np

def robot_pose_pub():
    rospy.init_node('robot_pose', anonymous=True)
    tf_listener = tf.TransformListener()
    robot_pose_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=1)
    rate = rospy.Rate(30) # 10hz
    while not rospy.is_shutdown():
        trans = rot = None
        # look up the current pose of the base_footprint using the tf tree
        try:
            (trans,rot) = tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('Could not get robot pose')
            trans = list([-1,-1,-1])
            rot = list([-1,-1,-1,-1])
        # publish robot pose:
        rob_pos = PoseStamped()
        rob_pos.header.stamp = rospy.Time.now()
        rob_pos.header.frame_id = '/map'
        rob_pos.pose.position.x = trans[0]
        rob_pos.pose.position.y = trans[1]
        rob_pos.pose.position.z = trans[2]
        rob_pos.pose.orientation.x = rot[0]
        rob_pos.pose.orientation.y = rot[1]
        rob_pos.pose.orientation.z = rot[2]
        rob_pos.pose.orientation.w = rot[3]
        robot_pose_pub.publish(rob_pos)

        rate.sleep()

if __name__ == '__main__':
    try:
        robot_pose_pub()
    except rospy.ROSInterruptException:
        pass
