#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to visualize the goal point in the rviz.
#------------------------------------------------------------------------------

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

def goal_callback(goal_msg):
    
    # initialize header and color
    h = Header()
    h.frame_id = "map"
    h.stamp = rospy.Time.now()
    
    # initialize goal marker message
    goal_marker = Marker()
    goal_marker.header = h
    goal_marker.type = Marker.SPHERE
    goal_marker.action = Marker.ADD
    goal_marker.pose = goal_msg.pose
    goal_marker.scale.x = 1.8
    goal_marker.scale.y = 1.8
    goal_marker.scale.z = 1.8
    goal_marker.color.r = 1.0
    goal_marker.color.g = 0.0
    goal_marker.color.b = 0.0
    goal_marker.color.a = 0.5 # set transparency
    
    goal_vis_pub.publish(goal_marker)

if __name__ == '__main__':
    try:
        rospy.init_node('goal_vis')
        goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, goal_callback)
        goal_vis_pub = rospy.Publisher('goal_markers', Marker, queue_size=1, latch=True)
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass