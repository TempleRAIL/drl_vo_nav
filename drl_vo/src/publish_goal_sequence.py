#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish the goal points and evaluate the navigation performance.
#------------------------------------------------------------------------------

import rospy
import math

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

from math import hypot
from kobuki_msgs.msg import BumperEvent
from nav_msgs.msg import Odometry


GOAL_NUM = 3

class MoveBaseSeq():

    # metrics:
    # success number and time:
    success_num = None
    total_time = None
    start_time = None
    end_time = None

    # odometry:
    total_distance = None   # total trajectory length
    previous_x = None       # robot previous position: x 
    previous_y = None       # robot previous position: y
    odom_start = None       # odometry start flag

    def __init__(self):
        # initialize :
        # success number and time:
        self.success_num = 0
        self.total_time = 0
        self.start_time = 0
        self.end_time = 0
        # distance:
        self.total_distance = 0.
        self.previous_x = 0
        self.previous_y = 0
        self.odom_start = True
        # initialize node:
        rospy.init_node('move_base_sequence')
        points_seq = rospy.get_param('~p_seq')
        # Only yaw angle required (no ratotions around x and y axes) in deg:
        yaweulerangles_seq = rospy.get_param('~yea_seq')

        # bumper:
        self.bump_flag = False
        self.bumper_sub = rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bumper_callback)

        # odometry:
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

        #List of goal quaternions:
        quat_seq = list()
        #List of goal poses:
        self.pose_seq = list()
        self.goal_cnt = 0
        '''
        for yawangle in yaweulerangles_seq:
            #Unpacking the quaternion tuple and passing it as arguments to Quaternion message constructor
            quat_seq.append(Quaternion(*(quaternion_from_euler(0, 0, yawangle*math.pi/180, axes='sxyz'))))
        '''
        # Returns a list of lists [[point1], [point2],...[pointn]]
        n = GOAL_NUM
        points = [points_seq[i:i+n] for i in range(0, len(points_seq), n)]
        rospy.loginfo(str(points))
        for i in range(len(points)):
            #Unpacking the quaternion tuple and passing it as arguments to Quaternion message constructor
            quat_seq.append(Quaternion(*(quaternion_from_euler(0, 0, 0, axes='sxyz'))))
        for point in points:
            #Exploit n variable to cycle in quat_seq
            self.pose_seq.append(Pose(Point(*point),quat_seq[n-3]))
            n += 1
        #rospy.loginfo(str(self.pose_seq))

        #Create action client
        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        #wait = self.client.wait_for_server(rospy.Duration(5.0))
        wait = self.client.wait_for_server()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting goals achievements ...")
        self.movebase_client()

    # Callback function for the bumper subscriber
    def bumper_callback(self, bumper_msg):
        #bumper_msg.bumper: LEFT (0), CENTER (1), RIGHT (2)
        #bumper_msg.state: RELEASED(0), PRESSED(1)
        if(bumper_msg.state == BumperEvent.PRESSED):
            self.bump_flag = True
        else:
            pass
        rospy.loginfo("Bumper Event:" + str(bumper_msg.bumper))

     # Callback function for the odometry subscriber
    def odom_callback(self, odom_msg):
        if(self.odom_start):
            self.previous_x = odom_msg.pose.pose.position.x
            self.previous_y = odom_msg.pose.pose.position.y
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        d_increment = hypot((x - self.previous_x), (y - self.previous_y))
        self.total_distance = self.total_distance + d_increment
        #print("Total distance traveled is {:.4f} m".format(self.total_distance))
        self.previous_x = odom_msg.pose.pose.position.x
        self.previous_y = odom_msg.pose.pose.position.y
        self.odom_start = False

    # Callback functions for publishing goals:
    def active_cb(self):
        rospy.loginfo("Goal pose "+str(self.goal_cnt+1)+" is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        #rospy.loginfo("Feedback for goal "+str(self.goal_cnt)+": "+str(feedback))
        rospy.loginfo("Feedback for goal pose "+str(self.goal_cnt+1)+" received")

    def done_cb(self, status, result):
        self.goal_cnt += 1
        # Reference for terminal status values: http://docs.ros.org/diamondback/api/actionlib_msgs/html/msg/GoalStatus.html
        if status == 2:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" received a cancel request after it started executing, completed execution!")
            # total success number:
            rospy.loginfo("Success Number: " + str(self.success_num))
            # total running time:
            rospy.loginfo("Total Running Time: " + str(self.total_time) + " secs") 
            # total distance:
            rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m") 
            #status = 3

        if status == 3:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" reached") 
            # success number:
            if(self.bump_flag == False):  # no collision
                self.success_num += 1
            #rospy.loginfo("Success Number: " + str(self.success_num))
            # reset bumpe_flag:
            self.bump_flag = False

            # running time: 
            self.end_time = rospy.get_time()
            rospy.loginfo("Start Time: " + str(self.start_time) + " secs") 
            rospy.loginfo("End Time: " + str(self.end_time) + " secs") 
            self.total_time = self.end_time - self.start_time
            

            # total distance:
            #rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m")
            
            # total success number:
            rospy.loginfo("Success Number: " + str(self.success_num) + " in total number " + str(self.goal_cnt))
            # total running time:
            rospy.loginfo("Total Running Time: " + str(self.total_time) + " secs") 
            # total distance:
            rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m") 
            
            # send the next goal:
            if self.goal_cnt< len(self.pose_seq):
                next_goal = MoveBaseGoal()
                next_goal.target_pose.header.frame_id = "map"
                next_goal.target_pose.header.stamp = rospy.Time.now()
                next_goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
                rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
                self.client.send_goal(next_goal, self.done_cb, self.active_cb, self.feedback_cb) 
            else:
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return

        if status == 4:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" was aborted by the Action Server")
            rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" aborted, shutting down!")
            # total success number:
            rospy.loginfo("Success Number: " + str(self.success_num) + " in total number " + str(self.goal_cnt))
            # total running time:
            rospy.loginfo("Total Running Time: " + str(self.total_time) + " secs") 
            # total distance:
            rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m") 
            return

        if status == 5:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" has been rejected by the Action Server")
            rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" rejected, shutting down!")
            # total success number:
            rospy.loginfo("Success Number: " + str(self.success_num) + " in total number " + str(self.goal_cnt))
            # total running time:
            rospy.loginfo("Total Running Time: " + str(self.total_time) + " secs") 
            # total distance:
            rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m") 
            return

        if status == 8:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" received a cancel request before it started executing, successfully cancelled!")
            # total success number:
            rospy.loginfo("Success Number: " + str(self.success_num) + " in total number " + str(self.goal_cnt))
            # total running time:
            rospy.loginfo("Total Running Time: " + str(self.total_time) + " secs") 
            # total distance:
            rospy.loginfo("Total Trajectory Length: " + str(self.total_distance) + " m") 

    def movebase_client(self):
    #for pose in pose_seq:   
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now() 
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
      
        while(rospy.get_time() < 12.237): # lobby world 34 peds: 12.237 s; lobby world 50 peds and new world: 6.238 s; 
            # start time:
            self.start_time = rospy.get_time()
      
        self.start_time = rospy.get_time()
        rospy.loginfo("Start Time: " + str(self.start_time) + " secs")
        # send goal:
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
        rospy.spin()

if __name__ == '__main__':
    try:
        MoveBaseSeq()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")