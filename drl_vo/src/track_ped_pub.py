#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish the ground truth of pedestrian kinematics from Gazebo.
#------------------------------------------------------------------------------

import rospy
import tf
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
from pedsim_msgs.msg  import TrackedPersons, TrackedPerson
from gazebo_msgs.srv import GetModelState
import numpy as np

class TrackPed:
    # Constructor
    def __init__(self):
        # Initialize ROS objects
        self.ped_sub = rospy.Subscriber('/pedsim_visualizer/tracked_persons', TrackedPersons, self.ped_callback)
        self.get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.track_ped_pub = rospy.Publisher('/track_ped', TrackedPersons, queue_size=10)
    
    def get_robot_states(self):
        """
        Obtaining robot current position (x, y, theta) 
        :param x x-position of the robot
        :param y y-position of the robot
        :param theta theta-position of the robot
        """
        robot = None
        # get robot and pedestrian states:
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            # get robot state:
            model_name = 'mobile_base' 
            relative_entity_name = 'world'
            robot = self.get_state_service(model_name, relative_entity_name)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed") 
        
        return robot
    
    # Callback function for the path subscriber
    def ped_callback(self, peds_msg):
        peds = peds_msg
        # get robot states:
        robot = self.get_robot_states()

        if(robot is not None):
            # get robot poses and velocities: 
            robot_pos = np.zeros(3)
            robot_pos[:2] = np.array([robot.pose.position.x, robot.pose.position.y])
            robot_quaternion = (
                                robot.pose.orientation.x,
                                robot.pose.orientation.y,
                                robot.pose.orientation.z,
                                robot.pose.orientation.w)
            (_,_, robot_pos[2]) = tf.transformations.euler_from_quaternion(robot_quaternion)
            robot_vel = np.array([robot.twist.linear.x, robot.twist.linear.y])
            #print(robot_vel)
            # homogeneous transformation matrix: map_T_robot
            map_R_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2])],
                                    [np.sin(robot_pos[2]),  np.cos(robot_pos[2])],
                                ])
            map_T_robot = np.array([[np.cos(robot_pos[2]), -np.sin(robot_pos[2]), robot_pos[0]],
                                    [np.sin(robot_pos[2]),  np.cos(robot_pos[2]), robot_pos[1]],
                                    [0, 0, 1]])
            # robot_T_map = (map_T_robot)^(-1)
            robot_R_map = np.linalg.inv(map_R_robot)
            robot_T_map = np.linalg.inv(map_T_robot)

            # get pedestrian poses and velocities:
            tracked_peds = TrackedPersons()
            #tracked_peds.header = peds.header
            tracked_peds.header.frame_id = 'base_footprint'
            tracked_peds.header.stamp = rospy.Time.now()
            for ped in peds.tracks:
                tracked_ped = TrackedPerson()
                # relative positions and velocities:
                ped_pos = np.array([ped.pose.pose.position.x, ped.pose.pose.position.y, 1])
                ped_vel = np.array([ped.twist.twist.linear.x, ped.twist.twist.linear.y])
                ped_pos_in_robot = np.matmul(robot_T_map, ped_pos.T)
                ped_vel_in_robot = np.matmul(robot_R_map, ped_vel.T) 
                # pedestrain message: 
                tracked_ped = ped
                tracked_ped.pose.pose.position.x = ped_pos_in_robot[0]
                tracked_ped.pose.pose.position.y = ped_pos_in_robot[1]
                tracked_ped.twist.twist.linear.x = ped_vel_in_robot[0]
                tracked_ped.twist.twist.linear.y = ped_vel_in_robot[1]
                tracked_peds.tracks.append(tracked_ped)
            # publish the pedestrains
            self.track_ped_pub.publish(tracked_peds)

 
if __name__ == '__main__':
    try:
        rospy.init_node('track_ped')
        tp = TrackPed()
        # spin() simply keeps python from exiting until this node is stopped    
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
