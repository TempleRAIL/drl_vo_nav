#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish the sub-goal point using the pure pursit algorithm.
#------------------------------------------------------------------------------

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
import tf
from geometry_msgs.msg import Point


import numpy as np
import threading

class PurePursuit:
    # parameters of the controller
    lookahead = None # lookahead distance [m]
    rate = None # rate to run controller [Hz]
    goal_margin = None # maximum distance to goal before stopping [m]
    
    # parameters of the robot
    wheel_base = None # distance between left and right wheels [m]
    wheel_radius = None # wheel radius [m]
    v_max = None # maximum linear velocity [m/s]
    w_max = None # maximum angular velocity [rad/s]
    
    # ROS objects
    goal_sub = None # subscriber to get the global goal
    path_sub = None # subscriber to get the global path
    tf_listener = None # tf listener to get the pose of the robot
    cmd_vel_pub = None # publisher to send the velocity commands
    timer = None # timer to compute velocity commands
    cnn_goal_pub = None
    final_goal_pub = None
    
    # data
    #end_goal_pos = None # store the end goal position
    #end_goal_rot = None # store the end goal rotation
    path = None # store the path to the goal
    lock = threading.Lock() # lock to keep data thread safe
    
    # Constructor
    def __init__(self):
        # initialize parameters
        self.lookahead = 2 #rospy.get_param('~lookahead', 5.0)
        self.rate = rospy.get_param('~rate', 20.)
        self.goal_margin = 0.9 #rospy.get_param('~goal_margin', 3.0)
        
        self.wheel_base = 0.23 #rospy.get_param('~wheel_base', 0.16)
        self.wheel_radius = 0.025 #rospy.get_param('~wheel_radius', 0.033)
        self.v_max = 0.5 #0.5 #rospy.get_param('~v_max', 0.22)
        self.w_max = 5 #5 #2 #rospy.get_param('~w_max', 2.84)
    
        # Initialize ROS objects
        #self.goal_sub = rospy.Subscriber("/move_base/current_goal", PoseStamped, self.goal_callback)
        self.path_sub = rospy.Subscriber('path', Path, self.path_callback)
        self.tf_listener = tf.TransformListener()
        #self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.cnn_goal_pub = rospy.Publisher('cnn_goal', Point, queue_size=1)#, latch=True)
        self.final_goal_pub = rospy.Publisher('final_goal', Point, queue_size=1)#, latch=True)
    
    # Callback function for the path subscriber
    def path_callback(self, msg):
        rospy.logdebug('PurePursuit: Got path')
        # lock this data to ensure that it is not changed while other processes are using it
        self.lock.acquire()
        self.path = msg # store the path in the class member
        self.lock.release()
        # start the timer if this is the first path received
        if self.timer is None:
            self.start()

        
    # Start the timer that calculates command velocities
    def start(self):
        # initialize timer for controller update
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)
    
    # Get the current pose of the robot from the tf tree
    def get_current_pose(self):
        trans = rot = None
        # look up the current pose of the base_footprint using the tf tree
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('Could not get robot pose')
            return (np.array([np.nan, np.nan]), np.nan)
        x = np.array([trans[0], trans[1]])
        (roll, pitch, theta) = tf.transformations.euler_from_quaternion(rot)
        rospy.logdebug("x = {}, y = {}, theta = {}".format(x[0], x[1], theta))
        
        return (x, theta)
    
    # Find the closest point on the current path to the point x
    # Inputs: 
    #   x = numpy array with 2 elements (x and y position of robot)
    #   seg = optional argument that selects which segment of the path to compute the closest point on
    # Outputs:
    #   pt_min = closest point on the path to x
    #   dist_min = distance from the closest point to x
    #   seg_min = index of closest segment to x
    def find_closest_point(self, x, seg=-1):
        # initialize return values
        pt_min = np.array([np.nan, np.nan])
        dist_min = np.inf
        seg_min = -1
        
        # check if path has been received yet
        if self.path is None:
            rospy.logwarn('Pure Pursuit: No path received yet')
            return (pt_min, dist_min, seg_min)
        
        ##### YOUR CODE STARTS HERE #####
        if seg == -1:
            # find closest point on entire pathd
            for i in range(len(self.path.poses) - 1): # gets total number of segments and iterates over them all
                (pt, dist, s) = self.find_closest_point(x, i) # find the closest point to the robot on segment i
                if dist < dist_min: # if new point is closer than the previous best, keep it as the new best point
                    pt_min = pt
                    dist_min = dist
                    seg_min = s
        else:
            # find closest point on segment seg
            # extract the start and end of segment seg from the path
            p_start = np.array([self.path.poses[seg].pose.position.x, self.path.poses[seg].pose.position.y])
            p_end = np.array([self.path.poses[seg+1].pose.position.x, self.path.poses[seg+1].pose.position.y])

            # calculate the unit direction vector and segment length
            v = p_end - p_start
            length_seg = np.linalg.norm(v)
            v = v / length_seg

            # calculate projected distance
            dist_projected = np.dot(x - p_start, v)

            # find closest point on the line segment to x
            if dist_projected < 0.:
                pt_min = p_start
            elif dist_projected > length_seg:
                pt_min = p_end
            else:
                pt_min = p_start + dist_projected * v

            # calculate other outputs
            dist_min = np.linalg.norm(pt_min - x)
            seg_min = seg
            
        ##### YOUR CODE ENDS HERE #####
        return (pt_min, dist_min, seg_min)
    
    # Find the goal point to drive the robot towards
    # Inputs: 
    #   x = numpy array with 2 elements (x and y position of robot)
    #   pt, dist, seg = outputs of find_closest_point
    # Outputs:
    #   goal = numpy array with 2 elements (x and y position of goal)
    def find_goal(self, x, pt, dist, seg):
        goal = None
        end_goal_pos = None
        end_goal_rot = None
        if dist > self.lookahead:
            # if further than lookahead from the path, drive towards the path
            goal = pt
        else:
            ##### YOUR CODE STARTS HERE #####
            seg_max = len(self.path.poses) - 2
            # extract the end of segment seg from the path
            p_end = np.array([self.path.poses[seg+1].pose.position.x, self.path.poses[seg+1].pose.position.y])
            # calculate the distance from x to p_end:
            dist_end = np.linalg.norm(x - p_end) 

            # start from the nearest segment and iterate forward until you find either the last segment or a segment that leaves the lookahead circle
            while(dist_end < self.lookahead and seg < seg_max):
                seg = seg + 1
                # extract the end of segment seg from the path
                p_end = np.array([self.path.poses[seg+1].pose.position.x, self.path.poses[seg+1].pose.position.y])
                # calculate the distance from x to p_end:
                dist_end = np.linalg.norm(x - p_end)

            # if searched the whole path, set the goal as the end of the path
            if(dist_end < self.lookahead): 
                pt = np.array([self.path.poses[seg_max+1].pose.position.x, self.path.poses[seg_max+1].pose.position.y])
            # if found a segment that leaves the circle, find the intersection with the circle
            else: 
                # find the closest point:
                (pt, dist, seg) = self.find_closest_point(x, seg)
                # extract the start and end of segment seg from the path
                p_start = np.array([self.path.poses[seg].pose.position.x, self.path.poses[seg].pose.position.y])
                p_end = np.array([self.path.poses[seg+1].pose.position.x, self.path.poses[seg+1].pose.position.y])
                # calculate the unit direction vector and segment length
                v = p_end - p_start
                length_seg = np.linalg.norm(v)
                v = v / length_seg
                # calculate projected distance:
                dist_projected_x = np.dot(x - pt, v)
                dist_projected_y = np.linalg.norm(np.cross(x - pt, v))
                pt = pt + (np.sqrt(self.lookahead**2 - dist_projected_y**2) + dist_projected_x)*v

            goal = pt  
            ##### YOUR CODE ENDS HERE #####
            
        end_goal_pos = [self.path.poses[-1].pose.position.x, self.path.poses[-1].pose.position.y]
        end_goal_rot = [self.path.poses[-1].pose.orientation.x, self.path.poses[-1].pose.orientation.y, \
                            self.path.poses[-1].pose.orientation.z, self.path.poses[-1].pose.orientation.w,] 

        return (goal, end_goal_pos, end_goal_rot)

    
    # function that runs every time the timer finishes to ensure that velocity commands are sent regularly
    def timer_callback(self, event):    
        # lock the path to ensure it is not updated during processing
        self.lock.acquire()
        try:
            # get current pose
            # (x, theta) = self.get_current_pose()
            trans = rot = None
            # look up the current pose of the base_footprint using the tf tree
            try:
                (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_footprint', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn('Could not get robot pose')
                return (np.array([np.nan, np.nan]), np.nan)
            x = np.array([trans[0], trans[1]])
            (roll, pitch, theta) = tf.transformations.euler_from_quaternion(rot)
            rospy.logdebug("x = {}, y = {}, theta = {}".format(x[0], x[1], theta))
            if np.isnan(x[0]): # ensure data is valid
                return
        
            # find the closest point
            (pt, dist, seg) = self.find_closest_point(x)
            if np.isnan(pt).any(): # ensure data is valid
                return
        
            # find the goal point
            (goal, end_goal_pos, end_goal_rot) = self.find_goal(x, pt, dist, seg)
            if goal is None or end_goal_pos is None: # ensure data is valid
                return
        finally:
            # ensure the lock is released
            self.lock.release()
        
        # transform goal to local coordinates
        ##### YOUR CODE STARTS HERE #####
        # homogeneous transformation matrix:
        map_T_robot = np.array([[np.cos(theta), -np.sin(theta), x[0]],
                                    [np.sin(theta), np.cos(theta), x[1]],
                                    [0, 0, 1]])

        goal = np.matmul(np.linalg.inv(map_T_robot), np.array([[goal[0]],[goal[1]],[1]])) #np.dot(np.linalg.inv(map_T_robot), np.array([goal[0], goal[1],1])) #
        goal = goal[0:2]
        ##### YOUR CODE ENDS HERE #####

        # final relative goal:
        relative_goal = np.matmul(np.linalg.inv(map_T_robot), np.array([[end_goal_pos[0]],[end_goal_pos[1]],[1]])) 
        # Compute the difference to the goal orientation
        orientation_to_target = tf.transformations.quaternion_multiply(end_goal_rot, \
                tf.transformations.quaternion_inverse(rot))
        yaw = tf.transformations.euler_from_quaternion(orientation_to_target)[2]     
	
        # publish the cnn goal:
        cnn_goal = Point()
        cnn_goal.x = goal[0]
        cnn_goal.y = goal[1]
        cnn_goal.z = 0
        if not np.isnan(cnn_goal.x) and not np.isnan(cnn_goal.y): # ensure data is valid
                self.cnn_goal_pub.publish(cnn_goal)

        # publish the final goal:
        final_goal = Point()
        final_goal.x = relative_goal[0]
        final_goal.y = relative_goal[1]
        final_goal.z = yaw
        if not np.isnan(final_goal.x) and not np.isnan(final_goal.y): # ensure data is valid
                self.final_goal_pub.publish(final_goal)
        
if __name__ == '__main__':
    try:
        rospy.init_node('pure_pursuit')
        PurePursuit()
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
