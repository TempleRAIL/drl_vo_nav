#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage: python drl_vo_inference.py
#
# This script is the inference code of the DRL-VO policy.
#------------------------------------------------------------------------------

# import modules
#
import sys
import os

# ros:
import rospy
import tf
import numpy as np
import message_filters 

# custom define messages:
from sensor_msgs.msg import LaserScan
from cnn_msgs.msg import CNN_data
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from stable_baselines3 import PPO
from custom_cnn_full import *


#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------


# for reproducibility, we seed the rng
#       
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------
class DrlInference:
    # Constructor
    def __init__(self):
        # initialize data:  
        self.ped_pos = [] #np.ones((3, 20))*20.
        self.scan = [] #np.zeros((3, 720))
        self.goal = [] #np.zeros((3, 2))
        self.vx = 0
        self.wz = 0
        self.model = None

        # load model:
        model_file = rospy.get_param('~model_file', "./model/drl_vo.zip")
        self.model = PPO.load(model_file)
        print("Finish loading model.")

        # initialize ROS objects
        self.cnn_data_sub = rospy.Subscriber("/cnn_data", CNN_data, self.cnn_data_callback)
        self.cmd_vel_pub = rospy.Publisher('/drl_cmd_vel', Twist, queue_size=10, latch=False)


    # Callback function for the cnn_data subscriber
    def cnn_data_callback(self, cnn_data_msg):
        self.ped_pos = cnn_data_msg.ped_pos_map
        self.scan = cnn_data_msg.scan
        self.goal = cnn_data_msg.goal_cart
        cmd_vel = Twist()

        # minimum distance:
        scan = np.array(self.scan[-540:-180])
        scan = scan[scan!=0]
        if(scan.size!=0):
            min_scan_dist = np.amin(scan)
        else:
            min_scan_dist = 10

        # if the goal is close to the robot:
        if(np.linalg.norm(self.goal) <= 0.9):  # goal margin
                cmd_vel.linear.x = 0
                cmd_vel.angular.z = 0
        elif(min_scan_dist <= 0.4): # obstacle margin
            cmd_vel.linear.x = 0
            cmd_vel.angular.z = 0.7
        else:
            # MaxAbsScaler:
            v_min = -2 
            v_max = 2 
            self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
            self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)

            # MaxAbsScaler:
            temp = np.array(self.scan, dtype=np.float32)
            scan_avg = np.zeros((20,80))
            for n in range(10):
                scan_tmp = temp[n*720:(n+1)*720]
                for i in range(80):
                    scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                    scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])
            
            scan_avg = scan_avg.reshape(1600)
            scan_avg_map = np.matlib.repmat(scan_avg,1,4)
            self.scan = scan_avg_map.reshape(6400)
            s_min = 0
            s_max = 30
            self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)
            
            # goal:
            # MaxAbsScaler:
            g_min = -2
            g_max = 2
            goal_orignal = np.array(self.goal, dtype=np.float32)
            self.goal = 2 * (goal_orignal - g_min) / (g_max - g_min) + (-1)
            #self.goal = self.goal.tolist()

            # observation:
            self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None) 

            #self.inference()
            action, _states = self.model.predict(self.observation)
            # calculate the goal velocity of the robot and send the command
            # MaxAbsScaler:
            vx_min = 0
            vx_max = 0.5
            vz_min = -2 # -0.7
            vz_max = 2 # 0.7
            cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min
            cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min
        

        if not np.isnan(cmd_vel.linear.x) and not np.isnan(cmd_vel.angular.z): # ensure data is valid
            self.cmd_vel_pub.publish(cmd_vel)


    #
    # end of function


# begin gracefully
#

if __name__ == '__main__':
    rospy.init_node('drl_inference')
    drl_infe = DrlInference()
    rospy.spin()

# end of file
