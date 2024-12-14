#!/usr/bin/env python
#
# revision history: xzt
#  20210604 (TE): first version
#
# usage:
#
# This script is to publish pedestrian kinematic maps and lidar historical map.
#------------------------------------------------------------------------------

import message_filters
import numpy as np
from random import choice
import rospy
import tf
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
#from pedsim_msgs.msg import TrackedPerson, TrackedPersons
from sensor_msgs.msg import Image, LaserScan
# custom define messages:
from cnn_msgs.msg import CNN_data
from zed_interfaces.msg import ObjectsStamped

# parameters:
NUM_TP = 10     # the number of timestamps
NUM_PEDS = 34+1 # the number of total pedestrians

class CnnData:
    # Constructor
    def __init__(self):
        # initialize data:  
        self.ped_pos_map = []
        self.scan = [] #np.zeros(720)
        self.scan_all = np.zeros(1080)
        self.goal_cart = np.zeros(2)
        self.goal_final_cart = np.zeros(2)
        self.vel = np.zeros(2)

        # timer:
        self.timer = None
        self.rate = 10  # 20 Hz velocity controller
        self.ts_cnt = 0  # maximum 10 timesteps

        # temporal data:
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)

        # initialize ROS objects
        self.ped_sub = rospy.Subscriber("zed_node/obj_det/objects", ObjectsStamped, self.ped_callback)
        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.goal_sub = rospy.Subscriber("cnn_goal", Point, self.goal_callback)
        self.vel_sub = rospy.Subscriber("jackal_velocity_controller/cmd_vel", Twist, self.vel_callback)
        self.cnn_data_pub = rospy.Publisher('cnn_data', CNN_data, queue_size=1, latch=False)

    
    # Callback function for the pedestrian subscriber
    def ped_callback(self, trackPed_msg):
        # get the pedstrain's position:
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        #self.ped_pos_map1_tmp = np.zeros((2,80,80))  # cartesian velocity map    
        if(len(trackPed_msg.objects) != 0):  # tracker results
            for ped in trackPed_msg.objects:
                #ped_id = ped.label_id 
                # create pedestrian's postion costmap: 10*10 m
                x = ped.position[0]
                y = ped.position[1]
                vx = ped.velocity[0]
                vy = ped.velocity[1]
                # 10m * 10m occupancy map:
                if(x >= 0 and x <= 10 and np.abs(y) <= 5):
                    # bin size: 0.125 m
                    c = int(np.floor(-(y-5)/0.125))
                    r = int(np.floor(x/0.125))

                    if(r == 80):
                        r = r - 1
                    if(c == 80):
                        c = c - 1
                    # cartesian velocity map
                    self.ped_pos_map_tmp[0,r,c] = vx
                    self.ped_pos_map_tmp[1,r,c] = vy
                
        # start the timer if this is the first path received
        if self.timer is None:
            self.start()

    # Callback function for the scan measurement subscriber
    def scan_callback(self, laserScan_msg):
        # get the laser scan data:
        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)
        scan_data[np.isnan(scan_data)] = 30.
        scan_data[np.isinf(scan_data)] = 30.
        self.scan_tmp = scan_data[180:900] #scan_data[180:900]
        self.scan_all_tmp = scan_data

        # start the timer if this is the first path received
        if self.timer is None:
            self.start()

    # Callback function for the current goal subscriber
    def goal_callback(self, goal_msg):
        # Cartesian coordinate:
        self.goal_cart = np.zeros(2)
        self.goal_cart[0] = goal_msg.x
        self.goal_cart[1] = goal_msg.y

    # Callback function for the velocity subscriber
    def vel_callback(self, vel_msg):
        self.vel = np.zeros(2)
        self.vel[0] = vel_msg.linear.x
        self.vel[1] = vel_msg.angular.z


    # Start the timer that calculates command velocities
    def start(self):
        # initialize timer for controller update
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)

     # function that runs every time the timer finishes to ensure that velocity commands are sent regularly
    def timer_callback(self, event):  
        # generate the trajectory of pedstrians:
        self.ped_pos_map = self.ped_pos_map_tmp
        self.scan.append(self.scan_tmp.tolist())
        self.scan_all = self.scan_all_tmp

        self.ts_cnt = self.ts_cnt + 1
        if(self.ts_cnt == NUM_TP): 
            # publish cnn data:
            cnn_data = CNN_data()
            cnn_data.ped_pos_map = [float(val) for sublist in self.ped_pos_map for subb in sublist for val in subb]
            cnn_data.scan = [float(val) for sublist in self.scan for val in sublist]
            cnn_data.scan_all = self.scan_all 
            cnn_data.goal_cart = self.goal_cart
            cnn_data.vel = self.vel
            self.cnn_data_pub.publish(cnn_data)

            # reset the position data list:
            self.ts_cnt = NUM_TP-1
            self.scan = self.scan[1:NUM_TP]

if __name__ == '__main__':
    try:
        rospy.init_node('cnn_data')
        CnnData()
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
