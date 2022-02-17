#!/usr/bin/env python

import cv2
import message_filters
import numpy as np
from random import choice
import rospy
import tf
from cnn_msgs.msg import CNN_data
from cv_bridge import CvBridge, CvBridgeError
# custom define messages:
from darknet_ros_msgs.msg import BoundingBoxes
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from pedsim_msgs.msg import TrackedPerson, TrackedPersons
from scipy.optimize import linprog
from sensor_msgs.msg import Image, LaserScan
from mht_msgs.msg import TrackingHistory
from mht_msgs.msg import TrackingResults


# parameters:
NUM_TP = 10     # the number of timestamps
NUM_PEDS = 34+1 # the number of total pedestrians

class CnnData:
    # cnn data:
    ped_pos_map = None
    #ped_pos_map1 = None

    scan = None	  # 720 range data from the laser scan
    scan_all = None	  # 1080 range data from the laser scan
    # image = None  # image data from the zed camera
    image_gray = None  # image data from the zed camera
    depth = None  # depth image data from the zed camera
    goal_cart = None   # current goal in robot frame
    #goal_polar = None   # current goal in robot frame
    goal_final_cart = None   # final goal in robot frame
    goal_final_polar = None   # final goal in robot frame

    vel  = None   # current velocity in robot frame

    # temproal data:
    vel_cart = None
    #ped_pos_cart_tmp = None
    ped_pos_map_tmp = None
    ped_pos_map1_tmp = None

    scan_tmp = None
    scan_all_tmp = None
    #depth_tmp = None
    bridge = None

    # ROS objects
    ped_sub = None       # subscriber to get the pedestrian's position
    scan_sub = None      # subscriber to get the laser scan data
    image_sub = None     # subscriber to get the image depth data
    depth_sub = None     # subscriber to get the image depth data
    goal_sub = None      # subscriber to get the goal
    final_goal_sub = None
    cnn_data_pub = None  # publisher to send the cnn data
    measurements = None  
    tf_listener = None # tf listener to get the pose of the robot

    timer = None         # timer to publish cnn data
    rate = None          # rate to publish cnn data [Hz]
    ts_cnt = None        # timestep counter

    # Constructor
    def __init__(self):
        # initialize data:  
        self.ped_pos_map = []
        #self.ped_pos_map1 = []
        
        self.scan = [] #np.zeros(720)
        self.scan_all = np.zeros(1080)
        #self.image = np.zeros((3,80,80))
        self.image_gray = np.zeros((80,80))
        self.depth = np.zeros((80,80))
        self.goal_cart = np.zeros(2)
        #self.goal_polar = np.zeros(2)
        self.goal_final_cart = np.zeros(2)
        self.goal_final_polar = np.zeros(3)
        self.vel = np.zeros(2)

        # temporal data:
        self.vel_cart = np.zeros((NUM_PEDS, 2))
        #self.ped_pos_cart_tmp = np.zeros((NUM_TP, NUM_PEDS, 2))
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        #self.ped_pos_map1_tmp = np.zeros((2,80,80))  # cartesian velocity map

        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)
        #self.depth_tmp = np.zeros((64,64))
        self.bridge = CvBridge()

        # initialize ROS objects
        #self.ped_sub = message_filters.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes)
        #self.ped_sub = rospy.Subscriber("/track_ped", TrackedPersons, self.ped_callback)
        self.ped_sub = rospy.Subscriber("/tracking_conclude", TrackingResults, self.ped_callback)

        self.scan_sub = message_filters.Subscriber("/scan", LaserScan)
        self.image_sub = message_filters.Subscriber('/zed/zed_node/rgb_raw/image_raw_color', Image)
        self.depth_sub = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image)
        self.measurements = message_filters.ApproximateTimeSynchronizer([self.scan_sub, self.depth_sub, self.image_sub], queue_size=5, slop=0.1)
        self.measurements.registerCallback(self.measurement_callback)

        #self.tf_listener = tf.TransformListener()
        #self.final_goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.final_goal_callback)
        self.final_goal_sub = rospy.Subscriber("/final_goal", Point, self.final_goal_callback)

        self.goal_sub = rospy.Subscriber("/cnn_goal", Point, self.goal_callback)
        self.vel_sub = rospy.Subscriber("/mobile_base/commands/velocity", Twist, self.vel_callback)

        self.cnn_data_pub = rospy.Publisher('/cnn_data', CNN_data, queue_size=1, latch=False)

        # timer:
        self.rate = 20  # 20 Hz velocity controller
        self.ts_cnt = 0  # maximum 7 timesteps
    
    # Callback function for the pedestrian subscriber
    def ped_callback(self, trackPed_msg):
        # get the pedstrain's position:
        self.ped_pos_map_tmp = np.zeros((2,80,80))  # cartesian velocity map
        #self.ped_pos_map1_tmp = np.zeros((2,80,80))  # cartesian velocity map    
        if(trackPed_msg.isempty == 0):  # tracker results
            for ped in trackPed_msg.trackinghistory:
                #ped_id = ped.track_id 
                # create pedestrian's postion costmap: 10*10 m
                x = ped.position[-1].x
                y = ped.position[-1].y
                vx = ped.velocity[-1].x
                vy = ped.velocity[-1].y
                # 20m * 20m occupancy map:
                if(x >= 0 and x <= 20 and np.abs(y) <= 10):
                    # bin size: 0.25 m
                    c = int(np.floor(-(y-10)/0.25))
                    r = int(np.floor(x/0.25))

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
    def measurement_callback(self, laserScan_msg, depth_msg, image_msg):
        # get the laser scan data:
        self.scan_tmp = np.zeros(720)
        self.scan_all_tmp = np.zeros(1080)
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)
        scan_data[np.isnan(scan_data)] = 0.
        scan_data[np.isinf(scan_data)] = 0.
        self.scan_tmp = scan_data[180:900]
        self.scan_all_tmp = scan_data

        # get the depth image data:
        self.depth = np.zeros((80,80))
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            print(e)
        depth_data = cv2.resize(depth_img, (80, 80))
        depth_data = np.array(depth_data, dtype=np.float32)
        depth_data[np.isnan(depth_data)] = 0. # represents the most distant possible depth value. 
        depth_data[np.isinf(depth_data)] = 0. # represents the most distant possible depth value.
        # normalize:
        #cv2.normalize(depth_data, depth_data, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.depth = depth_data

        # get the image data:
        #self.image = np.zeros((3,80,80))
        self.image_gray = np.zeros((80,80))
        try:
            image_img = self.bridge.imgmsg_to_cv2(image_msg,  "bgr8")
        except CvBridgeError as e:
            print(e)
        image_img_sized = cv2.resize(image_img, (80, 80))


        # get the gray image data:
        image_gray_data = cv2.cvtColor(image_img_sized, cv2.COLOR_BGR2GRAY)
        image_gray_data = np.array(image_gray_data, dtype=np.float32)
        # normalize:
        #cv2.normalize(image_gray_data, image_gray_data, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.image_gray = image_gray_data

        # start the timer if this is the first path received
        if self.timer is None:
            self.start()

    # Callback function for the current goal subscriber
    def goal_callback(self, goal_msg):
        # Cartesian coordinate:
        self.goal_cart = np.zeros(2)
        self.goal_cart[0] = goal_msg.x
        self.goal_cart[1] = goal_msg.y


    # Callback function for the final goal subscriber
    def final_goal_callback(self, final_goal_msg):
        # Cartesian coordinate:
        self.goal_final_cart = np.zeros(2)
        self.goal_final_cart[0] = final_goal_msg.x
        self.goal_final_cart[1] = final_goal_msg.y
        # Polar coordinate:
        self.goal_final_polar = np.zeros(3)
        self.goal_final_polar[0] = np.arctan2(final_goal_msg.y, final_goal_msg.x)
        self.goal_final_polar[1] = np.minimum(np.linalg.norm(self.goal_final_cart), 10.0)
        self.goal_final_polar[2] = final_goal_msg.z
          
        
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
        #self.ped_pos_map1 = self.ped_pos_map1_tmp

        self.scan.append(self.scan_tmp.tolist())
        self.scan_all = self.scan_all_tmp

        self.ts_cnt = self.ts_cnt + 1
        if(self.ts_cnt == NUM_TP): 
            # publish cnn data:
            cnn_data = CNN_data()
            cnn_data.ped_pos_map = [float(val) for sublist in self.ped_pos_map for subb in sublist for val in subb]
            #cnn_data.ped_pos_map1 = [float(val) for sublist in self.ped_pos_map1 for subb in sublist for val in subb]
            cnn_data.scan = [float(val) for sublist in self.scan for val in sublist]
            cnn_data.scan_all = self.scan_all 
            cnn_data.depth = [] #[float(val) for sublist in self.depth for val in sublist]
            #cnn_data.image = [float(val) for sublist in self.image for subb in sublist for val in subb]
            cnn_data.image_gray = [] #[float(val) for sublist in self.image_gray for val in sublist]
            cnn_data.goal_cart = self.goal_cart
            cnn_data.goal_final_polar = self.goal_final_polar
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
    
